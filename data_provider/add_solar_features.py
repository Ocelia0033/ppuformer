# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


#   lat = -25.2456°,  lon = 130.985°,  elevation ≈ 509 m
LATITUDE = -25.2456
LONGITUDE = 130.985
TIMEZONE_OFFSET_HOURS = 9.5    # ACST = UTC+9:30
STANDARD_LONGITUDE = TIMEZONE_OFFSET_HOURS * 15.0
SOLAR_CONSTANT = 1361.0
SITE_ALTITUDE_M = 509.0

LINKE_TURBIDITY_MONTHLY = np.array([
    3.8, 3.2, 3.5,   # Jan, Feb, Mar
    3.0, 2.8, 2.7,   # Apr, May, Jun
    2.7, 2.8, 3.2,   # Jul, Aug, Sep
    3.2, 3.2, 3.2,   # Oct, Nov, Dec
], dtype=np.float64)

SOLAR_COLS = [
    "theta_z", "gamma", "omega", "delta", "AM",
    "GHI_cs", "kt", "I0", "alpha_s", "DL", "tnoon", "Iday",
]


# =============================================================================
# =============================================================================

def _spencer_B_rad(day_of_year: np.ndarray) -> np.ndarray:
    return 2.0 * np.pi * (day_of_year - 1.0) / 365.0


def _solar_declination_deg(B: np.ndarray) -> np.ndarray:
    delta_rad = (
        0.006918
        - 0.399912 * np.cos(B)
        + 0.070257 * np.sin(B)
        - 0.006758 * np.cos(2.0 * B)
        + 0.000907 * np.sin(2.0 * B)
        - 0.002697 * np.cos(3.0 * B)
        + 0.001480 * np.sin(3.0 * B)
    )
    return np.rad2deg(delta_rad)


def _equation_of_time_min(B: np.ndarray) -> np.ndarray:
    return 229.18 * (
        0.000075
        + 0.001868 * np.cos(B)
        - 0.032077 * np.sin(B)
        - 0.014615 * np.cos(2.0 * B)
        - 0.040890 * np.sin(2.0 * B)
    )


def _solar_time_hours(local_clock_hours: np.ndarray, eot_min: np.ndarray) -> np.ndarray:
    return local_clock_hours + eot_min / 60.0 + (LONGITUDE - STANDARD_LONGITUDE) / 15.0


def _hour_angle_deg(local_solar_time_hours: np.ndarray) -> np.ndarray:
    return 15.0 * (local_solar_time_hours - 12.0)


def _solar_altitude_deg(lat_deg: float, delta_deg: np.ndarray, omega_deg: np.ndarray) -> np.ndarray:
    phi = np.deg2rad(lat_deg)
    delta = np.deg2rad(delta_deg)
    omega = np.deg2rad(omega_deg)
    sin_alpha = np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.cos(omega)
    sin_alpha = np.clip(sin_alpha, -1.0, 1.0)
    return np.rad2deg(np.arcsin(sin_alpha))


def _solar_azimuth_deg(lat_deg: float, delta_deg: np.ndarray, omega_deg: np.ndarray) -> np.ndarray:
    phi = np.deg2rad(lat_deg)
    delta = np.deg2rad(delta_deg)
    omega = np.deg2rad(omega_deg)
    num = np.sin(omega)
    den = np.sin(phi) * np.cos(omega) - np.cos(phi) * np.tan(delta)
    return np.rad2deg(np.arctan2(num, den))


def _air_mass(theta_z_deg: np.ndarray, alpha_deg: np.ndarray) -> np.ndarray:
    cos_z = np.cos(np.deg2rad(np.clip(theta_z_deg, 0.0, 96.0)))
    bracket = np.maximum(96.07995 - theta_z_deg, 0.1)
    am = 1.0 / (cos_z + 0.50572 * bracket ** -1.6364)
    return np.where(alpha_deg > 0.0, am, 40.0)


def _extraterrestrial_radiation(day_of_year: np.ndarray, theta_z_deg: np.ndarray) -> np.ndarray:
    eccentricity = 1.0 + 0.033 * np.cos(2.0 * np.pi * day_of_year / 365.0)
    cos_z = np.cos(np.deg2rad(theta_z_deg))
    cos_z = np.maximum(cos_z, 0.0)
    return SOLAR_CONSTANT * eccentricity * cos_z


def _clear_sky_ghi_ineichen_perez(day_of_year: np.ndarray, month: np.ndarray,
                                  theta_z_deg: np.ndarray, alpha_deg: np.ndarray,
                                  am: np.ndarray) -> np.ndarray:
    h = SITE_ALTITUDE_M
    cg1 = 5.09e-5 * h + 0.868
    cg2 = 3.92e-5 * h + 0.0387
    fh1 = np.exp(-h / 8000.0)
    fh2 = np.exp(-h / 1250.0)

    month_idx = (month.astype(np.int64) - 1) % 12
    tl = LINKE_TURBIDITY_MONTHLY[month_idx]

    eccentricity = 1.0 + 0.033 * np.cos(2.0 * np.pi * day_of_year / 365.0)
    i0_dni = SOLAR_CONSTANT * eccentricity

    cos_z = np.maximum(np.cos(np.deg2rad(theta_z_deg)), 0.0)
    am_safe = np.clip(am, 1.0, 40.0)

    ghi_cs = (
        cg1 * i0_dni * cos_z
        * np.exp(-cg2 * am_safe * (fh1 + fh2 * (tl - 1.0)))
        * np.exp(0.01 * am_safe ** 1.8)
    )
    ghi_cs = np.maximum(ghi_cs, 0.0)
    return np.where(alpha_deg > 0.0, ghi_cs, 0.0)


def _day_length_hours(lat_deg: float, delta_deg: np.ndarray) -> np.ndarray:
    phi = np.deg2rad(lat_deg)
    delta = np.deg2rad(delta_deg)
    arg = -np.tan(phi) * np.tan(delta)
    arg = np.clip(arg, -1.0, 1.0)
    return (2.0 / 15.0) * np.rad2deg(np.arccos(arg))


def _solar_noon_local_hour(eot_min: np.ndarray) -> np.ndarray:
    return 12.0 - eot_min / 60.0 - (LONGITUDE - STANDARD_LONGITUDE) / 15.0


# =============================================================================
# =============================================================================

def add_solar_features_to_df(df: pd.DataFrame, time_col: str = "date",
                             ghr_col: str = "Global_Horizontal_Radiation") -> pd.DataFrame:
    out = df.copy()

    ts = pd.to_datetime(out[time_col])
    day_of_year = ts.dt.dayofyear.values.astype(np.float64)
    month = ts.dt.month.values.astype(np.float64)
    local_hours = ts.dt.hour.values.astype(np.float64) + ts.dt.minute.values.astype(np.float64) / 60.0

    B = _spencer_B_rad(day_of_year)

    delta = _solar_declination_deg(B)
    eot_min = _equation_of_time_min(B)
    lst = _solar_time_hours(local_hours, eot_min)
    omega = _hour_angle_deg(lst)
    alpha = _solar_altitude_deg(LATITUDE, delta, omega)
    theta_z = 90.0 - alpha
    gamma = _solar_azimuth_deg(LATITUDE, delta, omega)

    am = _air_mass(theta_z, alpha)
    i0 = _extraterrestrial_radiation(day_of_year, theta_z)
    ghi_cs = _clear_sky_ghi_ineichen_perez(day_of_year, month, theta_z, alpha, am)

    ghr_obs = out[ghr_col].values.astype(np.float64)
    is_day = alpha > 0.0
    kt = np.where(is_day & (i0 > 1.0), ghr_obs / np.maximum(i0, 1.0), 0.0)
    kt = np.clip(kt, 0.0, 1.5)

    dl = _day_length_hours(LATITUDE, delta)
    tnoon = _solar_noon_local_hour(eot_min)
    iday = (alpha > 0.0).astype(np.float64)

    feature_dict = {
        "theta_z": theta_z, "gamma": gamma, "omega": omega, "delta": delta,
        "AM": am, "GHI_cs": ghi_cs, "kt": kt, "I0": i0,
        "alpha_s": alpha, "DL": dl, "tnoon": tnoon, "Iday": iday,
    }
    for col in SOLAR_COLS:
        out[col] = feature_dict[col].astype(np.float32)

    return out


def process_one_file(src_csv: str, dst_csv: str) -> None:
    df = pd.read_csv(src_csv)
    print(f"[读取] {src_csv}  shape={df.shape}")
    new_df = add_solar_features_to_df(df)
    print(f"[扩展] 加上 12 维物理特征，新 shape={new_df.shape}")
    os.makedirs(os.path.dirname(dst_csv) or ".", exist_ok=True)
    new_df.to_csv(dst_csv, index=False, encoding="utf-8")
    print(f"[保存] {dst_csv}")


def main():
    parser = argparse.ArgumentParser(description="把 5 列 PV CSV 扩展为 17 列（加 12 维太阳物理特征）。")
    parser.add_argument("--src", type=str, default=None, help="源 CSV 路径；为空则批处理 dataset/pv2017,2018,2019.csv")
    parser.add_argument("--dst", type=str, default=None, help="目标 CSV 路径（仅 --src 有值时使用）")
    args = parser.parse_args()

    if args.src is not None:
        dst = args.dst or args.src.replace(".csv", "_ext.csv")
        process_one_file(args.src, dst)
        return

    for year in (2017, 2018, 2019):
        src = os.path.join("dataset", f"pv{year}.csv")
        dst = os.path.join("dataset", f"pv{year}_ext.csv")
        if not os.path.exists(src):
            print(f"[跳过] {src} 不存在")
            continue
        process_one_file(src, dst)


if __name__ == "__main__":
    main()
