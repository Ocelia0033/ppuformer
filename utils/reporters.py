
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ----------------------------------------------------------------------
# 画图通用工具
# ----------------------------------------------------------------------
#
# 全局硬约束（与论文图风格一致，绝不允许任何函数破坏）：
#   1. 所有 loss / MAE / MSE / R² 曲线一律使用线性坐标。
#      —— 本模块**任何位置**都不会调用 plt.yscale("log") /
#         ax.set_yscale("log")，也不会根据 loss 最大最小比值自动切到 log；
#   2. y 轴显示普通十进制小数（0.01 / 0.1 / 1 / 10 / 100 ...），
#      绝不使用 10^k / 1e-3 这种科学计数法，也不显示坐标轴偏移量；
#   3. 调参 / 模块筛查 / PSO 阶段：第二条曲线叫 "Validation"，title 含 "Validation"；
#      最终测试阶段：第二条曲线叫 "Test"，title 含 "Test"；
#   4. 若前几轮 loss 特别大导致后段看不清，**不要**改成 log，
#      而是再另存一张从 skip_epochs（默认 10）开始截取的局部线性图。
# ----------------------------------------------------------------------


def _apply_linear_y_axis(ax: plt.Axes, fmt: Optional[str] = None) -> None:
    """把 y 轴统一设成线性 + 普通数值显示，禁用科学计数法 / 偏移量。

    fmt = None（默认）：用 ScalarFormatter，根据数据范围自动选择普通小数显示，
        既不会丢精度也不会出现 10^k。
    fmt = "%.4f" 等：强制按指定 printf 格式显示。
    """
    ax.set_yscale("linear")
    try:
        ax.ticklabel_format(style="plain", axis="y", useOffset=False)
    except Exception:
        pass
    if fmt is None:
        sf = mticker.ScalarFormatter(useOffset=False, useMathText=False)
        sf.set_scientific(False)
        ax.yaxis.set_major_formatter(sf)
    else:
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(fmt))


def save_curve(
    epochs: Sequence[int],
    train_vals: Sequence[float],
    test_vals: Sequence[float],
    csv_path: Optional[str],
    png_path: str,
    ylabel: str,
    title: str,
    ylim: Optional[tuple] = None,
    series_label: str = "Test",
    y_fmt: Optional[str] = None,
) -> None:
    """
    单 metric 双曲线图（Train + {Test|Validation}），**强制线性 y 轴**。

    本函数永远不会启用 log y 轴；如果前几轮 loss 太大导致后段看不清，
    请额外调用 :func:`save_curve_after_epoch` 另存一张从 skip_epochs 起的局部线性图。

    参数
    ----
    series_label : "Test" / "Validation"
        第二条曲线和 csv 列名的前缀。调参 / 筛查 / PSO 阶段传 "Validation"；
        最终评估阶段传 "Test"（默认）。
    y_fmt : str | None
        y 轴普通数值格式串。默认 None = 由 ScalarFormatter 自适应选择普通小数显示，
        既不会出现 10^k 也不会丢失精度（推荐）。传 "%.4f" 等可强制固定小数位。
    """
    series_key = series_label.lower()

    if csv_path is not None:
        df = pd.DataFrame({
            "epoch": list(epochs),
            f"train_{ylabel.lower()}": list(train_vals),
            f"{series_key}_{ylabel.lower()}":  list(test_vals),
        })
        df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_vals, label=f"Train {ylabel}")
    ax.plot(epochs, test_vals,  label=f"{series_label} {ylabel}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _apply_linear_y_axis(ax, fmt=y_fmt)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def save_curve_after_epoch(
    epochs: Sequence[int],
    train_vals: Sequence[float],
    test_vals: Sequence[float],
    png_path: str,
    ylabel: str,
    title: str,
    skip_epochs: int = 10,
    series_label: str = "Test",
    y_fmt: Optional[str] = None,
) -> None:
    """
    从第 skip_epochs 个 epoch 开始截取的局部曲线，**强制线性坐标**。

    用于前几轮 loss 巨大导致整体图后段被压扁、看不出波动时的解决方案：
    我们**不**改用 log 坐标，而是另存一张去掉前 skip_epochs 个 epoch 的
    线性 zoom 图。y 轴同样禁用科学计数法 / 10^k。
    """
    n = len(epochs)
    start = min(skip_epochs, max(0, n - 2))
    e = list(epochs)[start:]
    t = list(train_vals)[start:]
    v = list(test_vals)[start:]
    if not e:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(e, t, label=f"Train {ylabel}")
    ax.plot(e, v, label=f"{series_label} {ylabel}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _apply_linear_y_axis(ax, fmt=y_fmt)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


def save_overall_indicators(
    csv_path: str,
    rmse: float,
    mae: float,
    r2: float,
    extras: Optional[dict] = None,
) -> None:
    rows = [
        {"metric": "RMSE", "value": rmse},
        {"metric": "MAE",  "value": mae},
        {"metric": "R2",   "value": r2},
    ]
    if extras:
        for k, v in extras.items():
            rows.append({"metric": k, "value": v})
    pd.DataFrame(rows).to_csv(csv_path, index=False)


# ----------------------------------------------------------------------
# Best / ALL / Best-Part
# ----------------------------------------------------------------------


def _best_sample_idx(all_preds: np.ndarray, all_targets: np.ndarray, pred_len: int) -> int:
    if pred_len > 1:
        best_idx = 0
        best_r2 = -np.inf
        for i in range(all_preds.shape[0]):
            r2 = r2_score(all_targets[i], all_preds[i])
            if r2 > best_r2:
                best_r2 = r2
                best_idx = i
        return best_idx
    errs = np.abs(all_preds[:, 0] - all_targets[:, 0])
    return int(np.argmin(errs))


def save_best(
    csv_path: str,
    png_path: str,
    all_preds: np.ndarray,
    all_targets: np.ndarray,
    pred_len: int,
    test_timestamps: np.ndarray,
    lookback_len: int,
) -> int:
    best_idx = _best_sample_idx(all_preds, all_targets, pred_len)
    actual = all_targets[best_idx]
    predicted = all_preds[best_idx]

    rows = []
    for j in range(pred_len):
        ts_idx = lookback_len + best_idx + j
        ts = test_timestamps[ts_idx] if ts_idx < len(test_timestamps) else f"step_{ts_idx}"
        rows.append({"timestamp": ts, "step": j + 1, "actual": actual[j], "predicted": predicted[j]})
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 6))
    steps = np.arange(1, pred_len + 1)
    if pred_len == 1:
        plt.plot(steps, actual, "o", label="Actual", markersize=10)
        plt.plot(steps, predicted, "x", label="Predicted", markersize=12)
    else:
        plt.plot(steps, actual, "-o", label="Actual")
        plt.plot(steps, predicted, "-x", label="Predicted")
    plt.xlabel("Step")
    plt.ylabel("Active Power")
    plt.title(f"Best Sample (idx={best_idx})  pred_len={pred_len}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    return best_idx


def save_best_part(
    csv_path: str,
    png_path: str,
    best_idx: int,
    all_preds: np.ndarray,
    all_targets: np.ndarray,
    raw_test_target: np.ndarray,
    test_timestamps: np.ndarray,
    lookback_len: int,
    context_window: int = 24,
) -> None:
    pred_pos = lookback_len + best_idx
    start = max(0, pred_pos - context_window)
    end = min(len(raw_test_target), pred_pos + context_window + 1)

    history_idx = np.arange(start, end)
    history_actual = raw_test_target[start:end]

    rows = []
    for k, idx in enumerate(history_idx):
        ts = test_timestamps[idx] if idx < len(test_timestamps) else f"step_{idx}"
        is_pred = idx == pred_pos
        rows.append({
            "timestamp": ts,
            "actual": history_actual[k],
            "predicted": all_preds[best_idx, 0] if is_pred else np.nan,
            "is_predicted_point": int(is_pred),
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(history_idx, history_actual, "-o", label="Actual", color="tab:blue", markersize=4)
    plt.plot([pred_pos], [all_preds[best_idx, 0]], "x", label="Predicted",
             color="tab:red", markersize=14, mew=3)
    plt.axvline(x=pred_pos, color="gray", linestyle="--", alpha=0.5)
    plt.xlabel("Time index (in test set)")
    plt.ylabel("Active Power")
    plt.title(f"Best-Part: 1-step prediction with ±{context_window}h context  (best_idx={best_idx})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()


def save_all(
    all_csv: str,
    scatter_png: str,
    error_png: str,
    all_preds: np.ndarray,
    all_targets: np.ndarray,
    pred_len: int,
    test_timestamps: np.ndarray,
    lookback_len: int,
    predictions_csv: Optional[str] = None,
) -> None:
    rows = []
    num_samples = all_preds.shape[0]
    for i in range(num_samples):
        for j in range(pred_len):
            ts_idx = lookback_len + i + j
            ts = test_timestamps[ts_idx] if ts_idx < len(test_timestamps) else f"step_{ts_idx}"
            rows.append({
                "DateTime": ts,
                "sample_idx": i,
                "step": j + 1,
                "actual": all_targets[i, j],
                "predicted": all_preds[i, j],
            })
    pd.DataFrame(rows).to_csv(all_csv, index=False)

    if predictions_csv is not None:
        simple_rows = []
        for i in range(num_samples):
            ts_idx = lookback_len + i
            ts = test_timestamps[ts_idx] if ts_idx < len(test_timestamps) else f"step_{ts_idx}"
            simple_rows.append({
                "datetime": ts,
                "true": all_targets[i, 0],
                "pred": all_preds[i, 0],
            })
        pd.DataFrame(simple_rows).to_csv(predictions_csv, index=False)

    preds_flat = all_preds.flatten()
    targets_flat = all_targets.flatten()

    plt.figure(figsize=(8, 8))
    plt.scatter(targets_flat, preds_flat, s=8, alpha=0.4)
    lo = float(min(targets_flat.min(), preds_flat.min()))
    hi = float(max(targets_flat.max(), preds_flat.max()))
    plt.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="y = x")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual (all test samples)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(scatter_png, dpi=150)
    plt.close()

    errors = targets_flat - preds_flat
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=80, edgecolor="black", alpha=0.75)
    plt.axvline(0, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Error  (Actual − Predicted)")
    plt.ylabel("Count")
    plt.title(
        f"Error Distribution   mean={errors.mean():.4f}, std={errors.std():.4f}"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(error_png, dpi=150)
    plt.close()


def save_prediction_curve(
    png_path: str,
    all_preds: np.ndarray,
    all_targets: np.ndarray,
    zoom_168h_png: Optional[str] = None,
    zoom_168h_csv: Optional[str] = None,
    timestamps: Optional[np.ndarray] = None,
    lookback_len: int = 0,
    phase: str = "test",
) -> None:
    """
    画 **step-1 rolling prediction** 曲线（在 pred_len>1 时只取每个滑窗的第 1 步）。

    phase : "test" / "val"
        - "test"：标题 "Test Step-1 Rolling Prediction Curve"
        - "val" ：标题 "Validation Step-1 Rolling Prediction Curve"

    168h zoom 图：
        - x 轴使用 **真实 datetime**（不再是 Hour=0~167，避免误判晚上有白天峰），
          每 24h 一个主刻度，并把日期标在刻度上；
        - 同时输出 CSV：datetime / hour_of_day / true / pred / is_daytime。
          is_daytime 由 true>0（夜间 PV 实际接近 0） 与 6<=hour<=18 双重判定，
          任一为 True 即视作白天，便于排查"夜间应为 0 却有峰"的目标列对齐错误。
    """
    series_label = "Validation" if phase == "val" else "Test"

    true_vals = all_targets[:, 0]
    pred_vals = all_preds[:, 0]

    fig, ax = plt.subplots(figsize=(14, 4), dpi=300)
    ax.plot(range(len(true_vals)), true_vals, color="#1f77b4", linewidth=0.7, label="True")
    ax.plot(range(len(pred_vals)), pred_vals, color="#ff7f0e", linewidth=0.7, alpha=0.85, label="Pred")
    ax.set_xlabel(f"{series_label} Sample Index")
    ax.set_ylabel("Active Power")
    ax.set_title(f"{series_label} Step-1 Rolling Prediction Curve")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.set_xlim(0, len(true_vals) - 1)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    if zoom_168h_png is not None and len(true_vals) >= 168:
        t = true_vals[:168]
        p_vals = pred_vals[:168]

        ts_dt_168 = None
        hour_of_day = None
        if timestamps is not None and len(timestamps) >= lookback_len + 168:
            ts_slice = timestamps[lookback_len:lookback_len + 168]
            try:
                ts_dt_168 = pd.to_datetime(ts_slice)
                hour_of_day = ts_dt_168.hour.to_numpy()
            except Exception:
                ts_dt_168 = None
                hour_of_day = None

        if zoom_168h_csv is not None and ts_dt_168 is not None:
            # is_daytime：白天双重判定（实际 AP>0 或 hour∈[6,18]），任一为 True 即视作白天
            is_day_by_value = t > 0
            is_day_by_hour = (hour_of_day >= 6) & (hour_of_day <= 18)
            is_daytime = (is_day_by_value | is_day_by_hour).astype(int)

            df_168 = pd.DataFrame({
                "datetime":    ts_dt_168.astype(str),
                "hour_of_day": hour_of_day,
                "true":        t,
                "pred":        p_vals,
                "is_daytime":  is_daytime,
            })
            df_168.to_csv(zoom_168h_csv, index=False)

        fig, ax = plt.subplots(figsize=(14, 4.5), dpi=300)
        if ts_dt_168 is not None:
            import matplotlib.dates as mdates
            x_dt = ts_dt_168.to_pydatetime()
            ax.plot(x_dt, t, color="#1f77b4", linewidth=1.2, label="True",
                    marker="o", markersize=1.5)
            ax.plot(x_dt, p_vals, color="#ff7f0e", linewidth=1.2, label="Pred",
                    marker="s", markersize=1.5, alpha=0.85)
            # 每 24h 一个主刻度 + 每 6h 一个次刻度，日期+小时显示
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=24))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
            ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
            ax.set_xlabel("Datetime")
            ax.set_xlim(x_dt[0], x_dt[-1])
        else:
            hours = np.arange(168)
            ax.plot(hours, t, color="#1f77b4", linewidth=1.2, label="True",
                    marker="o", markersize=1.5)
            ax.plot(hours, p_vals, color="#ff7f0e", linewidth=1.2, label="Pred",
                    marker="s", markersize=1.5, alpha=0.85)
            ax.set_xlabel("Hour (from validation start)")
            ax.set_xlim(0, 167)

        ax.set_ylabel("Active Power")
        title_ts = ""
        if ts_dt_168 is not None:
            ts0 = str(ts_dt_168[0])
            ts1 = str(ts_dt_168[-1])
            title_ts = f" ({ts0} ~ {ts1})"
        ax.set_title(f"{series_label} Step-1 Rolling Prediction Curve (168h){title_ts}")
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, linewidth=0.3, alpha=0.5)
        fig.tight_layout()
        fig.savefig(zoom_168h_png, dpi=300)
        plt.close(fig)


def save_raw_target_curve_168h(
    png_path: str,
    csv_path: str,
    raw_target: np.ndarray,
    timestamps: np.ndarray,
    lookback_len: int = 0,
    phase: str = "val",
    series_name: str = "Active Power",
) -> None:
    """
    画 validation/test 集**反归一化原始目标列**前 168 小时曲线（**与任何模型无关**）。

    用途：sanity check —— 夜间 PV 应当 ≈ 0；若 raw 曲线本身夜间出现峰，
    说明 dataset 时间戳 / target 列 / 预处理有 bug，需先修数据本身。

    输出
    ----
    png_path : 真 datetime x 轴的 168h 曲线
    csv_path : datetime / hour_of_day / true / is_daytime
    """
    series_label = "Validation" if phase == "val" else "Test"

    if len(raw_target) < 168:
        return
    t = np.asarray(raw_target[:168])

    ts_slice = timestamps[lookback_len:lookback_len + 168] \
        if len(timestamps) >= lookback_len + 168 else timestamps[:168]
    try:
        ts_dt = pd.to_datetime(ts_slice)
    except Exception:
        ts_dt = None

    if ts_dt is not None:
        hour_of_day = ts_dt.hour.to_numpy()
        is_day_by_value = t > 0
        is_day_by_hour = (hour_of_day >= 6) & (hour_of_day <= 18)
        is_daytime = (is_day_by_value | is_day_by_hour).astype(int)

        pd.DataFrame({
            "datetime":    ts_dt.astype(str),
            "hour_of_day": hour_of_day,
            "true":        t,
            "is_daytime":  is_daytime,
        }).to_csv(csv_path, index=False)
    else:
        pd.DataFrame({"true": t}).to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(14, 4.5), dpi=300)
    if ts_dt is not None:
        import matplotlib.dates as mdates
        x_dt = ts_dt.to_pydatetime()
        ax.plot(x_dt, t, color="#1f77b4", linewidth=1.2, marker="o", markersize=1.8,
                label=f"Raw {series_name} (no model)")
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=24))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
        ax.set_xlabel("Datetime")
        ax.set_xlim(x_dt[0], x_dt[-1])
        title_ts = f" ({ts_dt[0]} ~ {ts_dt[-1]})"
    else:
        ax.plot(np.arange(len(t)), t, color="#1f77b4", linewidth=1.2,
                marker="o", markersize=1.8, label=f"Raw {series_name}")
        ax.set_xlabel("Hour (from validation start)")
        ax.set_xlim(0, len(t) - 1)
        title_ts = ""

    ax.set_ylabel(series_name)
    ax.set_title(f"Raw {series_label} {series_name} (first 168h){title_ts}  (raw data, no model)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    _apply_linear_y_axis(ax)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


def write_full_report(
    paths: dict,
    history: dict,
    all_preds: np.ndarray,
    all_targets: np.ndarray,
    raw_test_target: np.ndarray,
    test_timestamps: np.ndarray,
    lookback_len: int,
    pred_len: int,
    loss_ylim: Optional[tuple] = None,  # 已弃用：完整图保持真实全范围，不再裁 y
    phase: str = "test",
    skip_epochs_for_zoom: int = 10,
) -> dict:
    """
    生成完整训练报告（曲线图、CSV、指标）。

    **绘图协议（硬约束）**：
        - 所有 loss / MAE / MSE / R² 曲线一律使用线性 y 轴；
        - y 轴显示普通十进制小数，绝不使用 10^k / 1e-k 科学计数法；
        - 前几轮 loss 太大导致后段看不清时，通过 ``skip_epochs_for_zoom``
          另存一张去掉前 N 个 epoch 的**线性** zoom 图，而**不**改 log 坐标。

    phase : "test" / "val"
        - "test"（默认）：最终评估阶段，第二条曲线叫 "Test"，标题含 "Test"。
        - "val"：调参 / 模块筛查 / PSO 阶段，第二条曲线叫 "Validation"，标题含 "Validation"。

    skip_epochs_for_zoom : int
        前几轮 loss 巨大时另存一张"从第 N 个 epoch 起"的局部图（**线性坐标**）。
        默认跳过 10 个 epoch。设为 0 关闭该图。

    loss_ylim : tuple | None
        **已弃用**，签名仅为向后兼容保留，函数内部不再使用。
        新协议下完整 loss 图固定展示真实全范围，不再依赖人工裁 y 轴；
        前几轮 loss 巨大时改用 skip_epochs_for_zoom 另存局部线性图。
    """
    del loss_ylim  # 显式忽略，避免静态检查告警；该参数已弃用
    epochs = history["epochs"]

    series_label = "Validation" if phase == "val" else "Test"
    series_lower = series_label.lower()

    train_loss = history["train_loss"]
    test_loss = history["test_loss"]

    save_curve(
        epochs, train_loss, test_loss,
        paths["loss_csv"], paths["loss_png"],
        ylabel="Loss",
        title=f"Training and {series_label} Loss Curve",
        ylim=None,
        series_label=series_label,
    )
    # 局部图：从第 N 个 epoch 起截取（线性坐标，**不**用 log），替代原 zoom-y 行为
    if skip_epochs_for_zoom > 0 and "loss_zoom_png" in paths and len(epochs) > skip_epochs_for_zoom:
        save_curve_after_epoch(
            epochs, train_loss, test_loss,
            paths["loss_zoom_png"],
            ylabel="Loss",
            title=f"Training and {series_label} Loss Curve (from epoch {skip_epochs_for_zoom})",
            skip_epochs=skip_epochs_for_zoom,
            series_label=series_label,
        )
    save_curve(
        epochs, history["train_mae"], history["test_mae"],
        paths["mae_csv"], paths["mae_png"],
        ylabel="MAE",
        title=f"Training and {series_label} MAE Curve",
        series_label=series_label,
    )
    save_curve(
        epochs, history["train_mse"], history["test_mse"],
        paths["mse_csv"], paths["mse_png"],
        ylabel="MSE",
        title=f"Training and {series_label} MSE Curve",
        series_label=series_label,
    )
    save_curve(
        epochs, history["train_r2"], history["test_r2"],
        paths["r2_csv"], paths["r2_png"],
        ylabel="R²",
        title=f"Training and {series_label} R² Curve",
        series_label=series_label,
    )

    preds_flat = all_preds.flatten()
    targets_flat = all_targets.flatten()
    rmse = float(np.sqrt(mean_squared_error(targets_flat, preds_flat)))
    mae = float(mean_absolute_error(targets_flat, preds_flat))
    r2 = float(r2_score(targets_flat, preds_flat))
    save_overall_indicators(paths["overall_csv"], rmse, mae, r2)

    best_idx = save_best(
        paths["best_csv"], paths["best_png"],
        all_preds, all_targets, pred_len, test_timestamps, lookback_len,
    )

    save_all(
        paths["all_csv"], paths["all_scatter_png"], paths["all_error_png"],
        all_preds, all_targets, pred_len, test_timestamps, lookback_len,
        predictions_csv=paths.get("predictions_csv"),
    )

    if "prediction_curve_png" in paths:
        save_prediction_curve(
            paths["prediction_curve_png"],
            all_preds, all_targets,
            zoom_168h_png=paths.get("prediction_curve_168h_png"),
            zoom_168h_csv=paths.get("prediction_curve_168h_csv"),
            timestamps=test_timestamps,
            lookback_len=lookback_len,
            phase=phase,
        )

    # raw sanity 图：仅在调用方提供路径时画一次（与模型无关，可全实验复用一份）
    if "raw_val_curve_168h_png" in paths and "raw_val_curve_168h_csv" in paths:
        save_raw_target_curve_168h(
            png_path=paths["raw_val_curve_168h_png"],
            csv_path=paths["raw_val_curve_168h_csv"],
            raw_target=raw_test_target,
            timestamps=test_timestamps,
            lookback_len=lookback_len,
            phase=phase,
        )

    if pred_len == 1:
        save_best_part(
            paths["best_part_csv"], paths["best_part_png"],
            best_idx, all_preds, all_targets,
            raw_test_target, test_timestamps, lookback_len,
        )

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "best_sample_idx": int(best_idx),
        "num_test_samples": int(all_preds.shape[0]),
    }
