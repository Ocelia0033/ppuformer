# -*- coding: utf-8 -*-
"""
scripts/audit_run_result.py
============================
审计单个实验结果目录的完整性和质量。

Usage:
    python scripts/audit_run_result.py --run_dir results/iTransformer/pl4/y2017/train1
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


def check_file_exists(run_dir, filename):
    path = os.path.join(run_dir, filename)
    if not os.path.exists(path):
        return "MISSING", None
    if os.path.getsize(path) == 0:
        return "EMPTY", None
    return "OK", path


def audit_files(run_dir):
    required = [
        "args.json", "loss.csv", "loss.png", "loss_zoom.png",
        "mae.csv", "mse.csv", "R².csv", "Overall indicators.csv",
        "predictions.csv", "prediction_curve.png",
        "prediction_curve_168h.png", "prediction_curve_168h.csv",
    ]
    results = {}
    all_pass = True
    for f in required:
        status, path = check_file_exists(run_dir, f)
        results[f] = status
        if status != "OK":
            all_pass = False
    return results, all_pass


def audit_metrics(run_dir):
    path = os.path.join(run_dir, "Overall indicators.csv")
    if not os.path.exists(path):
        return None, "FAIL: file missing"
    df = pd.read_csv(path)
    info = {}

    # Format 1: columns named metric/value (rows are indicators)
    if "metric" in df.columns and "value" in df.columns:
        for _, row in df.iterrows():
            name = str(row["metric"]).strip()
            val = float(row["value"])
            if name.upper() == "RMSE":
                info["RMSE"] = val
            elif name.upper() == "MAE":
                info["MAE"] = val
            elif name.upper() in ("R2", "R²"):
                info["R2"] = val
    # Format 2: columns named RMSE, MAE, R2 directly
    else:
        for col in df.columns:
            c = col.strip()
            if c.upper() == "RMSE":
                info["RMSE"] = float(df[col].iloc[0])
            elif c.upper() == "MAE":
                info["MAE"] = float(df[col].iloc[0])
            elif c.upper() in ("R2", "R²"):
                info["R2"] = float(df[col].iloc[0])

    # Format 3: first col = metric name, second col = value (no header match)
    if not info and len(df.columns) >= 2:
        col0, col1 = df.columns[0], df.columns[1]
        for _, row in df.iterrows():
            name = str(row[col0]).strip().upper()
            try:
                val = float(row[col1])
            except (ValueError, TypeError):
                continue
            if name == "RMSE":
                info["RMSE"] = val
            elif name == "MAE":
                info["MAE"] = val
            elif name in ("R2", "R²"):
                info["R2"] = val

    warnings = []
    for key in ["RMSE", "MAE", "R2"]:
        if key not in info:
            warnings.append(f"WARN: {key} not found in Overall indicators.csv")

    return info, warnings if warnings else "OK"


def audit_loss(run_dir):
    path = os.path.join(run_dir, "loss.csv")
    if not os.path.exists(path):
        return {}, "FAIL: file missing"

    df = pd.read_csv(path)
    issues = []
    info = {}

    train_col = None
    test_col = None
    for c in df.columns:
        cl = c.lower().strip()
        if "train" in cl and "loss" in cl:
            train_col = c
        elif "test" in cl and "loss" in cl:
            test_col = c
        elif cl == "train_loss":
            train_col = c
        elif cl in ("test_loss", "val_loss"):
            test_col = c

    if train_col is None and test_col is None:
        if len(df.columns) >= 2:
            train_col = df.columns[0]
            test_col = df.columns[1]

    if train_col:
        train_loss = df[train_col].values
        info["first_train_loss"] = float(train_loss[0])
        info["last_train_loss"] = float(train_loss[-1])
        if np.any(np.isnan(train_loss)) or np.any(np.isinf(train_loss)):
            issues.append("FAIL: train_loss contains NaN/Inf")
        if train_loss[-1] > train_loss[0] * 1.5:
            issues.append("WARN: train_loss did not decrease overall")

    if test_col:
        test_loss = df[test_col].values
        info["first_test_loss"] = float(test_loss[0])
        info["last_test_loss"] = float(test_loss[-1])
        info["best_test_loss"] = float(np.nanmin(test_loss))
        if np.any(np.isnan(test_loss)) or np.any(np.isinf(test_loss)):
            issues.append("FAIL: test_loss contains NaN/Inf")

        n = len(test_loss)
        if n > 20:
            last_quarter = test_loss[3*n//4:]
            mid_quarter = test_loss[n//4:n//2]
            if np.nanmean(last_quarter) > np.nanmean(mid_quarter) * 1.3:
                issues.append("WARN: test_loss rises in later epochs — possible overfitting")

        if train_col:
            train_loss = df[train_col].values
            if n > 30:
                last_train = train_loss[-n//4:]
                last_test = test_loss[-n//4:]
                if np.nanmean(last_train) < np.nanmean(last_test) * 0.1:
                    if np.nanmean(last_test) > info["best_test_loss"] * 1.5:
                        issues.append("WARN: large train/test gap — generalization issue")

    return info, issues if issues else "OK"


def audit_r2(run_dir):
    path = os.path.join(run_dir, "R².csv")
    if not os.path.exists(path):
        return {}, "FAIL: file missing"

    df = pd.read_csv(path)
    issues = []
    info = {}

    test_col = None
    for c in df.columns:
        cl = c.lower().strip()
        if "test" in cl and "r" in cl:
            test_col = c
            break
    if test_col is None:
        for c in df.columns:
            if "r2" in c.lower() or "r²" in c.lower():
                test_col = c
                break
    if test_col is None and len(df.columns) >= 2:
        test_col = df.columns[-1]

    if test_col:
        r2 = df[test_col].values
        info["final_test_r2"] = float(r2[-1])
        info["best_test_r2"] = float(np.nanmax(r2))
        if info["best_test_r2"] < 0:
            issues.append("WARN: R² never positive — model worse than mean predictor")
        n = len(r2)
        if n > 20:
            std_last = np.nanstd(r2[-n//4:])
            if std_last > 0.1:
                issues.append("WARN: R² highly unstable in later epochs")

    return info, issues if issues else "OK"


def _find_pred_true_cols(df):
    pred_col = None
    true_col = None
    hour_col = None
    for c in df.columns:
        cl = c.lower().strip()
        if "pred" in cl:
            pred_col = c
        if "true" in cl or "actual" in cl or "target" in cl:
            true_col = c
        if "hour" in cl:
            hour_col = c
    return pred_col, true_col, hour_col


def _night_mask_from_df(df, hour_col=None):
    if hour_col is not None and hour_col in df.columns:
        hours = df[hour_col].values
        return (hours < 6) | (hours > 18)
    if "datetime" in df.columns:
        hours = pd.to_datetime(df["datetime"]).dt.hour.values
        return (hours < 6) | (hours > 18)
    if "is_daytime" in df.columns:
        return df["is_daytime"].values == 0
    return None


def audit_predictions(run_dir):
    """Adaptive negative-value and night-peak audit based on true data distribution."""
    issues = []
    info_notes = []
    info = {}

    preds = None
    trues = None
    night_mask = None
    daytime_true_max = None

    pred_path = os.path.join(run_dir, "predictions.csv")
    h168_path = os.path.join(run_dir, "prediction_curve_168h.csv")

    # --- Full-set true/pred stats (prefer predictions.csv) ---
    if os.path.exists(pred_path):
        df = pd.read_csv(pred_path)
        pred_col, true_col, _ = _find_pred_true_cols(df)
        if pred_col:
            preds = df[pred_col].values.astype(float)
            if np.any(np.isnan(preds)):
                issues.append("FAIL: predictions contain NaN")
            if np.any(np.isinf(preds)):
                issues.append("FAIL: predictions contain Inf")
        if true_col:
            trues = df[true_col].values.astype(float)

    # --- Night / daytime context (prefer 168h csv for hour alignment) ---
    if os.path.exists(h168_path):
        df168 = pd.read_csv(h168_path)
        p168, t168, h168 = _find_pred_true_cols(df168)
        night_mask_168 = _night_mask_from_df(df168, h168)

        if p168 and t168 and night_mask_168 is not None:
            preds168 = df168[p168].values.astype(float)
            trues168 = df168[t168].values.astype(float)
            day_mask = ~night_mask_168

            if np.any(day_mask):
                daytime_true_max = float(np.nanmax(trues168[day_mask]))
            else:
                daytime_true_max = float(np.nanmax(trues168))

            if np.any(night_mask_168):
                night_preds = preds168[night_mask_168]
                info["night_pred_max"] = float(np.nanmax(night_preds))

            # Use 168h for night thresholds if full-set not loaded
            if trues is None:
                trues = trues168
            if preds is None:
                preds = preds168
            night_mask = night_mask_168
            night_trues = trues168[night_mask_168] if np.any(night_mask_168) else trues168
        else:
            night_trues = None
    else:
        night_trues = None
        if preds is not None and trues is not None:
            df_full = pd.read_csv(pred_path)
            night_mask = _night_mask_from_df(df_full)
            if night_mask is not None and np.any(night_mask):
                night_trues = trues[night_mask]
                day_mask = ~night_mask
                if np.any(day_mask):
                    daytime_true_max = float(np.nanmax(trues[day_mask]))
                else:
                    daytime_true_max = float(np.nanmax(trues))
                info["night_pred_max"] = float(np.nanmax(preds[night_mask]))

    if preds is None or trues is None:
        return info, issues if issues else "OK", info_notes

    # --- Negative distribution stats ---
    true_min = float(np.nanmin(trues))
    pred_min = float(np.nanmin(preds))
    true_neg_ratio = float(np.mean(trues < 0))
    pred_neg_ratio = float(np.mean(preds < 0))
    neg_gap = pred_neg_ratio - true_neg_ratio

    info["true_min"] = true_min
    info["pred_min"] = pred_min
    info["true_negative_ratio"] = round(true_neg_ratio, 4)
    info["pred_negative_ratio"] = round(pred_neg_ratio, 4)
    info["negative_ratio_gap"] = round(neg_gap, 4)
    info["negative_pred_count"] = int(np.sum(preds < 0))

    margin = 0.01
    if night_trues is not None and len(night_trues) > 0:
        true_night_min = float(np.nanmin(night_trues))
        true_night_p01 = float(np.nanpercentile(night_trues, 1))
    else:
        true_night_min = true_min
        true_night_p01 = float(np.nanpercentile(trues[trues < 0], 1)) if np.any(trues < 0) else 0.0

    if daytime_true_max is None:
        daytime_true_max = float(np.nanmax(trues[trues > 0])) if np.any(trues > 0) else float(np.nanmax(trues))

    info["true_night_min"] = true_night_min
    info["true_night_p01"] = round(true_night_p01, 6)
    info["daytime_true_max"] = daytime_true_max

    large_negative_threshold = min(true_night_p01 - margin, -0.05 * daytime_true_max)
    extreme_negative_threshold = -0.20 * daytime_true_max
    info["large_negative_threshold"] = round(large_negative_threshold, 6)
    info["extreme_negative_threshold"] = round(extreme_negative_threshold, 6)

    # --- Adaptive negative rules ---
    has_extreme = pred_min < extreme_negative_threshold
    has_large = pred_min < large_negative_threshold
    has_ratio_gap = neg_gap > 0.10

    if has_extreme:
        issues.append(
            f"FAIL: extreme negative prediction "
            f"(pred_min={pred_min:.4f} < {extreme_negative_threshold:.4f})"
        )
    elif has_large:
        issues.append(
            f"WARN: large negative prediction beyond true data range "
            f"(pred_min={pred_min:.4f} < {large_negative_threshold:.4f}, "
            f"true_night_p01={true_night_p01:.4f})"
        )

    if has_ratio_gap:
        issues.append(
            f"WARN: pred negative ratio much higher than true "
            f"(pred={pred_neg_ratio*100:.1f}%, true={true_neg_ratio*100:.1f}%, gap={neg_gap*100:.1f}%)"
        )

    if not has_extreme and not has_large and not has_ratio_gap and pred_min < 0:
        info_notes.append(
            f"minor negative predictions within true data range "
            f"(pred_min={pred_min:.4f}, true_min={true_min:.4f}, "
            f"true_night_p01={true_night_p01:.4f}, "
            f"pred_neg={pred_neg_ratio*100:.1f}%, true_neg={true_neg_ratio*100:.1f}%)"
        )

    # --- Night positive peak (separate from negative check) ---
    night_pred_max = info.get("night_pred_max")
    if night_pred_max is not None and daytime_true_max > 0:
        ratio = night_pred_max / daytime_true_max
        if ratio > 0.4:
            issues.append(
                f"FAIL: night positive peak anomaly "
                f"(night_pred_max={night_pred_max:.3f} > 40% of daytime_true_max={daytime_true_max:.3f})"
            )
        elif ratio > 0.2:
            issues.append(
                f"WARN: night positive peak anomaly "
                f"(night_pred_max={night_pred_max:.3f} > 20% of daytime_true_max={daytime_true_max:.3f})"
            )

    status = issues if issues else "OK"
    return info, status, info_notes


def audit_time_shift(run_dir):
    h168_path = os.path.join(run_dir, "prediction_curve_168h.csv")
    if not os.path.exists(h168_path):
        return {}, "SKIP: no 168h csv"

    df = pd.read_csv(h168_path)
    pred_col = None
    true_col = None
    for c in df.columns:
        cl = c.lower().strip()
        if "pred" in cl:
            pred_col = c
        if "true" in cl or "actual" in cl or "target" in cl:
            true_col = c

    if pred_col is None or true_col is None:
        return {}, "SKIP: columns not found"

    preds = df[pred_col].values
    trues = df[true_col].values

    if len(preds) < 48:
        return {}, "SKIP: too few data points"

    corrs = {}
    for shift in range(-24, 25):
        if shift == 0:
            corrs[0] = float(np.corrcoef(preds, trues)[0, 1])
        elif shift > 0:
            corrs[shift] = float(np.corrcoef(preds[shift:], trues[:-shift])[0, 1])
        else:
            s = -shift
            corrs[shift] = float(np.corrcoef(preds[:-s], trues[s:])[0, 1])

    best_shift = max(corrs, key=corrs.get)
    corr_improvement = corrs[best_shift] - corrs[0]
    info = {
        "corr_shift0": round(corrs[0], 4),
        "best_shift": int(best_shift),
        "best_corr": round(corrs[best_shift], 4),
        "corr_improvement": round(corr_improvement, 4),
    }

    issues = []
    if best_shift != 0 and corr_improvement > 0.05:
        issues.append(
            f"WARN: better correlation at shift={best_shift} "
            f"(corr={corrs[best_shift]:.4f} vs shift0={corrs[0]:.4f}, "
            f"improvement={corr_improvement:.4f}) — possible time misalignment"
        )

    return info, issues if issues else "OK"


def main():
    parser = argparse.ArgumentParser(description="Audit a run result directory")
    parser.add_argument("--run_dir", required=True, help="Path to result directory")
    args = parser.parse_args()

    run_dir = args.run_dir
    if not os.path.isdir(run_dir):
        print(f"ERROR: {run_dir} is not a directory")
        sys.exit(1)

    print("="*60)
    print(f"  AUDIT: {run_dir}")
    print("="*60)

    summary = {"run_dir": run_dir, "checks": {}}
    all_issues = []

    # 1. File completeness
    print("\n[1] File Completeness")
    file_results, files_ok = audit_files(run_dir)
    for f, status in file_results.items():
        icon = "✓" if status == "OK" else "✗"
        print(f"  {icon} {f}: {status}")
        if status != "OK":
            all_issues.append(f"FAIL: {f} {status}")
    summary["checks"]["files"] = file_results

    # 2. Metrics
    print("\n[2] Overall Metrics")
    metrics, m_status = audit_metrics(run_dir)
    if metrics:
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                print(f"  {k} = {v:.6f}")
            else:
                print(f"  {k} = {v}")
        summary["checks"]["metrics"] = metrics
    if isinstance(m_status, list):
        for w in m_status:
            print(f"  {w}")
            all_issues.append(w)
    elif m_status != "OK":
        print(f"  {m_status}")
        all_issues.append(m_status)

    # 3. Loss curve
    print("\n[3] Loss Curve Audit")
    loss_info, loss_status = audit_loss(run_dir)
    if isinstance(loss_status, list):
        for iss in loss_status:
            print(f"  {iss}")
            all_issues.append(iss)
    else:
        print(f"  {loss_status}")
    for k, v in loss_info.items():
        print(f"  {k} = {v:.6f}")
    summary["checks"]["loss"] = loss_info

    # 4. R² curve
    print("\n[4] R² Curve Audit")
    r2_info, r2_status = audit_r2(run_dir)
    if isinstance(r2_status, list):
        for iss in r2_status:
            print(f"  {iss}")
            all_issues.append(iss)
    else:
        print(f"  {r2_status}")
    for k, v in r2_info.items():
        print(f"  {k} = {v:.4f}")
    summary["checks"]["r2"] = r2_info

    # 5. Predictions
    print("\n[5] Prediction Audit")
    pred_info, pred_status, pred_info_notes = audit_predictions(run_dir)
    for note in pred_info_notes:
        print(f"  INFO: {note}")
    if isinstance(pred_status, list):
        for iss in pred_status:
            print(f"  {iss}")
            all_issues.append(iss)
    else:
        print(f"  {pred_status}")
    for k, v in pred_info.items():
        if isinstance(v, float):
            print(f"  {k} = {v:.6f}" if abs(v) < 1000 else f"  {k} = {v}")
        else:
            print(f"  {k} = {v}")
    summary["checks"]["predictions"] = pred_info
    if pred_info_notes:
        summary["checks"]["prediction_info_notes"] = pred_info_notes

    # 6. Time shift
    print("\n[6] Time Shift Check")
    shift_info, shift_status = audit_time_shift(run_dir)
    if isinstance(shift_status, list):
        for iss in shift_status:
            print(f"  {iss}")
            all_issues.append(iss)
    else:
        print(f"  {shift_status}")
    for k, v in shift_info.items():
        print(f"  {k} = {v}")
    summary["checks"]["time_shift"] = shift_info

    # Final verdict
    print("\n" + "="*60)
    fails = [i for i in all_issues if i.startswith("FAIL")]
    warns = [i for i in all_issues if i.startswith("WARN")]

    if fails:
        verdict = "FAIL"
    elif warns:
        verdict = "WARN"
    else:
        verdict = "PASS"

    summary["verdict"] = verdict
    summary["issues"] = all_issues

    print(f"  VERDICT: {verdict}")
    if fails:
        print(f"  FAIL count: {len(fails)}")
        for f in fails:
            print(f"    - {f}")
    if warns:
        print(f"  WARN count: {len(warns)}")
        for w in warns:
            print(f"    - {w}")
    if not fails and not warns:
        print("  All checks passed.")
    print("="*60)

    # Save outputs
    json_path = os.path.join(run_dir, "audit_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {json_path}")

    md_path = os.path.join(run_dir, "audit_summary.md")
    with open(md_path, "w") as f:
        f.write(f"# Audit Summary\n\n")
        f.write(f"**Run Dir**: `{run_dir}`\n\n")
        f.write(f"**Verdict**: **{verdict}**\n\n")
        if metrics:
            f.write("## Metrics\n\n")
            for k, v in metrics.items():
                f.write(f"- {k}: {v}\n")
            f.write("\n")
        if all_issues:
            f.write("## Issues\n\n")
            for iss in all_issues:
                f.write(f"- {iss}\n")
            f.write("\n")
        if loss_info:
            f.write("## Loss Info\n\n")
            for k, v in loss_info.items():
                f.write(f"- {k}: {v:.6f}\n")
            f.write("\n")
        if r2_info:
            f.write("## R² Info\n\n")
            for k, v in r2_info.items():
                f.write(f"- {k}: {v:.4f}\n")
            f.write("\n")
        if pred_info:
            f.write("## Prediction Audit\n\n")
            pred_fields = [
                "true_min", "pred_min", "true_negative_ratio", "pred_negative_ratio",
                "negative_ratio_gap", "true_night_min", "true_night_p01",
                "large_negative_threshold", "extreme_negative_threshold",
                "night_pred_max", "daytime_true_max",
            ]
            for k in pred_fields:
                if k in pred_info:
                    v = pred_info[k]
                    f.write(f"- {k}: {v}\n")
            f.write("\n")
        if pred_info_notes:
            f.write("## Prediction Info Notes\n\n")
            for note in pred_info_notes:
                f.write(f"- INFO: {note}\n")
            f.write("\n")
        if shift_info:
            f.write("## Time Shift\n\n")
            for k, v in shift_info.items():
                f.write(f"- {k}: {v}\n")
    print(f"  Saved: {md_path}")


if __name__ == "__main__":
    main()
