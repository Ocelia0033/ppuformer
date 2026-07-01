# -*- coding: utf-8 -*-
"""
scripts/check_loss_credibility.py
==================================
检查 loss.csv 是否真实、loader 是否混用、预测是否时间错位。
可从已有结果目录离线重生成 loss_zoom.png。

Usage:
    python scripts/check_loss_credibility.py
    python scripts/check_loss_credibility.py --run_dir results/PPU_PSG_WASE_PGIA/pl4/y2017/train1
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.reporters import save_loss_extended_plots


def find_loss_csvs(root="results"):
    return sorted(glob.glob(os.path.join(root, "**", "loss.csv"), recursive=True))


def _detect_cols(df):
    train_col = test_col = None
    for c in df.columns:
        cl = c.lower()
        if "train" in cl and "loss" in cl:
            train_col = c
        elif ("test" in cl or "val" in cl) and "loss" in cl:
            test_col = c
    if train_col is None and len(df.columns) >= 2:
        train_col = df.columns[1]
    if test_col is None and len(df.columns) >= 3:
        test_col = df.columns[2]
    return train_col, test_col


def inspect_loss_csv(path):
    df = pd.read_csv(path)
    train_col, test_col = _detect_cols(df)
    out = {"path": path, "rows": len(df), "columns": list(df.columns)}

    print(f"\n{'='*70}")
    print(f"  LOSS CSV: {path}")
    print(f"{'='*70}")
    print(f"  columns: {list(df.columns)}  |  rows: {len(df)}")
    print("\n  --- first 20 rows ---")
    print(df.head(20).to_string(index=False))
    print("\n  --- last 10 rows ---")
    print(df.tail(10).to_string(index=False))

    for name, col in [("train_loss", train_col), ("test_loss", test_col)]:
        if col is None:
            continue
        vals = df[col].values.astype(float)
        out[f"{name}_first"] = float(vals[0])
        out[f"{name}_last"] = float(vals[-1])
        out[f"{name}_min"] = float(np.min(vals))
        out[f"{name}_max"] = float(np.max(vals))
        out[f"{name}_best"] = float(np.min(vals))
        print(f"\n  [{name}] col={col}")
        print(f"    first={vals[0]:.6f}  last={vals[-1]:.6f}")
        print(f"    min={np.min(vals):.6f}  max={np.max(vals):.6f}  best={np.min(vals):.6f}")

        diffs = np.diff(vals)
        if len(diffs) > 0:
            mono_dec = np.all(diffs <= 1e-9)
            big_jumps = int(np.sum(np.abs(diffs) > 0.5 * np.abs(vals[:-1] + 1e-12)))
            print(f"    monotonic_decrease={mono_dec}  large_relative_jumps={big_jumps}")

    # 检查是否像被平滑（相邻 epoch 变化极小且非单调）
    if train_col and test_col and len(df) > 5:
        tr = df[train_col].values
        te = df[test_col].values
        tr_std = np.std(np.diff(tr))
        te_std = np.std(np.diff(te))
        out["train_diff_std"] = float(tr_std)
        out["test_diff_std"] = float(te_std)
        if tr_std < 1e-8 and te_std < 1e-8:
            print("  WARN: epoch loss diffs nearly constant — suspicious smoothing?")
        else:
            print(f"  OK: train_diff_std={tr_std:.6e}, test_diff_std={te_std:.6e} (raw epoch averages)")

    return df, train_col, test_col, out


def regenerate_plots(run_dir, df, train_col, test_col):
    epoch_col = "epoch" if "epoch" in df.columns else df.columns[0]
    epochs = df[epoch_col].tolist()
    save_loss_extended_plots(
        epochs, df[train_col].tolist(), df[test_col].tolist(),
        save_dir=run_dir, series_label="Test", zoom_start_epoch=10,
    )
    print(f"  regenerated: loss_zoom.png in {run_dir}")


def time_shift_check(run_dir):
    pred_path = os.path.join(run_dir, "predictions.csv")
    if not os.path.exists(pred_path):
        return None
    df = pd.read_csv(pred_path)
    pred_col = true_col = None
    for c in df.columns:
        cl = c.lower()
        if "pred" in cl:
            pred_col = c
        if "true" in cl or "actual" in cl or "target" in cl:
            true_col = c
    if pred_col is None or true_col is None:
        return None

    preds = df[pred_col].values.astype(float)
    trues = df[true_col].values.astype(float)
    if len(preds) < 48:
        return None

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
    info = {
        "corr_shift0": round(corrs[0], 4),
        "best_shift": int(best_shift),
        "best_corr": round(corrs[best_shift], 4),
        "corr_improvement": round(corrs[best_shift] - corrs[0], 4),
    }
    warn = best_shift != 0 and info["corr_improvement"] > 0.05
    info["status"] = "WARN: possible time misalignment" if warn else "OK"
    return info


def loader_audit_summary():
    return """
LOADER AUDIT (code review, run.py / run_ppuformer.py):
  train_loss  <- train_one_epoch(model, train_loader)     [shuffle=True, drop_last=True]
  test_loss   <- evaluate(model, test_loader)             [shuffle=False]
  train_mae/mse/r2 <- evaluate(model, train_eval_loader)  [shuffle=False, same train data]
  test_mae/mse/r2  <- evaluate(model, test_loader)
  => train_loss 与 test_loss 来自不同 loader，未混用。
  => history['train_*'] 来自 train_eval_loader，history['test_*'] 来自 test_loader（已修复 baseline 脚本）。
  => loss.csv 保存的是 history['train_loss'] 和 history['test_loss']，未经平滑。
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default=None, help="Single run dir; default=scan all results")
    parser.add_argument("--no_regen", action="store_true", help="Skip regenerating loss plots")
    args = parser.parse_args()

    if args.run_dir:
        csv_paths = [os.path.join(args.run_dir, "loss.csv")]
    else:
        csv_paths = find_loss_csvs()

    print(loader_audit_summary())

    all_summaries = []
    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            print(f"SKIP: {csv_path} not found")
            continue
        run_dir = os.path.dirname(csv_path)
        df, train_col, test_col, summary = inspect_loss_csv(csv_path)
        all_summaries.append(summary)

        if not args.no_regen and train_col and test_col:
            regenerate_plots(run_dir, df, train_col, test_col)

        ts = time_shift_check(run_dir)
        if ts:
            print(f"\n  [time shift] {run_dir}")
            for k, v in ts.items():
                print(f"    {k} = {v}")

        batch_csv = os.path.join(run_dir, "batch_loss_epoch1.csv")
        if os.path.exists(batch_csv):
            bdf = pd.read_csv(batch_csv)
            bl = bdf["loss"].values
            print(f"\n  [batch loss epoch1] batches={len(bl)}  "
                  f"min={bl.min():.4f} max={bl.max():.4f} std={bl.std():.4f}")
        else:
            print(f"\n  [batch loss] batch_loss_epoch1.csv not found (need re-run with save_batch_loss_epochs>0)")

    out_path = os.path.join("results", "loss_credibility_report.json")
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved summary: {out_path}")


if __name__ == "__main__":
    main()
