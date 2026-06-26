
from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


def _loss_zoom_ylim(
    train_vals: Sequence[float],
    test_vals: Sequence[float],
    skip_epochs: int = 5,
) -> tuple[float, float]:
    """跳过前几轮 loss 爆炸区，按后续数值自动定 zoom 纵轴，便于看波动。"""
    n = len(train_vals)
    start = min(skip_epochs, max(0, n - 2))
    vals = [float(v) for v in list(train_vals[start:]) + list(test_vals[start:]) if v is not None]
    if not vals:
        return (0.0, 1.0)
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        hi = lo + max(abs(lo) * 0.1, 1e-8)
    pad = (hi - lo) * 0.12
    return (max(0.0, lo - pad), hi + pad)


def save_curve(
    epochs: Sequence[int],
    train_vals: Sequence[float],
    test_vals: Sequence[float],
    csv_path: Optional[str],
    png_path: str,
    ylabel: str,
    title: str,
    ylim: Optional[tuple] = None,
    yscale_log: bool = False,
) -> None:
    if csv_path is not None:
        df = pd.DataFrame({
            "epoch": list(epochs),
            f"train_{ylabel.lower()}": list(train_vals),
            f"test_{ylabel.lower()}":  list(test_vals),
        })
        df.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_vals, label=f"Train {ylabel}")
    plt.plot(epochs, test_vals,  label=f"Test {ylabel}")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    if yscale_log:
        plt.yscale("log")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()


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
    loss_ylim: Optional[tuple] = None,
) -> dict:
    epochs = history["epochs"]

    train_loss = history["train_loss"]
    test_loss = history["test_loss"]
    use_log = max(train_loss + test_loss) / max(min(train_loss + test_loss), 1e-12) > 50

    save_curve(
        epochs, train_loss, test_loss,
        paths["loss_csv"], paths["loss_png"],
        ylabel="Loss", title="Training and Test Loss Curve",
        ylim=None, yscale_log=use_log,
    )
    if "loss_zoom_png" in paths:
        zoom_ylim = loss_ylim if loss_ylim is not None else _loss_zoom_ylim(train_loss, test_loss)
        save_curve(
            epochs, train_loss, test_loss,
            None, paths["loss_zoom_png"],
            ylabel="Loss",
            title=(
                f"Training and Test Loss Curve (zoom y={zoom_ylim[0]:.4g}-{zoom_ylim[1]:.4g})"
            ),
            ylim=zoom_ylim,
        )
    save_curve(
        epochs, history["train_mae"], history["test_mae"],
        paths["mae_csv"], paths["mae_png"],
        ylabel="MAE", title="Training and Test MAE Curve",
    )
    save_curve(
        epochs, history["train_mse"], history["test_mse"],
        paths["mse_csv"], paths["mse_png"],
        ylabel="MSE", title="Training and Test MSE Curve",
    )
    save_curve(
        epochs, history["train_r2"], history["test_r2"],
        paths["r2_csv"], paths["r2_png"],
        ylabel="R²", title="Training and Test R² Curve",
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
