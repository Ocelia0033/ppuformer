
from __future__ import annotations

import json
import os
from typing import Optional


def _next_train_id(parent_dir: str) -> int:
    if not os.path.isdir(parent_dir):
        return 1
    max_id = 0
    for name in os.listdir(parent_dir):
        full = os.path.join(parent_dir, name)
        if not os.path.isdir(full):
            continue
        if name.startswith("train"):
            suffix = name[len("train"):]
            if suffix.isdigit():
                idx = int(suffix)
                if idx > max_id:
                    max_id = idx
    return max_id + 1


def create_save_paths(
    model_name: str,
    year: int,
    pred_len: int,
    base_dir: str = "results",
) -> dict:
    parent_dir = os.path.join(base_dir, model_name, f"pl{pred_len}", f"y{year}")
    train_id = _next_train_id(parent_dir)
    save_dir = os.path.join(parent_dir, f"train{train_id}")
    os.makedirs(save_dir, exist_ok=True)

    def p(filename: str) -> str:
        return os.path.join(save_dir, filename)

    paths = {
        "save_dir": save_dir,
        "train_id": train_id,
        "args_json": p("args.json"),
        "model_pth": p(f"{model_name}.pth"),

        "loss_csv":      p("loss.csv"),
        "loss_png":      p("loss.png"),
        "loss_zoom_png": p("loss-zoom.png"),
        "mae_csv":       p("mae.csv"),
        "mae_png":       p("mae.png"),
        "mse_csv":       p("mse.csv"),
        "mse_png":       p("mse.png"),
        "r2_csv":        p("R².csv"),
        "r2_png":        p("R².png"),

        "overall_csv":     p("Overall indicators.csv"),
        "best_csv":        p("Best.csv"),
        "best_png":        p("Best.png"),
        "all_csv":         p("ALL.csv"),
        "all_scatter_png": p("ALL-scatter.png"),
        "all_error_png":   p("ALL-error.png"),
        "predictions_csv": p("predictions.csv"),

        "best_part_csv": p("Best-Part.csv"),
        "best_part_png": p("Best-Part.png"),

        "best7d_csv":             p("Best7天.csv"),
        "best7d_png":             p("Best7天.png"),
        "best14d_compare_png":    p("Best对比14天.png"),
        "best7d_opt_csv":         p("Best7天优化.csv"),
        "best7d_opt_hourly_png":  p("Best7天优化hourly.png"),
        "best14d_opt_csv":        p("Best14天优化.csv"),
        "best14d_opt_hourly_png": p("Best14天优化hourly.png"),
    }
    return paths


def save_args_json(args_path: str, config: dict, metrics: Optional[dict] = None) -> None:
    payload = dict(config)
    if metrics is not None:
        payload["metrics"] = metrics
    os.makedirs(os.path.dirname(args_path), exist_ok=True)
    with open(args_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
