# -*- coding: utf-8 -*-

from __future__ import annotations

import datetime
import os
from typing import Optional


_FIELDS = [
    ("datetime",  19),
    ("model",     20),
    ("year",       6),
    ("pl",         4),
    ("train",      6),
    ("des",       18),
    ("RMSE",      10),
    ("MAE",       10),
    ("R2",        10),
    ("best_ep",    8),
    ("time(s)",    9),
]


def _format_row(values):
    parts = []
    for (_, w), v in zip(_FIELDS, values):
        s = str(v)
        if len(s) > w:
            s = s[:w]
        parts.append(s.ljust(w))
    return "  ".join(parts)


def _write_header_if_needed(path: str) -> None:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    header = _format_row([name for name, _ in _FIELDS])
    sep = "-" * len(header)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 累积实验记录（每跑完一次自动追加一行）\n")
        f.write(header + "\n")
        f.write(sep + "\n")


def append_run_summary(config: dict, metrics: dict, paths: dict,
                       summary_path: Optional[str] = None) -> str:
    if summary_path is None:
        summary_path = os.path.abspath("all_runs.txt")

    _write_header_if_needed(summary_path)

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = _format_row([
        now,
        config.get("model", "?"),
        config.get("year", "?"),
        config.get("pred_len", "?"),
        paths.get("train_id", "?"),
        config.get("des", "-"),
        f"{metrics.get('RMSE', float('nan')):.4f}",
        f"{metrics.get('MAE',  float('nan')):.4f}",
        f"{metrics.get('R2',   float('nan')):.4f}",
        metrics.get("best_epoch", "-"),
        f"{metrics.get('train_time_sec', 0.0):.1f}",
    ])
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(row + "\n")

    return summary_path
