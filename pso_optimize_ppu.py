# -*- coding: utf-8 -*-
"""
Batch-aware Robust PSO for PPU_Full.

固定约束：
1) 训练框架固定为 Adam + weight_decay=0.0 + 无 scheduler + MSE。
2) PSO 搜索使用严格时间顺序 70/10/20（train/val/test）。
3) 最终正式训练使用 80/20（train/test），不再使用 validation。
4) 搜索 11 个参数，并将 batch_size 纳入搜索空间。
5) fitness 同时考虑验证集精度、过拟合程度、后期反弹和夜间稳定性。
"""

import gc
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

from model.iTransformer_PGIA import iTransformerPGIA
from data_provider.split_utils import strict_chronological_split
from utils import create_save_paths, save_args_json
from utils.reporters import (
    save_all,
    save_best,
    save_curve,
    save_curve_after_epoch,
    save_overall_indicators,
    save_prediction_curve,
)


# ========================== 固定配置 ==========================

dataset_name = "pv2017_ext"
year = None

num_variates = 17
target_idx = 4
lookback_len = 168
pred_len = 4

FIXED_SEED = 35040
WEIGHT_DECAY = 0.0

USE_REVIN = True
USE_PPU = True
USE_PSG = True
USE_WASE = True
USE_DSC = True
USE_PGIA = True

dsc_kernels = (3, 5, 7)
psg_hidden_dim = 32
wase_hidden_dim = 64

TRAIN_RATIO = 0.7
VAL_END_RATIO = 0.8

EPOCHS_PER_CANDIDATE = 150
TOP_K_FINAL = 5
PSO_PARTICLES = 10
PSO_ITERATIONS = 6
PSO_W = 0.7
PSO_C1 = 1.5
PSO_C2 = 1.5

RESULTS_DIR = "results"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================== 搜索空间 ==========================

PARAM_NAMES = [
    "dim",
    "depth",
    "heads",
    "dim_head",
    "batch_size",
    "log10_lr",
    "dropout",
    "gate_lr_mult",
    "dsc_lr_mult",
    "dsc_gamma_lr_mult",
    "dsc_gamma_bound",
]

dim_choices = [96, 128, 192, 256]
depth_choices = [2, 3, 4]
heads_choices = [2, 4, 8]
dim_head_choices = [16, 32, 64]
batch_size_choices = [32, 64, 128]

PARAM_BOUNDS = np.array([
    [min(dim_choices), max(dim_choices)],                 # dim
    [min(depth_choices), max(depth_choices)],             # depth
    [min(heads_choices), max(heads_choices)],             # heads
    [min(dim_head_choices), max(dim_head_choices)],       # dim_head
    [min(batch_size_choices), max(batch_size_choices)],   # batch_size
    [-4.0, -3.1],                                         # log10_lr
    [0.10, 0.30],                                         # dropout
    [3.0, 12.0],                                          # gate_lr_mult
    [0.5, 4.0],                                           # dsc_lr_mult
    [0.5, 4.0],                                           # dsc_gamma_lr_mult
    [0.01, 0.04],                                         # dsc_gamma_bound
], dtype=np.float32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_torch_random_seed(seed: int) -> None:
    """仅重置 torch/random，避免影响 PSO 本身的 numpy 状态。"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def infer_year_from_dataset(name: str) -> int:
    match = re.search(r"(20\d{2})", name)
    if not match:
        raise ValueError(f"无法从 dataset_name='{name}' 推导年份")
    return int(match.group(1))


def load_data(ds_name: str) -> Tuple[np.ndarray, np.ndarray]:
    csv_path = os.path.join("dataset", f"{ds_name}.csv")
    df = pd.read_csv(csv_path)
    timestamps = df.iloc[:, 0].values
    features = df.iloc[:, 1:].values.astype(np.float32)
    return features, timestamps


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        timestamps: np.ndarray,
        lookback: int,
        horizon: int,
        target_col: int,
    ):
        self.data = data
        self.timestamps = timestamps
        self.lookback = lookback
        self.horizon = horizon
        self.target_col = target_col
        self.length = len(data) - lookback - horizon + 1
        if self.length <= 0:
            raise ValueError(
                f"样本数不足：len(data)={len(data)}, lookback={lookback}, pred_len={horizon}"
            )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        x = self.data[idx: idx + self.lookback]
        y = self.data[idx + self.lookback: idx + self.lookback + self.horizon, self.target_col]
        return torch.FloatTensor(x), torch.FloatTensor(y)


def nearest_choice(x: float, choices: List[int]) -> int:
    return min(choices, key=lambda v: abs(v - x))


def decode_params(position: np.ndarray) -> Dict[str, float]:
    raw = position.flatten().copy()
    for i in range(len(raw)):
        raw[i] = float(np.clip(raw[i], PARAM_BOUNDS[i, 0], PARAM_BOUNDS[i, 1]))

    dim = nearest_choice(raw[0], dim_choices)
    depth = nearest_choice(raw[1], depth_choices)
    heads = nearest_choice(raw[2], heads_choices)
    dim_head = nearest_choice(raw[3], dim_head_choices)
    batch_size = nearest_choice(raw[4], batch_size_choices)
    log10_lr = raw[5]
    dropout = raw[6]

    return {
        "dim": int(dim),
        "depth": int(depth),
        "heads": int(heads),
        "dim_head": int(dim_head),
        "batch_size": int(batch_size),
        "log10_lr": float(log10_lr),
        "learning_rate": float(10 ** log10_lr),
        "dropout": float(dropout),
        "attn_dropout": float(dropout),
        "ff_dropout": float(dropout),
        "gate_lr_mult": float(raw[7]),
        "dsc_lr_mult": float(raw[8]),
        "dsc_gamma_lr_mult": float(raw[9]),
        "dsc_gamma_bound": float(raw[10]),
    }


def build_model(params: Dict[str, float]) -> iTransformerPGIA:
    return iTransformerPGIA(
        num_variates=num_variates,
        lookback_len=lookback_len,
        pred_length=pred_len,
        target_idx=target_idx,
        dim=params["dim"],
        depth=params["depth"],
        heads=params["heads"],
        dim_head=params["dim_head"],
        num_tokens_per_variate=1,
        use_reversible_instance_norm=USE_REVIN,
        flash_attn=True,
        attn_dropout=params["attn_dropout"],
        ff_dropout=params["ff_dropout"],
        phys_hidden_dim=32,
        psg_hidden_dim=psg_hidden_dim,
        wase_hidden_dim=wase_hidden_dim,
        dsc_kernels=dsc_kernels,
        dsc_dropout=0.0,
        use_psg=USE_PSG,
        use_wase=USE_WASE,
        use_dsc=USE_DSC,
        use_pgia=USE_PGIA,
        use_ppu=USE_PPU,
        dsc_gamma_bound=params["dsc_gamma_bound"],
    ).to(device)


def build_optimizer(model: nn.Module, params: Dict[str, float]) -> torch.optim.Optimizer:
    dsc_params, dsc_gamma_params, gate_params, other_params = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "dsc." in name:
            if name.endswith("gamma") or name == "dsc.gamma":
                dsc_gamma_params.append(p)
            else:
                dsc_params.append(p)
        elif p.numel() == 1 and "gamma" in name:
            gate_params.append(p)
        else:
            other_params.append(p)

    lr = params["learning_rate"]
    param_groups = [
        {"params": other_params, "lr": lr},
        {"params": gate_params, "lr": lr * params["gate_lr_mult"]},
    ]
    if dsc_params:
        param_groups.append({"params": dsc_params, "lr": lr * params["dsc_lr_mult"]})
    if dsc_gamma_params:
        param_groups.append(
            {"params": dsc_gamma_params, "lr": lr * params["dsc_gamma_lr_mult"]}
        )
    return torch.optim.Adam(param_groups, weight_decay=WEIGHT_DECAY)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        pred_ap = pred[:, :, target_idx]
        loss = criterion(pred_ap, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def build_label_hours(sample_timestamps: np.ndarray) -> np.ndarray:
    ts_dt = pd.to_datetime(sample_timestamps)
    n_samples = len(sample_timestamps) - lookback_len - pred_len + 1
    rows = []
    for i in range(n_samples):
        rows.append(ts_dt[lookback_len + i: lookback_len + i + pred_len].hour.to_numpy())
    return np.asarray(rows, dtype=np.int64)


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    t_min: float,
    t_max: float,
    label_hours: np.ndarray,
) -> Dict[str, float]:
    model.eval()
    sse = 0.0
    count = 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            pred_ap = pred[:, :, target_idx]
            sse += torch.sum((pred_ap - y) ** 2).item()
            count += y.numel()
            all_preds.append(pred_ap.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    avg_loss = sse / count
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    preds_orig = all_preds * (t_max - t_min) + t_min
    targets_orig = all_targets * (t_max - t_min) + t_min

    pf = preds_orig.flatten()
    tf = targets_orig.flatten()
    mse = float(mean_squared_error(tf, pf))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(tf, pf))
    r2 = float(r2_score(tf, pf))
    step1_r2 = float(r2_score(targets_orig[:, 0], preds_orig[:, 0]))

    deep_mask = (label_hours >= 0) & (label_hours <= 5)
    deep_pred = preds_orig[deep_mask]
    deep_true = targets_orig[deep_mask]
    if deep_true.size > 0:
        deep_night_rmse = float(np.sqrt(mean_squared_error(deep_true, deep_pred)))
        deep_night_pred_abs_max = float(np.max(np.abs(deep_pred)))
    else:
        deep_night_rmse = 0.0
        deep_night_pred_abs_max = 0.0

    return {
        "avg_loss": float(avg_loss),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "step1_r2": step1_r2,
        "deep_night_rmse": deep_night_rmse,
        "deep_night_pred_abs_max": deep_night_pred_abs_max,
        "preds_orig": preds_orig,
        "targets_orig": targets_orig,
    }


def create_strict_70_10_20_bundle(features: np.ndarray, timestamps: np.ndarray) -> Dict[str, object]:
    total_len = len(features)
    train_end = int(total_len * TRAIN_RATIO)
    val_end = int(total_len * VAL_END_RATIO)

    train_raw = features[:train_end]
    val_raw = features[train_end - lookback_len:val_end]
    test_raw = features[val_end - lookback_len:]

    train_ts = timestamps[:train_end]
    val_ts = timestamps[train_end - lookback_len:val_end]
    test_ts = timestamps[val_end - lookback_len:]

    scaler = MinMaxScaler()
    scaler.fit(train_raw)
    train_data = scaler.transform(train_raw)
    val_data = scaler.transform(val_raw)
    test_data = scaler.transform(test_raw)

    train_label_min = lookback_len
    train_label_max = train_end - 1
    val_label_min = train_end
    val_label_max = val_end - 1
    test_label_min = val_end
    test_label_max = total_len - 1

    assert train_label_max < val_label_min, "train/val 标签重叠"
    assert val_label_max < test_label_min, "val/test 标签重叠"

    bundle = {
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
        "train_data_raw": train_raw,
        "val_data_raw": val_raw,
        "test_data_raw": test_raw,
        "train_timestamps": train_ts,
        "val_timestamps": val_ts,
        "test_timestamps": test_ts,
        "train_label_hours": build_label_hours(train_ts),
        "val_label_hours": build_label_hours(val_ts),
        "test_label_hours": build_label_hours(test_ts),
        "target_min": float(scaler.data_min_[target_idx]),
        "target_max": float(scaler.data_max_[target_idx]),
        "raw_test_target": test_raw[:, target_idx],
        "split_info": {
            "total_len": total_len,
            "train_end": train_end,
            "val_end": val_end,
            "train_label_range": [train_label_min, train_label_max],
            "val_label_range": [val_label_min, val_label_max],
            "test_label_range": [test_label_min, test_label_max],
            "scaler_fit_range": [0, train_end - 1],
        },
    }
    return bundle


def create_strict_80_20_bundle(features: np.ndarray, timestamps: np.ndarray) -> Dict[str, object]:
    sp = strict_chronological_split(
        features,
        timestamps,
        lookback_len,
        pred_len,
        train_ratio=0.8,
        verbose=True,
    )
    return {
        "train_data": sp["train_data"],
        "test_data": sp["test_data"],
        "train_data_raw": sp["train_data_raw"],
        "test_data_raw": sp["test_data_raw"],
        "train_timestamps": sp["train_timestamps"],
        "test_timestamps": sp["test_timestamps"],
        "train_label_hours": build_label_hours(sp["train_timestamps"]),
        "test_label_hours": build_label_hours(sp["test_timestamps"]),
        "target_min": float(sp["scaler"].data_min_[target_idx]),
        "target_max": float(sp["scaler"].data_max_[target_idx]),
        "raw_test_target": sp["test_data_raw"][:, target_idx],
        "split_info": sp["split_info"],
    }


def make_loader_bundle(split_bundle: Dict[str, object], batch_size: int) -> Dict[str, DataLoader]:
    train_ds = TimeSeriesDataset(
        split_bundle["train_data"],
        split_bundle["train_timestamps"],
        lookback_len,
        pred_len,
        target_idx,
    )
    val_ds = TimeSeriesDataset(
        split_bundle["val_data"],
        split_bundle["val_timestamps"],
        lookback_len,
        pred_len,
        target_idx,
    )
    test_ds = TimeSeriesDataset(
        split_bundle["test_data"],
        split_bundle["test_timestamps"],
        lookback_len,
        pred_len,
        target_idx,
    )
    return {
        "train_loader": DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True),
        "train_eval_loader": DataLoader(
            train_ds, batch_size=batch_size, shuffle=False, drop_last=False
        ),
        "val_loader": DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False),
        "test_loader": DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False),
    }


def make_final_80_20_loader_bundle(
    split_bundle: Dict[str, object], batch_size: int
) -> Dict[str, DataLoader]:
    train_ds = TimeSeriesDataset(
        split_bundle["train_data"],
        split_bundle["train_timestamps"],
        lookback_len,
        pred_len,
        target_idx,
    )
    test_ds = TimeSeriesDataset(
        split_bundle["test_data"],
        split_bundle["test_timestamps"],
        lookback_len,
        pred_len,
        target_idx,
    )
    return {
        "train_loader": DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True),
        "train_eval_loader": DataLoader(
            train_ds, batch_size=batch_size, shuffle=False, drop_last=False
        ),
        "test_loader": DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False),
    }


def append_search_log(csv_path: str, row: Dict[str, object]) -> None:
    write_header = not os.path.exists(csv_path)
    pd.DataFrame([row]).to_csv(csv_path, mode="a", index=False, header=write_header)


def invalid_result(status: str = "invalid") -> Dict[str, object]:
    return {
        "fitness": float("inf"),
        "best_val_RMSE": float("inf"),
        "best_epoch": -1,
        "train_RMSE_at_best": float("inf"),
        "val_RMSE_at_best": float("inf"),
        "train_val_gap": float("inf"),
        "mean_last10_val_RMSE": float("inf"),
        "late_overfit_gap": float("inf"),
        "deep_night_RMSE_at_best": float("inf"),
        "deep_night_abs_max_at_best": float("inf"),
        "status": status,
    }


def has_nonfinite(values: List[float]) -> bool:
    return any(not np.isfinite(v) for v in values)


def evaluate_candidate(params: Dict[str, float], split_bundle: Dict[str, object]) -> Dict[str, object]:
    set_torch_random_seed(FIXED_SEED)
    loaders = make_loader_bundle(split_bundle, params["batch_size"])

    model = None
    optimizer = None
    criterion = None
    try:
        model = build_model(params)
        optimizer = build_optimizer(model, params)
        criterion = nn.MSELoss()

        history = {
            "train_step_loss": [],
            "train_eval_loss": [],
            "train_rmse": [],
            "train_mae": [],
            "train_r2": [],
            "val_loss": [],
            "val_rmse": [],
            "val_mae": [],
            "val_r2": [],
            "deep_night_rmse": [],
            "deep_night_pred_abs_max": [],
        }

        for _ in range(EPOCHS_PER_CANDIDATE):
            train_step_loss = train_one_epoch(model, loaders["train_loader"], optimizer, criterion)
            train_eval = evaluate_loader(
                model,
                loaders["train_eval_loader"],
                criterion,
                split_bundle["target_min"],
                split_bundle["target_max"],
                split_bundle["train_label_hours"],
            )
            val_eval = evaluate_loader(
                model,
                loaders["val_loader"],
                criterion,
                split_bundle["target_min"],
                split_bundle["target_max"],
                split_bundle["val_label_hours"],
            )

            epoch_values = [
                train_step_loss,
                train_eval["avg_loss"],
                train_eval["rmse"],
                train_eval["mae"],
                train_eval["r2"],
                val_eval["avg_loss"],
                val_eval["rmse"],
                val_eval["mae"],
                val_eval["r2"],
                val_eval["deep_night_rmse"],
                val_eval["deep_night_pred_abs_max"],
            ]
            if has_nonfinite(epoch_values):
                return invalid_result(status="invalid")

            history["train_step_loss"].append(float(train_step_loss))
            history["train_eval_loss"].append(float(train_eval["avg_loss"]))
            history["train_rmse"].append(float(train_eval["rmse"]))
            history["train_mae"].append(float(train_eval["mae"]))
            history["train_r2"].append(float(train_eval["r2"]))
            history["val_loss"].append(float(val_eval["avg_loss"]))
            history["val_rmse"].append(float(val_eval["rmse"]))
            history["val_mae"].append(float(val_eval["mae"]))
            history["val_r2"].append(float(val_eval["r2"]))
            history["deep_night_rmse"].append(float(val_eval["deep_night_rmse"]))
            history["deep_night_pred_abs_max"].append(float(val_eval["deep_night_pred_abs_max"]))

        best_idx = int(np.argmin(history["val_rmse"]))
        best_val_rmse = float(history["val_rmse"][best_idx])
        best_epoch = best_idx + 1
        train_rmse_at_best = float(history["train_rmse"][best_idx])
        val_rmse_at_best = float(history["val_rmse"][best_idx])
        train_val_gap = float(max(0.0, val_rmse_at_best - train_rmse_at_best))
        mean_last10_val_rmse = float(np.mean(history["val_rmse"][-10:]))
        late_overfit_gap = float(max(0.0, mean_last10_val_rmse - best_val_rmse))
        deep_night_rmse_at_best = float(history["deep_night_rmse"][best_idx])
        deep_night_abs_max_at_best = float(history["deep_night_pred_abs_max"][best_idx])
        night_abs_penalty = float(max(0.0, deep_night_abs_max_at_best - 2.0))

        fitness = float(
            best_val_rmse
            + 0.30 * train_val_gap
            + 0.50 * late_overfit_gap
            + 0.10 * deep_night_rmse_at_best
            + 0.05 * night_abs_penalty
        )
        result_values = [
            fitness,
            best_val_rmse,
            train_rmse_at_best,
            val_rmse_at_best,
            train_val_gap,
            mean_last10_val_rmse,
            late_overfit_gap,
            deep_night_rmse_at_best,
            deep_night_abs_max_at_best,
        ]
        if has_nonfinite(result_values):
            return invalid_result(status="invalid")

        return {
            "fitness": fitness,
            "best_val_RMSE": best_val_rmse,
            "best_epoch": best_epoch,
            "train_RMSE_at_best": train_rmse_at_best,
            "val_RMSE_at_best": val_rmse_at_best,
            "train_val_gap": train_val_gap,
            "mean_last10_val_RMSE": mean_last10_val_rmse,
            "late_overfit_gap": late_overfit_gap,
            "deep_night_RMSE_at_best": deep_night_rmse_at_best,
            "deep_night_abs_max_at_best": deep_night_abs_max_at_best,
            "status": "ok",
        }
    except Exception:
        return invalid_result(status="invalid")
    finally:
        if model is not None:
            del model
        if optimizer is not None:
            del optimizer
        if criterion is not None:
            del criterion
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@dataclass
class Particle:
    position: np.ndarray
    velocity: np.ndarray
    best_position: np.ndarray
    best_fitness: float
    fitness: float


class PSOOptimizer:
    def __init__(self, ndim: int, n_particles: int, n_iterations: int, w: float, c1: float, c2: float):
        self.ndim = ndim
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.global_best_position = np.zeros((1, ndim), dtype=np.float32)
        self.global_best_fitness = float("inf")
        self.fitness_history: List[float] = []
        self.particles = self._init_particles()

    def _init_particles(self) -> List[Particle]:
        particles = []
        max_vel = (PARAM_BOUNDS[:, 1] - PARAM_BOUNDS[:, 0]) * 0.2
        for _ in range(self.n_particles):
            pos = np.array(
                [
                    np.random.uniform(PARAM_BOUNDS[i, 0], PARAM_BOUNDS[i, 1])
                    for i in range(self.ndim)
                ],
                dtype=np.float32,
            ).reshape(1, self.ndim)
            vel = np.random.uniform(-max_vel, max_vel).astype(np.float32).reshape(1, self.ndim)
            particles.append(Particle(pos, vel, pos.copy(), float("inf"), float("inf")))
        return particles

    def optimize(self, split_bundle: Dict[str, object], search_log_csv: str) -> List[Dict[str, object]]:
        all_records = []
        print("=" * 80)
        print("PPU_Full Batch-aware Robust PSO 开始")
        print(
            f"particles={self.n_particles}, iterations={self.n_iterations}, "
            f"epochs_per_candidate={EPOCHS_PER_CANDIDATE}"
        )
        print(
            f"search split: train={TRAIN_RATIO:.1f}, "
            f"val=[{TRAIN_RATIO:.1f},{VAL_END_RATIO:.1f}], test=[{VAL_END_RATIO:.1f},1.0]"
        )
        print(f"PARAM_NAMES={PARAM_NAMES}")
        print("=" * 80)

        for iteration in range(1, self.n_iterations + 1):
            print(f"\n{'=' * 80}")
            print(f"Iteration [{iteration}/{self.n_iterations}]")
            print(f"{'=' * 80}")
            for p_idx, particle in enumerate(self.particles, start=1):
                params = decode_params(particle.position)
                t0 = time.time()
                res = evaluate_candidate(params, split_bundle)
                elapsed = time.time() - t0

                particle.fitness = res["fitness"]
                if particle.fitness < particle.best_fitness:
                    particle.best_fitness = particle.fitness
                    particle.best_position = particle.position.copy()
                if particle.fitness < self.global_best_fitness:
                    self.global_best_fitness = particle.fitness
                    self.global_best_position = particle.position.copy()

                row = {
                    "iteration": iteration,
                    "particle": p_idx,
                    "dim": params["dim"],
                    "depth": params["depth"],
                    "heads": params["heads"],
                    "dim_head": params["dim_head"],
                    "batch_size": params["batch_size"],
                    "log10_lr": params["log10_lr"],
                    "learning_rate": params["learning_rate"],
                    "dropout": params["dropout"],
                    "gate_lr_mult": params["gate_lr_mult"],
                    "dsc_lr_mult": params["dsc_lr_mult"],
                    "dsc_gamma_lr_mult": params["dsc_gamma_lr_mult"],
                    "dsc_gamma_bound": params["dsc_gamma_bound"],
                    "fitness": res["fitness"],
                    "best_val_RMSE": res["best_val_RMSE"],
                    "best_epoch": res["best_epoch"],
                    "train_RMSE_at_best": res["train_RMSE_at_best"],
                    "val_RMSE_at_best": res["val_RMSE_at_best"],
                    "train_val_gap": res["train_val_gap"],
                    "mean_last10_val_RMSE": res["mean_last10_val_RMSE"],
                    "late_overfit_gap": res["late_overfit_gap"],
                    "deep_night_RMSE_at_best": res["deep_night_RMSE_at_best"],
                    "deep_night_abs_max_at_best": res["deep_night_abs_max_at_best"],
                    "elapsed_seconds": round(elapsed, 2),
                    "status": res["status"],
                }
                append_search_log(search_log_csv, row)
                all_records.append(row)
                print(
                    f"Particle {p_idx}/{self.n_particles} | "
                    f"fitness={row['fitness']:.6f} | "
                    f"best_val={row['best_val_RMSE']:.6f} | "
                    f"gap={row['train_val_gap']:.6f} | "
                    f"late_gap={row['late_overfit_gap']:.6f} | "
                    f"night_abs={row['deep_night_abs_max_at_best']:.4f} | "
                    f"status={row['status']}"
                )

            for particle in self.particles:
                r1 = np.random.rand(1, self.ndim)
                r2 = np.random.rand(1, self.ndim)
                particle.velocity = (
                    self.w * particle.velocity
                    + self.c1 * r1 * (particle.best_position - particle.position)
                    + self.c2 * r2 * (self.global_best_position - particle.position)
                )
                max_vel = (PARAM_BOUNDS[:, 1] - PARAM_BOUNDS[:, 0]) * 0.3
                particle.velocity = np.clip(particle.velocity, -max_vel, max_vel)
                particle.position = particle.position + particle.velocity

            self.fitness_history.append(self.global_best_fitness)
            bp = decode_params(self.global_best_position)
            print(
                f"Current global best fitness={self.global_best_fitness:.6f} "
                f"(dim={bp['dim']}, depth={bp['depth']}, heads={bp['heads']}, "
                f"dim_head={bp['dim_head']}, batch={bp['batch_size']}, "
                f"lr={bp['learning_rate']:.8f})"
            )

        return all_records


def save_final_80_20_history_artifacts(paths: Dict[str, str], history: Dict[str, List[float]]) -> None:
    loss_df = pd.DataFrame(
        {
            "epoch": history["epochs"],
            "train_eval_loss": history["train_eval_loss"],
            "train_step_loss": history["train_step_loss"],
            "test_loss": history["test_loss"],
            "train_RMSE": history["train_rmse"],
            "test_RMSE": history["test_rmse"],
            "train_R2": history["train_r2"],
            "test_R2": history["test_r2"],
        }
    )
    loss_df.to_csv(paths["loss_csv"], index=False)

    save_curve(
        history["epochs"],
        history["train_eval_loss"],
        history["test_loss"],
        None,
        paths["loss_png"],
        ylabel="Loss",
        title="Train Eval Loss vs Test Loss",
        series_label="Test",
    )
    if len(history["epochs"]) > 10:
        save_curve_after_epoch(
            history["epochs"],
            history["train_eval_loss"],
            history["test_loss"],
            paths["loss_zoom_png"],
            ylabel="Loss",
            title="Train Eval Loss vs Test Loss (from epoch 10)",
            skip_epochs=10,
            series_label="Test",
        )


def save_test_artifacts(
    paths: Dict[str, str],
    test_eval: Dict[str, object],
    raw_test_target: np.ndarray,
    test_timestamps: np.ndarray,
) -> Dict[str, float]:
    preds = test_eval["preds_orig"]
    targets = test_eval["targets_orig"]

    rmse = float(test_eval["rmse"])
    mae = float(test_eval["mae"])
    r2 = float(test_eval["r2"])
    save_overall_indicators(paths["overall_csv"], rmse, mae, r2)

    save_best(
        paths["best_csv"],
        paths["best_png"],
        preds,
        targets,
        pred_len,
        test_timestamps,
        lookback_len,
    )
    save_all(
        paths["all_csv"],
        paths["all_scatter_png"],
        paths["all_error_png"],
        preds,
        targets,
        pred_len,
        test_timestamps,
        lookback_len,
        predictions_csv=paths.get("predictions_csv"),
    )
    save_prediction_curve(
        paths["prediction_curve_png"],
        preds,
        targets,
        zoom_168h_png=paths.get("prediction_curve_168h_png"),
        zoom_168h_csv=paths.get("prediction_curve_168h_csv"),
        timestamps=test_timestamps,
        lookback_len=lookback_len,
        phase="test",
    )

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "step1_R2": float(test_eval["step1_r2"]),
        "deep_night_RMSE": float(test_eval["deep_night_rmse"]),
        "deep_night_pred_abs_max": float(test_eval["deep_night_pred_abs_max"]),
        "num_test_samples": int(preds.shape[0]),
        "raw_test_points": int(len(raw_test_target)),
    }


def candidate_to_final_params(candidate: Dict[str, object]) -> Dict[str, float]:
    dropout = float(candidate["dropout"])
    return {
        "dim": int(candidate["dim"]),
        "depth": int(candidate["depth"]),
        "heads": int(candidate["heads"]),
        "dim_head": int(candidate["dim_head"]),
        "batch_size": int(candidate["batch_size"]),
        "learning_rate": float(candidate["learning_rate"]),
        "dropout": dropout,
        "attn_dropout": dropout,
        "ff_dropout": dropout,
        "gate_lr_mult": float(candidate["gate_lr_mult"]),
        "dsc_lr_mult": float(candidate["dsc_lr_mult"]),
        "dsc_gamma_lr_mult": float(candidate["dsc_gamma_lr_mult"]),
        "dsc_gamma_bound": float(candidate["dsc_gamma_bound"]),
    }


def run_final_training_for_candidate(
    rank: int,
    candidate: Dict[str, object],
    final_split_bundle: Dict[str, object],
    run_root_dir: str,
    exp_year: int,
) -> Dict[str, float]:
    set_torch_random_seed(FIXED_SEED)

    params = candidate_to_final_params(candidate)
    final_epochs = int(candidate["best_epoch"])
    if final_epochs <= 0:
        raise RuntimeError(f"Top{rank} 的 source best_epoch 非法: {final_epochs}")

    loaders = make_final_80_20_loader_bundle(final_split_bundle, params["batch_size"])
    model = build_model(params)
    optimizer = build_optimizer(model, params)
    criterion = nn.MSELoss()

    history = {
        "epochs": [],
        "train_step_loss": [],
        "train_eval_loss": [],
        "train_rmse": [],
        "train_r2": [],
        "test_loss": [],
        "test_rmse": [],
        "test_r2": [],
    }

    t0 = time.time()
    for epoch in range(1, final_epochs + 1):
        train_step_loss = train_one_epoch(model, loaders["train_loader"], optimizer, criterion)
        train_eval = evaluate_loader(
            model,
            loaders["train_eval_loader"],
            criterion,
            final_split_bundle["target_min"],
            final_split_bundle["target_max"],
            final_split_bundle["train_label_hours"],
        )
        test_eval = evaluate_loader(
            model,
            loaders["test_loader"],
            criterion,
            final_split_bundle["target_min"],
            final_split_bundle["target_max"],
            final_split_bundle["test_label_hours"],
        )

        epoch_values = [
            train_step_loss,
            train_eval["avg_loss"],
            train_eval["rmse"],
            train_eval["r2"],
            test_eval["avg_loss"],
            test_eval["rmse"],
            test_eval["r2"],
            test_eval["deep_night_rmse"],
            test_eval["deep_night_pred_abs_max"],
        ]
        if has_nonfinite(epoch_values):
            raise RuntimeError(f"Top{rank} 训练过程中出现 NaN/inf，candidate 无效")

        history["epochs"].append(epoch)
        history["train_step_loss"].append(float(train_step_loss))
        history["train_eval_loss"].append(float(train_eval["avg_loss"]))
        history["train_rmse"].append(float(train_eval["rmse"]))
        history["train_r2"].append(float(train_eval["r2"]))
        history["test_loss"].append(float(test_eval["avg_loss"]))
        history["test_rmse"].append(float(test_eval["rmse"]))
        history["test_r2"].append(float(test_eval["r2"]))

        if epoch == 1 or epoch == final_epochs or epoch % 10 == 0:
            print(
                f"[Top{rank}] Epoch [{epoch}/{final_epochs}]  "
                f"TrainEval Loss: {train_eval['avg_loss']:.6f}  "
                f"TrainStep Loss: {train_step_loss:.6f}  "
                f"Test Loss: {test_eval['avg_loss']:.6f}  "
                f"Test RMSE: {test_eval['rmse']:.6f}  "
                f"Test R2: {test_eval['r2']:.4f}"
            )
    model_name = f"PPU_Full_PSO_Top{rank}"
    paths = create_save_paths(model_name=model_name, year=exp_year, pred_len=pred_len, base_dir=run_root_dir)
    torch.save(model.state_dict(), paths["model_pth"])
    torch.save(model.state_dict(), os.path.join(paths["save_dir"], "model.pth"))

    final_test = evaluate_loader(
        model,
        loaders["test_loader"],
        criterion,
        final_split_bundle["target_min"],
        final_split_bundle["target_max"],
        final_split_bundle["test_label_hours"],
    )
    if has_nonfinite(
        [
            final_test["avg_loss"],
            final_test["rmse"],
            final_test["mae"],
            final_test["r2"],
            final_test["step1_r2"],
            final_test["deep_night_rmse"],
            final_test["deep_night_pred_abs_max"],
        ]
    ):
        raise RuntimeError(f"Top{rank} test 评估出现 NaN/inf")

    save_final_80_20_history_artifacts(paths, history)
    metrics = save_test_artifacts(
        paths,
        final_test,
        final_split_bundle["raw_test_target"],
        final_split_bundle["test_timestamps"],
    )
    metrics["final_epochs"] = int(final_epochs)
    metrics["source_best_epoch"] = int(candidate["best_epoch"])
    metrics["source_best_val_RMSE"] = float(candidate["best_val_RMSE"])
    metrics["source_fitness"] = float(candidate["fitness"])
    metrics["train_time_sec"] = round(time.time() - t0, 2)

    config = {
        "model": model_name,
        "dataset_name": dataset_name,
        "year": exp_year,
        "num_variates": num_variates,
        "target_idx": target_idx,
        "lookback_len": lookback_len,
        "pred_len": pred_len,
        "epochs": final_epochs,
        "seed": FIXED_SEED,
        "optimizer": "Adam",
        "scheduler": None,
        "loss": "MSE",
        "weight_decay": WEIGHT_DECAY,
        "use_revin": USE_REVIN,
        "use_ppu": USE_PPU,
        "use_psg": USE_PSG,
        "use_wase": USE_WASE,
        "use_dsc": USE_DSC,
        "use_pgia": USE_PGIA,
        "pso_search_split": "70/10/20",
        "final_train_split": "80/20",
        "final_epoch_source": "candidate_best_epoch_from_pso_validation",
        "uses_validation_in_final_training": False,
        "split": final_split_bundle["split_info"],
        "params": params,
        "source_candidate": {
            k: candidate[k]
            for k in candidate
            if k
            in (
                "fitness",
                "best_val_RMSE",
                "best_epoch",
                "train_val_gap",
                "late_overfit_gap",
                "deep_night_RMSE_at_best",
                "deep_night_abs_max_at_best",
                *PARAM_NAMES,
            )
        },
    }
    save_args_json(paths["args_json"], config, metrics)

    return {
        "rank": rank,
        "save_dir": paths["save_dir"],
        "batch_size": params["batch_size"],
        "final_epochs": int(final_epochs),
        "RMSE": metrics["RMSE"],
        "MAE": metrics["MAE"],
        "R2": metrics["R2"],
        "step1_R2": metrics["step1_R2"],
        "deep_night_RMSE": metrics["deep_night_RMSE"],
        "deep_night_pred_abs_max": metrics["deep_night_pred_abs_max"],
        "source_best_epoch": int(candidate["best_epoch"]),
        "source_best_val_RMSE": float(candidate["best_val_RMSE"]),
        "source_fitness": float(candidate["fitness"]),
    }


def dedup_key(row: Dict[str, object]) -> Tuple[object, ...]:
    return (
        int(row["dim"]),
        int(row["depth"]),
        int(row["heads"]),
        int(row["dim_head"]),
        int(row["batch_size"]),
        round(float(row["learning_rate"]), 8),
        round(float(row["dropout"]), 4),
        round(float(row["gate_lr_mult"]), 4),
        round(float(row["dsc_lr_mult"]), 4),
        round(float(row["dsc_gamma_lr_mult"]), 4),
        round(float(row["dsc_gamma_bound"]), 4),
    )


def main():
    set_seed(FIXED_SEED)
    exp_year = infer_year_from_dataset(dataset_name) if year is None else year
    print(f"Device: {device}")
    print(f"Dataset: {dataset_name} (year={exp_year}), pred_len={pred_len}, lookback={lookback_len}")
    print(
        f"PSO: particles={PSO_PARTICLES}, iterations={PSO_ITERATIONS}, "
        f"epochs_per_candidate={EPOCHS_PER_CANDIDATE}"
    )
    print(
        f"Final top{TOP_K_FINAL} training: split=80/20, "
        f"epochs source=candidate_best_epoch_from_pso_validation"
    )

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    run_root_dir = os.path.join(RESULTS_DIR, f"pso_ppu_{dataset_name}_{run_tag}")
    os.makedirs(run_root_dir, exist_ok=True)
    search_log_csv = os.path.join(run_root_dir, "search_log.csv")

    features, timestamps = load_data(dataset_name)
    search_split_bundle = create_strict_70_10_20_bundle(features, timestamps)
    final_split_bundle = create_strict_80_20_bundle(features, timestamps)

    pso = PSOOptimizer(
        ndim=len(PARAM_BOUNDS),
        n_particles=PSO_PARTICLES,
        n_iterations=PSO_ITERATIONS,
        w=PSO_W,
        c1=PSO_C1,
        c2=PSO_C2,
    )
    all_records = pso.optimize(search_split_bundle, search_log_csv)

    valid_records = [
        row for row in all_records
        if row["status"] == "ok" and np.isfinite(float(row["fitness"]))
    ]
    if not valid_records:
        raise RuntimeError("PSO 未产生任何有效 candidate，best_candidates.csv 不会写入 NaN/inf")

    sorted_records = sorted(valid_records, key=lambda row: float(row["fitness"]))
    unique_records = []
    seen = set()
    for row in sorted_records:
        key = dedup_key(row)
        if key in seen:
            continue
        seen.add(key)
        unique_records.append(row)

    best_k = unique_records[:TOP_K_FINAL]
    best_df = pd.DataFrame(best_k)
    best_candidates_csv = os.path.join(run_root_dir, "best_candidates.csv")
    best_df.to_csv(best_candidates_csv, index=False)

    print(f"\nTop{len(best_k)} candidates saved: {best_candidates_csv}")
    print(
        best_df[
            [
                "fitness",
                "best_val_RMSE",
                "best_epoch",
                "dim",
                "depth",
                "heads",
                "dim_head",
                "batch_size",
                "learning_rate",
            ]
        ]
    )

    final_rows = []
    for rank, cand in enumerate(best_k, start=1):
        print(f"\n{'=' * 80}\nFinal training Top{rank}/{len(best_k)}\n{'=' * 80}")
        final_rows.append(
            run_final_training_for_candidate(rank, cand, final_split_bundle, run_root_dir, exp_year)
        )

    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(os.path.join(run_root_dir, "top5_final_80_20_results.csv"), index=False)

    pso_result = {
        "dataset_name": dataset_name,
        "fixed": {
            "num_variates": num_variates,
            "target_idx": target_idx,
            "lookback_len": lookback_len,
            "pred_len": pred_len,
            "seed": FIXED_SEED,
            "optimizer": "Adam",
            "scheduler": None,
            "loss": "MSE",
            "weight_decay": WEIGHT_DECAY,
            "use_revin": USE_REVIN,
            "use_ppu": USE_PPU,
            "use_psg": USE_PSG,
            "use_wase": USE_WASE,
            "use_dsc": USE_DSC,
            "use_pgia": USE_PGIA,
            "pso_search_split": "70/10/20",
            "final_train_split": "80/20",
            "pso_search_split_info": search_split_bundle["split_info"],
            "final_train_split_info": final_split_bundle["split_info"],
        },
        "pso_config": {
            "num_particles": PSO_PARTICLES,
            "num_iterations": PSO_ITERATIONS,
            "epochs_per_candidate": EPOCHS_PER_CANDIDATE,
            "param_names": PARAM_NAMES,
        },
        "best_fitness": float(best_df.iloc[0]["fitness"]),
        "best_candidate": best_df.iloc[0].to_dict(),
    }
    with open(os.path.join(run_root_dir, "pso_result.json"), "w", encoding="utf-8") as f:
        json.dump(pso_result, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nAll outputs saved to: {run_root_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
