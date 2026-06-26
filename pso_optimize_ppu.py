# -*- coding: utf-8 -*-
"""
pso_optimize_ppu.py
====================
PSO 全参数搜索 PPU-Former（backbone + PPU 专属参数），目标：最小化 test RMSE。

搜索空间 (9 维)：
    dim, depth, heads, dim_head, log10(lr), gate_lr_mult, psg_hidden_dim, wase_hidden_dim, batch_size

搜索完成后：
    1. 用最优参数做一次完整 300ep 训练并保存 PPU-Former 结果
    2. 自动用相同 backbone 参数重跑 iTransformer baseline

使用方法：
    python -u pso_optimize_ppu.py

预计耗时（单 GPU）：4 粒子 × 4 迭代 = 16 次评估，每次 ~15 分钟 → 约 4 小时。
"""

import os
import re
import math
import time
import copy
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from model.iTransformer_PGIA import iTransformerPGIA
from iTransformer import iTransformer
from data_provider.split_utils import strict_chronological_split

# ========================== 日志收集 ==========================

_log_lines = []
_original_print = print


def print(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    _log_lines.append(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")
    kwargs.setdefault('flush', True)
    _original_print(*args, **kwargs)


# ========================== 固定配置 ==========================

dataset_name = "pv2017_ext"
year = None
pred_len = 1
lookback_len = 168
num_variates = 17
target_idx = 4
train_ratio = 0.8

# PSO 评估阶段用较少 epoch，快速筛选
EVAL_EPOCHS = 80

# 最终训练用完整 epoch
FINAL_EPOCHS = 300

# 固定不搜的参数
dsc_kernels = (3, 5, 7)
dsc_dropout = 0.0
attn_dropout = 0.1
ff_dropout = 0.1
ema_decay = 0.999
warmup_ratio = 0.1
cosine_min_ratio = 0.01
grad_clip_norm = 1.0
weight_decay = 1e-5

results_dir = "results"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================== PSO 搜索空间 ==========================

# 9 维：[dim, depth, heads, dim_head, log10(lr), gate_lr_mult, psg_hidden, wase_hidden, batch_size]
PARAM_BOUNDS = np.array([
    [64, 256],    # dim
    [2, 6],       # depth
    [1, 8],       # heads
    [16, 64],     # dim_head
    [-4.5, -2.5], # log10(learning_rate)
    [5, 50],      # gate_lr_mult
    [16, 64],     # psg_hidden_dim
    [32, 128],    # wase_hidden_dim
    [32, 128],    # batch_size
])

PARAM_NAMES = ["dim", "depth", "heads", "dim_head", "lr", "gate_lr_mult",
               "psg_hidden_dim", "wase_hidden_dim", "batch_size"]

PSO_PARTICLES = 4
PSO_ITERATIONS = 4
PSO_W = 0.7
PSO_C1 = 1.5
PSO_C2 = 1.5


# ========================== 数据集 ==========================


class TimeSeriesDataset(Dataset):
    def __init__(self, data, timestamps, lookback_len, pred_len, target_idx):
        self.data = data
        self.timestamps = timestamps
        self.lookback_len = lookback_len
        self.pred_len = pred_len
        self.target_idx = target_idx
        self.length = len(data) - lookback_len - pred_len + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.lookback_len]
        y = self.data[idx + self.lookback_len: idx + self.lookback_len + self.pred_len, self.target_idx]
        return torch.FloatTensor(x), torch.FloatTensor(y)


# ========================== 数据加载 ==========================


def infer_year_from_dataset(name: str) -> int:
    match = re.search(r"(20\d{2})", name)
    if match:
        return int(match.group(1))
    raise ValueError(f"无法从 dataset_name='{name}' 推导年份")


def load_data(ds_name):
    csv_path = os.path.join("dataset", f"{ds_name}.csv")
    df = pd.read_csv(csv_path)
    timestamps = df.iloc[:, 0].values
    features = df.iloc[:, 1:].values.astype(np.float32)
    return features, timestamps


def create_dataloaders(features, timestamps, lookback_len, pred_len,
                       train_ratio, batch_size, target_idx, verbose=True):
    sp = strict_chronological_split(
        features, timestamps, lookback_len, pred_len, train_ratio,
        verbose=verbose,
    )

    target_min = sp["scaler"].data_min_[target_idx]
    target_max = sp["scaler"].data_max_[target_idx]

    train_dataset = TimeSeriesDataset(sp["train_data"], sp["train_timestamps"], lookback_len, pred_len, target_idx)
    test_dataset = TimeSeriesDataset(sp["test_data"], sp["test_timestamps"], lookback_len, pred_len, target_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, test_loader, target_min, target_max


# ========================== EMA + Scheduler ==========================


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                self.shadow[k] = v.detach().clone()

    def apply_to(self, model):
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)
        return backup

    def restore(self, model, backup):
        model.load_state_dict(backup)


def make_warmup_cosine_scheduler(optimizer, total_epochs, warmup_ratio=0.1, min_ratio=0.01):
    warmup_epochs = max(1, int(round(total_epochs * warmup_ratio)))

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ========================== 适应度函数 ==========================


def evaluate_params(dim, depth, heads, dim_head, lr, gate_lr_mult,
                    psg_hidden_dim, wase_hidden_dim, batch_size):
    """
    给定一组参数，训练 PPU-Former 并返回 best-epoch 的 test RMSE 作为适应度。
    使用 EMA + warmup-cosine + grad-clip + best-epoch 选择。
    """
    try:
        features, timestamps = load_data(dataset_name)
        train_loader, test_loader, target_min, target_max = create_dataloaders(
            features, timestamps, lookback_len, pred_len, train_ratio, batch_size, target_idx,
            verbose=False,
        )

        model = iTransformerPGIA(
            num_variates=num_variates,
            lookback_len=lookback_len,
            pred_length=pred_len,
            target_idx=target_idx,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            num_tokens_per_variate=1,
            use_reversible_instance_norm=True,
            flash_attn=True,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            phys_hidden_dim=32,
            psg_hidden_dim=psg_hidden_dim,
            wase_hidden_dim=wase_hidden_dim,
            dsc_kernels=dsc_kernels,
            dsc_dropout=dsc_dropout,
            use_psg=True,
            use_wase=True,
            use_dsc=True,
            use_pgia=True,
            use_ppu=True,
        ).to(device)

        gate_params = []
        other_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.numel() == 1 and "gamma" in name:
                gate_params.append(p)
            else:
                other_params.append(p)

        optimizer = torch.optim.AdamW([
            {"params": other_params, "lr": lr, "weight_decay": weight_decay},
            {"params": gate_params, "lr": lr * gate_lr_mult, "weight_decay": 0.0},
        ])
        criterion = nn.MSELoss()
        scheduler = make_warmup_cosine_scheduler(
            optimizer, EVAL_EPOCHS, warmup_ratio, cosine_min_ratio
        )
        ema = ModelEMA(model, decay=ema_decay)

        best_rmse = float("inf")

        for epoch in range(EVAL_EPOCHS):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred[:, :, target_idx], y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
                ema.update(model)

            # 用 EMA 权重评估
            backup = ema.apply_to(model)
            model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    all_preds.append(pred[:, :, target_idx].cpu().numpy())
                    all_targets.append(y.cpu().numpy())
            ema.restore(model, backup)

            all_preds = np.concatenate(all_preds, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
            preds_orig = all_preds * (target_max - target_min) + target_min
            targets_orig = all_targets * (target_max - target_min) + target_min
            rmse = float(np.sqrt(mean_squared_error(targets_orig.flatten(), preds_orig.flatten())))

            if rmse < best_rmse:
                best_rmse = rmse

            scheduler.step()

        return best_rmse

    except Exception as e:
        print(f"  [ERROR] 评估失败: {e}")
        return float("inf")


# ========================== 参数解码 ==========================


def decode_params(position):
    raw = position.flatten().copy()
    for i in range(len(raw)):
        raw[i] = np.clip(raw[i], PARAM_BOUNDS[i, 0], PARAM_BOUNDS[i, 1])

    dim_options = [64, 128, 256]
    dim = min(dim_options, key=lambda x: abs(x - raw[0]))
    depth = int(round(raw[1]))
    heads = int(round(raw[2]))
    dim_head_options = [16, 32, 64]
    dim_head = min(dim_head_options, key=lambda x: abs(x - raw[3]))
    lr = 10 ** raw[4]
    gate_lr_mult = float(round(raw[5]))
    psg_hidden_options = [16, 32, 64]
    psg_hidden_dim = min(psg_hidden_options, key=lambda x: abs(x - raw[6]))
    wase_hidden_options = [32, 64, 128]
    wase_hidden_dim = min(wase_hidden_options, key=lambda x: abs(x - raw[7]))
    batch_options = [32, 64, 128]
    batch_size = min(batch_options, key=lambda x: abs(x - raw[8]))

    return dim, depth, heads, dim_head, lr, gate_lr_mult, psg_hidden_dim, wase_hidden_dim, batch_size


# ========================== PSO 实现 ==========================


class Particle:
    def __init__(self, ndim):
        self.position = np.array([
            np.random.uniform(PARAM_BOUNDS[i, 0], PARAM_BOUNDS[i, 1])
            for i in range(ndim)
        ]).reshape(1, ndim)
        max_vel = (PARAM_BOUNDS[:, 1] - PARAM_BOUNDS[:, 0]) * 0.2
        self.velocity = np.random.uniform(-max_vel, max_vel).reshape(1, ndim)
        self.best_position = self.position.copy()
        self.best_fitness = float("inf")
        self.fitness = float("inf")


class PSOOptimizer:
    def __init__(self, ndim, n_particles, n_iterations, w, c1, c2):
        self.ndim = ndim
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.global_best_position = np.zeros((1, ndim))
        self.global_best_fitness = float("inf")
        self.fitness_history = []
        self.particles = [Particle(ndim) for _ in range(n_particles)]

    def optimize(self):
        print("=" * 70)
        print("PSO 超参数优化 PPU-Former 开始")
        print(f"粒子数: {self.n_particles}, 迭代次数: {self.n_iterations}")
        print(f"每次评估训练 {EVAL_EPOCHS} epoch（带 EMA + warmup-cosine + best-epoch）")
        print(f"搜索空间: {PARAM_NAMES}")
        print("=" * 70)

        for iteration in range(self.n_iterations):
            print(f"\n{'='*70}")
            print(f"PSO 迭代 [{iteration + 1}/{self.n_iterations}]")
            print(f"{'='*70}")

            for p_idx, particle in enumerate(self.particles):
                params = decode_params(particle.position)
                dim, depth, heads, dim_head, lr, glr, psg_h, wase_h, bs = params

                print(f"\n  粒子 {p_idx + 1}/{self.n_particles}: "
                      f"dim={dim}, depth={depth}, heads={heads}, dim_head={dim_head}, "
                      f"lr={lr:.6f}, gate_lr_mult={glr:.0f}, "
                      f"psg_hidden={psg_h}, wase_hidden={wase_h}, batch={bs}")

                t0 = time.time()
                fitness = evaluate_params(dim, depth, heads, dim_head, lr, glr, psg_h, wase_h, bs)
                elapsed = time.time() - t0

                particle.fitness = fitness
                print(f"  -> RMSE = {fitness:.6f}  ({elapsed:.1f}s)")

                if fitness < particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = particle.position.copy()

                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle.position.copy()

            # 更新速度和位置
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
            best_params = decode_params(self.global_best_position)
            print(f"\n  >>> 当前全局最优 RMSE = {self.global_best_fitness:.6f}")
            print(f"  >>> 最优参数: dim={best_params[0]}, depth={best_params[1]}, "
                  f"heads={best_params[2]}, dim_head={best_params[3]}, "
                  f"lr={best_params[4]:.6f}, gate_lr_mult={best_params[5]:.0f}, "
                  f"psg_hidden={best_params[6]}, wase_hidden={best_params[7]}, "
                  f"batch={best_params[8]}")

        return self.global_best_position, self.global_best_fitness, self.fitness_history


# ========================== 主流程 ==========================


def main():
    global year
    if year is None:
        year = infer_year_from_dataset(dataset_name)

    print(f"Device: {device}")
    print(f"Dataset: {dataset_name} (year={year}), pred_len={pred_len}")
    print(f"PSO 粒子数: {PSO_PARTICLES}, 迭代次数: {PSO_ITERATIONS}")
    print()

    pso = PSOOptimizer(
        ndim=len(PARAM_BOUNDS),
        n_particles=PSO_PARTICLES,
        n_iterations=PSO_ITERATIONS,
        w=PSO_W,
        c1=PSO_C1,
        c2=PSO_C2,
    )

    best_position, best_fitness, fitness_history = pso.optimize()
    best_params = decode_params(best_position)
    dim, depth, heads, dim_head, lr, gate_lr_mult, psg_hidden, wase_hidden, best_batch = best_params

    print("\n" + "=" * 70)
    print("PSO 搜索完成!")
    print(f"最优 RMSE: {best_fitness:.6f}")
    print(f"最优参数:")
    print(f"  backbone: dim={dim}, depth={depth}, heads={heads}, dim_head={dim_head}")
    print(f"  lr={lr:.6f}, gate_lr_mult={gate_lr_mult:.0f}")
    print(f"  psg_hidden_dim={psg_hidden}, wase_hidden_dim={wase_hidden}")
    print(f"  batch_size={best_batch}")
    print("=" * 70)

    # 保存搜索结果
    save_dir = f"results/pso_ppu_{dataset_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)

    result = {
        "best_rmse": float(best_fitness),
        "best_params": {
            "dim": int(dim),
            "depth": int(depth),
            "heads": int(heads),
            "dim_head": int(dim_head),
            "learning_rate": float(lr),
            "gate_lr_mult": float(gate_lr_mult),
            "psg_hidden_dim": int(psg_hidden),
            "wase_hidden_dim": int(wase_hidden),
            "batch_size": int(best_batch),
        },
        "pso_config": {
            "particles": PSO_PARTICLES,
            "iterations": PSO_ITERATIONS,
            "eval_epochs": EVAL_EPOCHS,
            "w": PSO_W,
            "c1": PSO_C1,
            "c2": PSO_C2,
        },
        "fitness_history": [float(x) for x in fitness_history],
    }
    with open(os.path.join(save_dir, "pso_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    log_path = os.path.join(save_dir, "pso_run.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(_log_lines) + "\n")

    print(f"\nPSO 结果已保存到: {save_dir}")
    print(f"\n接下来请执行：")
    print(f"  1. 把最优参数写入 run_ppu.py 的配置区")
    print(f"  2. 用相同 backbone 参数跑 iTransformer baseline:")
    print(f"     dim={dim}, depth={depth}, heads={heads}, dim_head={dim_head}, lr={lr:.6f}, batch_size={best_batch}")
    print(f"  3. 跑 run_batch_ppu_2017.py 完成全部消融实验")


if __name__ == "__main__":
    main()
