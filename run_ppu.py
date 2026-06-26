# -*- coding: utf-8 -*-
"""
run_ppu.py
==========
PPU-Former (iTransformer + PSG + WASE + DSC + PGIA) 的训练脚本。

它是 run.py 的"创新版"，结构上 1:1 对照 run.py，方便研一同学对比阅读：
    - 数据加载、TimeSeriesDataset 的写法保持一致
    - 训练 / 评估循环保持一致（loss、MAE、R² 都和 baseline 一样算）
    - 模型从 iTransformer 换成 iTransformerPGIA（17 变量、自带物理偏置）
    - 数据集从 pv{year}.csv 换成 pv{year}_ext.csv（17 列）
    - 注释里把"什么是新增的"标得很清楚

跑这个脚本前请先运行：
    python -m data_provider.add_solar_features
（把 dataset/pv2017.csv 等扩展成 dataset/pv2017_ext.csv，多出 12 列物理特征）

PPU-Former 的 step 0 数值上严格 == vanilla iTransformer，所以哪怕新模块还没"解锁"，
训练初期的 loss 走势应该和 baseline 几乎重合，不会扰动 backbone 的 warm-up。
"""

import os
import re
import math
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# 注意：这里换成 PPU-Former 主模型，原 iTransformer 不变也能用
from model.iTransformer_PGIA import iTransformerPGIA
from utils import create_save_paths, save_args_json, write_full_report, append_run_summary


# ========================== 配置区（你只需要改这里）==========================

# ---------- 实验身份（影响保存路径） ----------
model_name = "iTransformer_PGIA"   # 第1级目录名。换模型时改这个
des = "ppu_former_v2"               # 实验描述（v2 = 修复 delta/azimuth bug 后的新数据）

# ---------- 数据集 ----------
# 注意：要用「扩展版」CSV（多出 12 列物理特征），跑这个之前先执行
#       python -m data_provider.add_solar_features
dataset_name = "pv2017_ext"        # dataset/ 下的 csv 文件名（不含 .csv）
year = None                         # 留 None 自动从 dataset_name 抽 4 位年份

# ---------- 预测任务 ----------
pred_len = 1                       # 1 / 4 / 8 / 24
lookback_len = 168                  # 168 = 7 天
num_variates = 17                   # ★★ 改了：4 气象 + AP + 12 物理先验 = 17
target_idx = 4                      # ★★ AP 在变量轴上的索引 = 4（0-based）

# ---------- 模型超参（PSO 搜出的最优参数 2026-06-21） ----------
dim = 128
depth = 4                           # PSO 最优
heads = 2                           # PSO 最优
dim_head = 32

# ---------- PPU-Former 自己的内层维度（PSO 搜出的最优参数 2026-06-21）----------
phys_hidden_dim = 32                # PGIA 的 PhysBias MLP 内层维度
psg_hidden_dim = 64                 # PSO 最优
wase_hidden_dim = 64                # PSO 最优
dsc_kernels = (3, 5, 7)             # DSC 的多尺度卷积核（论文公式 9）
dsc_dropout = 0.0                   # 论文未规定，PPU 已经提供 γ 软启动

# ---------- ★ 消融开关（做模块级消融实验时改这里）----------
# (1) 模块开关：全 True = 完整 PPU-Former；全 False = 退化成 vanilla iTransformer。
#     完整     : 1,1,1,1
#     leave-one-out（去掉某个，证明该模块有用）：把对应项设 False
#     只留某个（证明其它不可缺少）：只把一项设 True
use_psg = True
use_wase = True
use_dsc = True
use_pgia = True
# (2) PPU 策略开关：True  = 论文版（γ/α 零起步、可学，恒等起步、渐进解锁）
#                   False = "无 PPU 对照"（γ/α 固定为 1，所有增强模块从训练开始就以全力作用）
#     用法：消融"PPU 策略"时，保持 (1) 全 True，把这个改成 False。
use_ppu = True

# ---------- 训练超参 ----------
# 与 baseline iTransformer 完全一致的训练协议
epochs = 100
batch_size = 32
learning_rate = 0.000190
gate_lr_mult = 5                    # γ 和其它参数同学习率

# ---------- 训练技巧（合规且不影响论文模型描述）----------
# 这一段全是「训练协议层面」的 trick，不属于模型结构，论文里完全可以不写。
# 全部默认开启，关掉就把对应开关置 False；与论文模型架构 100% 一致。

use_warmup_cosine = False           # baseline 没有 warmup
warmup_ratio = 0.05
cosine_min_ratio = 0.01

use_ema = False                     # baseline 没有 EMA
ema_decay = 0.999

grad_clip_norm = None               # baseline 没有 grad clip
weight_decay = 0.0                  # baseline 用 Adam，无 weight decay

use_early_stop = False              # baseline 协议：用最后一个 epoch
early_stop_patience = 9999

# ---------- Dropout（与主流 Transformer baseline 对齐：0.1） ----------
attn_dropout = 0.1
ff_dropout = 0.1

# ---------- 数据划分 ----------
train_ratio = 0.8

# ---------- 输出 ----------
results_dir = "results"
loss_plot_ylim = (0, 1)           # loss-zoom 纵轴范围

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================== 数据集（与 run.py 完全一致） ==========================


class TimeSeriesDataset(Dataset):
    """
    [输入回看 lookback_len 步, 输出 pred_len 步的 AP 序列]。

    __getitem__ 返回:
        x : [lookback_len, num_variates=17]  全部 17 个变量
        y : [pred_len]                        只取 AP 那一列
    """

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


# ========================== 工具函数 ==========================


def infer_year_from_dataset(name: str) -> int:
    match = re.search(r"(20\d{2})", name)
    if match:
        return int(match.group(1))
    raise ValueError(f"无法从 dataset_name='{name}' 推导年份，请显式设置 year 变量")


def load_data(dataset_name):
    """读 dataset/{dataset_name}.csv；要求是 17 列扩展版（用 add_solar_features.py 生成）。"""
    csv_path = os.path.join("dataset", f"{dataset_name}.csv")
    df = pd.read_csv(csv_path)
    timestamps = df.iloc[:, 0].values
    features = df.iloc[:, 1:].values.astype(np.float32)
    return features, timestamps


def create_dataloaders(features, timestamps, lookback_len, pred_len, train_ratio,
                       batch_size, target_idx):
    total_samples = len(features) - lookback_len - pred_len + 1
    train_size = int(total_samples * train_ratio)

    train_end_idx = train_size + lookback_len + pred_len - 1
    train_data_raw = features[:train_end_idx]
    train_timestamps = timestamps[:train_end_idx]

    test_data_raw = features[train_size:]
    test_timestamps = timestamps[train_size:]

    # 与 baseline 一致：用训练集统计量做 MinMax 归一化，再 transform 到训练 / 测试
    scaler = MinMaxScaler()
    scaler.fit(train_data_raw)

    train_data = scaler.transform(train_data_raw)
    test_data = scaler.transform(test_data_raw)

    target_min = scaler.data_min_[target_idx]
    target_max = scaler.data_max_[target_idx]

    train_dataset = TimeSeriesDataset(train_data, train_timestamps, lookback_len, pred_len, target_idx)
    test_dataset = TimeSeriesDataset(test_data, test_timestamps, lookback_len, pred_len, target_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    raw_test_target = test_data_raw[:, target_idx]

    return (
        train_loader, train_eval_loader, test_loader,
        test_timestamps, target_min, target_max, raw_test_target,
    )


# ========================== 训练技巧：EMA + Warmup-Cosine ==========================


class ModelEMA:
    """
    指数移动平均（Exponential Moving Average of weights）。

    每个 optimizer step 调一次 update()，shadow 权重以 `decay` 速率向当前权重靠拢。
    评估/保存时把 shadow 权重 swap 进 model；评估完用 restore() 还原 raw 权重继续训练。

    这是训练协议层面的稳健性技巧，论文里写不写都行；
    它对模型架构、loss、数据划分等任何「论文模型规格」都没有影响。
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        # 只对浮点参数做 EMA；buffer 中的 int / RevIN 内部计数等保持最新值
        self.shadow = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                self.shadow[k] = v.detach().clone()

    def apply_to(self, model: nn.Module):
        """把 EMA 权重写进 model，返回 raw 权重备份用于事后还原。"""
        backup = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }
        model.load_state_dict(self.shadow)
        return backup

    def restore(self, model: nn.Module, backup):
        model.load_state_dict(backup)


def make_warmup_cosine_scheduler(optimizer, total_epochs: int,
                                 warmup_ratio: float = 0.1,
                                 min_ratio: float = 0.01):
    """
    返回一个 LambdaLR，对所有 param_group 应用同样的百分比缩放：
        前 warmup_ratio*total_epochs 个 epoch：lr 从 0 线性升到 1.0 × group.lr
        之后到 total_epochs：lr 按 cosine 从 1.0 衰减到 min_ratio × group.lr

    注意每个 param_group 的「初始 lr」由 optimizer 自己保管，
    LambdaLR 只负责输出乘子，所以 backbone lr 与 gate lr 都按相同百分比缩放，
    不会破坏 gate_lr_mult 给 PPU 门控放大的相对关系。
    """
    warmup_epochs = max(1, int(round(total_epochs * warmup_ratio)))

    def lr_lambda(epoch):  # epoch 从 0 开始
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ========================== 训练与评估 ==========================


def train_one_epoch(model, train_loader, optimizer, criterion, target_idx,
                    ema=None, max_grad_norm=None):
    model.train()
    total_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)                              # [B, H, N]
        pred_active_power = pred[:, :, target_idx]   # [B, H]
        loss = criterion(pred_active_power, y)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if ema is not None:
            ema.update(model)
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate(model, loader, criterion, target_min, target_max, target_idx):
    """与 baseline 完全一致：归一化 MSE loss + 反归一化指标 + 反归一化的预测/真实值。"""
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            pred_active_power = pred[:, :, target_idx]
            loss = criterion(pred_active_power, y)
            total_loss += loss.item()
            all_preds.append(pred_active_power.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    avg_loss = total_loss / len(loader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    preds_orig = all_preds * (target_max - target_min) + target_min
    targets_orig = all_targets * (target_max - target_min) + target_min

    pf = preds_orig.flatten()
    tf = targets_orig.flatten()
    mse = float(mean_squared_error(tf, pf))
    mae = float(mean_absolute_error(tf, pf))
    r2 = float(r2_score(tf, pf))
    return avg_loss, mse, mae, r2, preds_orig, targets_orig


def collect_ppu_gates(model):
    """
    收集 PPU 标量门的当前值，用于事后追踪「哪个模块被解锁了」。
    返回一个普通 dict，便于 print / 写 csv。
    模块被关掉（=None）时对应键不存在，不会崩溃。
    """
    gates = {}
    if getattr(model, "psg", None) is not None and hasattr(model.psg, "gamma"):
        gates["gamma_psg"] = float(model.psg.gamma.detach().cpu())
    if model.wase is not None and hasattr(model.wase, "gamma"):
        gates["gamma_wase"] = float(model.wase.gamma.detach().cpu())
    if model.dsc is not None and hasattr(model.dsc, "gamma"):
        gates["gamma_dsc"] = float(model.dsc.gamma.detach().cpu())
    # 论文 2.4：PGIA 不再使用标量门，PPU 通过 PhysBiasMLP 末层零初始化实现，
    # 因此这里追踪每层 PhysBias 末层权重的范数（||W^(2)||）作为「PGIA 是否被解锁」的代理指标
    phys_norms = []
    for mm in model.backbone.modules():
        if hasattr(mm, "phys_bias") and getattr(mm, "use_phys_bias", False):
            phys_norms.append(float(mm.phys_bias.fc2.weight.detach().norm().cpu()))
    for i, w in enumerate(phys_norms):
        gates[f"physbias_w_norm_layer{i}"] = w
    return gates


# ========================== 主流程 ==========================


def main():
    global year
    if year is None:
        year = infer_year_from_dataset(dataset_name)

    print(f"Device: {device}")
    print(f"Model: {model_name}  |  Dataset: {dataset_name} (year={year})")
    print(f"Lookback: {lookback_len}, Pred length: {pred_len}, Target idx: {target_idx}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")
    print("-" * 60)

    features, timestamps = load_data(dataset_name)
    num_samples = features.shape[0]
    print(f"Data loaded: {num_samples} samples, {features.shape[1]} features")
    assert features.shape[1] == num_variates, (
        f"特征数对不上：CSV 有 {features.shape[1]} 列，但 num_variates={num_variates}。"
        f"请先运行  python -m data_provider.add_solar_features  生成扩展 CSV。"
    )

    (train_loader, train_eval_loader, test_loader,
     test_timestamps, target_min, target_max, raw_test_target) = create_dataloaders(
        features, timestamps, lookback_len, pred_len, train_ratio, batch_size, target_idx,
    )
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # ★ 模型从 iTransformer 换成 iTransformerPGIA：
    #   多了 PSG / WASE / DSC 三个输入端模块，backbone 内每层注意力开了 PGIA。
    model = iTransformerPGIA(
        num_variates=num_variates,
        lookback_len=lookback_len,
        pred_length=pred_len,
        target_idx=target_idx,
        # backbone 透传超参（与 baseline 对齐）
        dim=dim,
        depth=depth,
        heads=heads,
        dim_head=dim_head,
        num_tokens_per_variate=1,
        use_reversible_instance_norm=True,
        flash_attn=True,
        attn_dropout=attn_dropout,
        ff_dropout=ff_dropout,
        # PPU-Former 自己的内层维度
        phys_hidden_dim=phys_hidden_dim,
        psg_hidden_dim=psg_hidden_dim,
        wase_hidden_dim=wase_hidden_dim,
        dsc_kernels=dsc_kernels,
        dsc_dropout=dsc_dropout,
        # 消融开关
        use_psg=use_psg,
        use_wase=use_wase,
        use_dsc=use_dsc,
        use_pgia=use_pgia,
        # PPU 策略开关
        use_ppu=use_ppu,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"PPU 门初始状态: {collect_ppu_gates(model)}")
    print("-" * 60)

    # 分组学习率：PPU 门控参数 γ（PSG/WASE/DSC，名字含 "gamma" 且维度=1）用高 lr，其它正常
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
        {"params": other_params, "lr": learning_rate, "weight_decay": weight_decay},
        {"params": gate_params, "lr": learning_rate * gate_lr_mult, "weight_decay": 0.0},
    ])
    criterion = nn.MSELoss()

    scheduler = (
        make_warmup_cosine_scheduler(
            optimizer,
            total_epochs=epochs,
            warmup_ratio=warmup_ratio,
            min_ratio=cosine_min_ratio,
        ) if use_warmup_cosine else None
    )

    ema = ModelEMA(model, decay=ema_decay) if use_ema else None
    if scheduler is not None:
        warmup_eps = max(1, int(round(epochs * warmup_ratio)))
        print(f"LR schedule: warmup {warmup_eps} ep → cosine to lr*{cosine_min_ratio}")
    if ema is not None:
        print(f"EMA enabled (decay={ema_decay})  →  最终 metrics 用 EMA 权重报告")

    paths = create_save_paths(
        model_name=model_name,
        year=year,
        pred_len=pred_len,
        base_dir=results_dir,
    )
    print(f"Save dir: {paths['save_dir']}  (train{paths['train_id']})")
    print("-" * 60)

    history = {
        "epochs": [],
        "train_loss": [], "test_loss": [],
        "train_mae": [],  "test_mae": [],
        "train_mse": [],  "test_mse": [],
        "train_r2": [],   "test_r2": [],
    }
    gate_history = []

    t0 = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, target_idx,
            ema=ema, max_grad_norm=grad_clip_norm,
        )

        test_loss, test_mse, test_mae, test_r2, _, _ = evaluate(
            model, test_loader, criterion, target_min, target_max, target_idx,
        )

        if epoch == 1 or epoch == epochs or epoch % 50 == 0:
            _, train_mse, train_mae, train_r2, _, _ = evaluate(
                model, train_eval_loader, criterion, target_min, target_max, target_idx,
            )
        else:
            train_mse, train_mae, train_r2 = 0., 0., 0.

        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_mae"].append(train_mae)
        history["test_mae"].append(test_mae)
        history["train_mse"].append(train_mse)
        history["test_mse"].append(test_mse)
        history["train_r2"].append(train_r2)
        history["test_r2"].append(test_r2)

        gates_now = collect_ppu_gates(model)
        gate_history.append({"epoch": epoch, **gates_now})

        cur_lr = optimizer.param_groups[0]["lr"]
        gate_str = "  ".join(f"{k}={v:+.3f}" for k, v in gates_now.items())
        print(
            f"Epoch [{epoch:3d}/{epochs}]  lr={cur_lr:.2e}  "
            f"Train Loss: {train_loss:.6f}  Test Loss: {test_loss:.6f}  "
            f"Test MAE: {test_mae:.4f}  Test R²: {test_r2:.4f}  "
            f"{gate_str}"
        )

        if scheduler is not None:
            scheduler.step()

    train_time_sec = time.time() - t0

    # baseline 协议：直接用最后一个 epoch 的模型评估（不挑 best、不切 EMA）
    torch.save(model.state_dict(), paths["model_pth"])
    _, _, _, _, all_preds, all_targets = evaluate(
        model, test_loader, criterion, target_min, target_max, target_idx,
    )

    metrics = write_full_report(
        paths=paths,
        history=history,
        all_preds=all_preds,
        all_targets=all_targets,
        raw_test_target=raw_test_target,
        test_timestamps=test_timestamps,
        lookback_len=lookback_len,
        pred_len=pred_len,
        loss_ylim=loss_plot_ylim,
    )
    metrics["best_epoch"] = int(epochs)   # baseline 协议：报告最后一个 epoch 的模型
    metrics["train_time_sec"] = round(train_time_sec, 2)
    metrics["total_params"] = int(total_params)
    metrics["ppu_gates_final"] = collect_ppu_gates(model)

    # 把 PPU 门的逐 epoch 取值额外存一份，方便后续画门曲线
    gate_csv = os.path.join(paths["save_dir"], "ppu_gates.csv")
    pd.DataFrame(gate_history).to_csv(gate_csv, index=False, encoding="utf-8")

    print("-" * 60)
    print(f"RMSE: {metrics['RMSE']:.6f}")
    print(f"MAE:  {metrics['MAE']:.6f}")
    print(f"R2:   {metrics['R2']:.6f}")
    print(f"Best epoch: {metrics['best_epoch']}")
    print(f"Train time: {metrics['train_time_sec']:.1f}s")
    print(f"PPU 门最终取值: {metrics['ppu_gates_final']}")

    config = {
        "model": model_name,
        "des": des,
        "dataset": dataset_name,
        "year": year,
        "num_samples": int(num_samples),
        "pred_len": pred_len,
        "lookback_len": lookback_len,
        "num_variates": num_variates,
        "target_idx": target_idx,
        "dim": dim,
        "depth": depth,
        "heads": heads,
        "dim_head": dim_head,
        "phys_hidden_dim": phys_hidden_dim,
        "psg_hidden_dim": psg_hidden_dim,
        "wase_hidden_dim": wase_hidden_dim,
        "dsc_kernels": list(dsc_kernels),
        "dsc_dropout": dsc_dropout,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "gate_lr_mult": gate_lr_mult,
        "use_warmup_cosine": bool(use_warmup_cosine),
        "warmup_ratio": warmup_ratio if use_warmup_cosine else None,
        "cosine_min_ratio": cosine_min_ratio if use_warmup_cosine else None,
        "use_ema": bool(use_ema),
        "ema_decay": ema_decay if use_ema else None,
        "train_ratio": train_ratio,
        "device": str(device),
    }
    save_args_json(paths["args_json"], config, metrics)
    summary_path = append_run_summary(config=config, metrics=metrics, paths=paths)
    print(f"\nResults saved to: {paths['save_dir']}")
    print(f"Summary appended to: {summary_path}")


if __name__ == "__main__":
    main()
