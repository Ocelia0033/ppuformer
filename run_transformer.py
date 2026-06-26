# -*- coding: utf-8 -*-
"""
run_transformer.py
==================
跑 Vanilla Transformer (Vaswani et al., 2017) baseline 的训练脚本。

整体结构 1:1 对照 ``run_informer.py``，只改了 4 处：
    (1) ``model_name`` / ``des`` 改成 Transformer
    (2) 模型从 ``InformerWrapper`` 换成 ``TransformerWrapper``
    (3) 删掉 Informer 自己的 ``factor`` / ``attn`` / ``distil`` / ``mix`` 这些超参
    (4) 默认超参换成 TSlib scripts/Transformer/ETTh1.sh 的标准配置

数据加载、归一化、9:1 chronological 切分、评估指标、保存路径与 PPU-Former /
Informer 完全一致，保证三者 / N 者对比 100% 公平。

★ 数据约定：baseline 用 5 列原始 CSV（``pv20XX.csv``，含 4 气象 + AP），**不**用
扩展后的 17 列。物理先验是 PPU-Former 框架的一部分，仅 PPU-Former 使用 17 列。
"""

import os
import re
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from models.transformer_wrapper import TransformerWrapper
from utils import create_save_paths, save_args_json, write_full_report, append_run_summary
from data_provider.split_utils import strict_chronological_split


# ========================== 配置区（你只需要改这里）==========================

# ---------- 实验身份 ----------
model_name = "Transformer"            # baseline 名字 → results/Transformer/...
des = "transformer_TSlib_std"

# ---------- 数据集（baseline 用 5 列原始 CSV，不含物理先验） ----------
dataset_name = "pv2017"                # pv2017 / pv2018 / pv2019（5 列原始）
year = None

# ---------- 预测任务（必须和 PPU-Former 保持一致） ----------
pred_len = 4                          # 1 / 4 / 8 / 24
lookback_len = 168                     # = seq_len，168h = 7 天
label_len = 48                         # decoder 起始 token 长度（与 Informer 一致）
num_variates = 5                       # 5 列原始：GHR / Py / WS / TP1 / AP
target_idx = 4                         # AP 索引（在 5 列里仍然是第 5 个 → idx=4）

# ---------- Transformer 模型超参（TSlib Transformer, d_model=64 → ~200K params） ----------
trans_d_model = 64
trans_n_heads = 4
trans_e_layers = 3
trans_d_layers = 1
trans_d_ff = 128
trans_dropout = 0.1
trans_activation = "gelu"
trans_factor = 5

# ---------- 训练超参（原论文用 warm-up + 1e-3 量级 lr，这里用裸 Adam + 大 lr 对齐论文设置） ----------
epochs = 200
batch_size = 128
learning_rate = 0.001951

# ---------- 数据划分（与 PPU-Former / Informer 一致） ----------
train_ratio = 0.8

# ---------- 输出 ----------
results_dir = "results"
loss_plot_ylim = (0, 20)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================== 工具函数（与 run_informer.py 完全相同） ==========================


def infer_year_from_dataset(name: str) -> int:
    match = re.search(r"(20\d{2})", name)
    if match:
        return int(match.group(1))
    raise ValueError(f"无法从 dataset_name='{name}' 推导年份，请显式设置 year")


def make_time_features(timestamps_str) -> np.ndarray:
    """
    与 zhouhaoyi/Informer2020/utils/timefeatures.py(timeenc=1, freq='h') 完全一致：
        4 通道 = [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear]
        全部归一化到 [-0.5, +0.5]
    """
    dt = pd.to_datetime(pd.Series(timestamps_str))
    hour_of_day  = dt.dt.hour            / 23.0  - 0.5
    day_of_week  = dt.dt.dayofweek       / 6.0   - 0.5
    day_of_month = (dt.dt.day - 1)       / 30.0  - 0.5
    day_of_year  = (dt.dt.dayofyear - 1) / 365.0 - 0.5
    return np.stack(
        [hour_of_day.values, day_of_week.values, day_of_month.values, day_of_year.values],
        axis=1,
    ).astype(np.float32)


def load_data(dataset_name):
    csv_path = os.path.join("dataset", f"{dataset_name}.csv")
    df = pd.read_csv(csv_path)
    timestamps = df.iloc[:, 0].values
    features = df.iloc[:, 1:].values.astype(np.float32)
    return features, timestamps


# ========================== 数据集（与 run_informer.py 一致） ==========================


class TimeSeriesDataset(Dataset):
    """返回 4 元组 (x, y, x_mark, y_mark)，与 run_informer.py 完全相同。"""

    def __init__(self, data, time_feats, lookback, label_len, pred_len, target_idx):
        self.data = data
        self.time_feats = time_feats
        self.lookback = lookback
        self.label_len = label_len
        self.pred_len = pred_len
        self.target_idx = target_idx
        self.length = len(data) - lookback - pred_len + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        s_begin = idx
        s_end = idx + self.lookback
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len

        x = self.data[s_begin:s_end]
        y = self.data[s_end:s_end + self.pred_len, self.target_idx]
        x_mark = self.time_feats[s_begin:s_end]
        y_mark = self.time_feats[r_begin:r_end]

        return (
            torch.FloatTensor(x),
            torch.FloatTensor(y),
            torch.FloatTensor(x_mark),
            torch.FloatTensor(y_mark),
        )


def create_dataloaders(features, timestamps, time_feats,
                       lookback, label_len, pred_len, train_ratio,
                       batch_size, target_idx):
    sp = strict_chronological_split(
        features, timestamps, lookback, pred_len, train_ratio,
        time_feats=time_feats,
    )

    target_min = sp["scaler"].data_min_[target_idx]
    target_max = sp["scaler"].data_max_[target_idx]

    train_dataset = TimeSeriesDataset(sp["train_data"], sp["train_time_feats"],
                                      lookback, label_len, pred_len, target_idx)
    test_dataset = TimeSeriesDataset(sp["test_data"], sp["test_time_feats"],
                                     lookback, label_len, pred_len, target_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    raw_test_target = sp["test_data_raw"][:, target_idx]

    return (
        train_loader, train_eval_loader, test_loader,
        sp["test_timestamps"], target_min, target_max, raw_test_target,
    )


# ========================== 训练与评估（与 run_informer.py 完全相同） ==========================


def train_one_epoch(model, loader, optimizer, criterion, target_idx):
    model.train()
    total_loss = 0.0
    for x, y, x_mark, y_mark in loader:
        x = x.to(device)
        y = y.to(device)
        x_mark = x_mark.to(device)
        y_mark = y_mark.to(device)

        optimizer.zero_grad()
        pred = model(x, x_mark, y_mark)              # [B, H, N]
        pred_ap = pred[:, :, target_idx]
        loss = criterion(pred_ap, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, target_min, target_max, target_idx):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y, x_mark, y_mark in loader:
            x = x.to(device)
            y = y.to(device)
            x_mark = x_mark.to(device)
            y_mark = y_mark.to(device)

            pred = model(x, x_mark, y_mark)
            pred_ap = pred[:, :, target_idx]
            loss = criterion(pred_ap, y)
            total_loss += loss.item()
            all_preds.append(pred_ap.cpu().numpy())
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


# ========================== 主流程 ==========================


def main():
    global year
    if year is None:
        year = infer_year_from_dataset(dataset_name)

    print(f"Device: {device}")
    print(f"Model: {model_name}  |  Dataset: {dataset_name} (year={year})")
    print(f"Lookback: {lookback_len}, Label: {label_len}, Pred: {pred_len}, Target idx: {target_idx}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")
    print("-" * 60)

    features, timestamps = load_data(dataset_name)
    time_feats = make_time_features(timestamps)
    num_samples = features.shape[0]
    print(f"Data loaded: {num_samples} samples, {features.shape[1]} features, "
          f"time_feats shape={time_feats.shape}")
    assert features.shape[1] == num_variates, (
        f"特征数对不上：CSV 有 {features.shape[1]} 列，但 num_variates={num_variates}。"
    )

    (train_loader, train_eval_loader, test_loader,
     test_timestamps, target_min, target_max, raw_test_target) = create_dataloaders(
        features, timestamps, time_feats,
        lookback_len, label_len, pred_len, train_ratio, batch_size, target_idx,
    )
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    model = TransformerWrapper(
        num_variates=num_variates,
        seq_len=lookback_len,
        label_len=label_len,
        pred_len=pred_len,
        d_model=trans_d_model,
        n_heads=trans_n_heads,
        e_layers=trans_e_layers,
        d_layers=trans_d_layers,
        d_ff=trans_d_ff,
        dropout=trans_dropout,
        activation=trans_activation,
        factor=trans_factor,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print("-" * 60)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

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

    t0 = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, target_idx)

        test_loss, test_mse, test_mae, test_r2, _, _ = evaluate(
            model, test_loader, criterion, target_min, target_max, target_idx,
        )

        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_mae"].append(test_mae)
        history["test_mae"].append(test_mae)
        history["train_mse"].append(test_mse)
        history["test_mse"].append(test_mse)
        history["train_r2"].append(test_r2)
        history["test_r2"].append(test_r2)

        print(
            f"Epoch [{epoch:3d}/{epochs}]  "
            f"Train Loss: {train_loss:.6f}  Test Loss: {test_loss:.6f}  "
            f"Test MAE: {test_mae:.4f}  Test R²: {test_r2:.4f}"
        )

    train_time_sec = time.time() - t0

    # 标准协议：训练跑满后保存最终模型，测试集只在这里评估一次
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
    metrics["best_epoch"] = int(epochs)   # 最终协议：报告的是跑满后最后一个 epoch 的模型
    metrics["train_time_sec"] = round(train_time_sec, 2)
    metrics["total_params"] = int(total_params)

    print("-" * 60)
    print(f"RMSE: {metrics['RMSE']:.6f}")
    print(f"MAE:  {metrics['MAE']:.6f}")
    print(f"R2:   {metrics['R2']:.6f}")
    print(f"Best epoch: {metrics['best_epoch']}")
    print(f"Train time: {metrics['train_time_sec']:.1f}s")

    config = {
        "model": model_name,
        "des": des,
        "dataset": dataset_name,
        "year": year,
        "num_samples": int(num_samples),
        "pred_len": pred_len,
        "label_len": label_len,
        "lookback_len": lookback_len,
        "num_variates": num_variates,
        "target_idx": target_idx,
        "trans_d_model": trans_d_model,
        "trans_n_heads": trans_n_heads,
        "trans_e_layers": trans_e_layers,
        "trans_d_layers": trans_d_layers,
        "trans_d_ff": trans_d_ff,
        "trans_dropout": trans_dropout,
        "trans_activation": trans_activation,
        "trans_factor": trans_factor,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "train_ratio": train_ratio,
        "device": str(device),
    }
    save_args_json(paths["args_json"], config, metrics)
    summary_path = append_run_summary(config=config, metrics=metrics, paths=paths)
    print(f"\nResults saved to: {paths['save_dir']}")
    print(f"Summary appended to: {summary_path}")


if __name__ == "__main__":
    main()
