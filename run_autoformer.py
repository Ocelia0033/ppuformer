# -*- coding: utf-8 -*-
"""
run_autoformer.py
==================
跑 Autoformer (Wu et al., NeurIPS 2021) baseline。

整体结构 1:1 对照 ``run_transformer.py`` / ``run_informer.py``，只改了 3 处：
    (1) 模型从 ``TransformerWrapper`` 换成 ``AutoformerWrapper``
    (2) 默认超参换成 TSlib scripts/Autoformer/ETTh1.sh 标准
        (d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048, factor=1, moving_avg=25)
    (3) config 字典里超参名前缀换成 ``auto_*``

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

from models.autoformer_wrapper import AutoformerWrapper
from utils import create_save_paths, save_args_json, write_full_report, append_run_summary


# ========================== 配置区 ==========================

model_name = "Autoformer"
des = "autoformer_TSlib_std"        # TSlib 标准配置 512/2048 + noTimeF（不用时间戳特征）

dataset_name = "pv2017"                # pv2017 / pv2018 / pv2019（5 列原始）
year = None

pred_len = 24
lookback_len = 168
label_len = 48
num_variates = 5                       # 5 列原始：GHR / Py / WS / TP1 / AP
target_idx = 4                         # AP 索引（在 5 列里仍然是第 5 个 → idx=4）

# ---------- Autoformer 超参（与 iTransformer 参数量对齐，保证公平对比） ----------
auto_d_model = 128
auto_n_heads = 4
auto_e_layers = 2
auto_d_layers = 1
auto_d_ff = 256
auto_dropout = 0.1                     # 与 Transformer baseline 统一（TSlib 默认 0.1）
auto_activation = "gelu"
auto_factor = 1                        # AutoCorrelation 取 top-k 时的乘数
auto_moving_avg = 25                   # 趋势分解滑动平均核（论文默认）
auto_embed = "timeF"
auto_freq = "h"

epochs = 300
batch_size = 32
learning_rate = 0.000190
train_ratio = 0.8

results_dir = "results"
loss_plot_ylim = (0, 20)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================== 工具函数 ==========================


def infer_year_from_dataset(name: str) -> int:
    match = re.search(r"(20\d{2})", name)
    if match:
        return int(match.group(1))
    raise ValueError(f"无法从 dataset_name='{name}' 推导年份")


def make_time_features(timestamps_str) -> np.ndarray:
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


# ========================== 数据集（4 元组返回） ==========================


class TimeSeriesDataset(Dataset):
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
    total_samples = len(features) - lookback - pred_len + 1
    train_size = int(total_samples * train_ratio)

    train_end_idx = train_size + lookback + pred_len - 1
    train_data_raw = features[:train_end_idx]
    train_time_feats = time_feats[:train_end_idx]
    test_data_raw = features[train_size:]
    test_timestamps = timestamps[train_size:]
    test_time_feats = time_feats[train_size:]

    scaler = MinMaxScaler()
    scaler.fit(train_data_raw)
    train_data = scaler.transform(train_data_raw)
    test_data = scaler.transform(test_data_raw)

    target_min = scaler.data_min_[target_idx]
    target_max = scaler.data_max_[target_idx]

    train_dataset = TimeSeriesDataset(train_data, train_time_feats, lookback, label_len, pred_len, target_idx)
    test_dataset = TimeSeriesDataset(test_data, test_time_feats, lookback, label_len, pred_len, target_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    raw_test_target = test_data_raw[:, target_idx]

    return (
        train_loader, train_eval_loader, test_loader,
        test_timestamps, target_min, target_max, raw_test_target,
    )


# ========================== 训练与评估 ==========================


def train_one_epoch(model, loader, optimizer, criterion, target_idx):
    model.train()
    total_loss = 0.0
    for x, y, x_mark, y_mark in loader:
        x = x.to(device); y = y.to(device)
        x_mark = x_mark.to(device); y_mark = y_mark.to(device)
        optimizer.zero_grad()
        pred = model(x, x_mark, y_mark)
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
            x = x.to(device); y = y.to(device)
            x_mark = x_mark.to(device); y_mark = y_mark.to(device)
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
    pf, tf = preds_orig.flatten(), targets_orig.flatten()

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
    print(f"Data loaded: {num_samples} samples, {features.shape[1]} features")
    assert features.shape[1] == num_variates

    (train_loader, train_eval_loader, test_loader,
     test_timestamps, target_min, target_max, raw_test_target) = create_dataloaders(
        features, timestamps, time_feats,
        lookback_len, label_len, pred_len, train_ratio, batch_size, target_idx,
    )
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    model = AutoformerWrapper(
        num_variates=num_variates,
        seq_len=lookback_len,
        label_len=label_len,
        pred_len=pred_len,
        d_model=auto_d_model,
        n_heads=auto_n_heads,
        e_layers=auto_e_layers,
        d_layers=auto_d_layers,
        d_ff=auto_d_ff,
        dropout=auto_dropout,
        activation=auto_activation,
        factor=auto_factor,
        moving_avg=auto_moving_avg,
        embed=auto_embed,
        freq=auto_freq,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print("-" * 60)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    paths = create_save_paths(
        model_name=model_name, year=year, pred_len=pred_len, base_dir=results_dir,
    )
    print(f"Save dir: {paths['save_dir']}  (train{paths['train_id']})")
    print("-" * 60)

    history = {
        "epochs": [], "train_loss": [], "test_loss": [],
        "train_mae": [],  "test_mae": [],
        "train_mse": [],  "test_mse": [],
        "train_r2": [],   "test_r2": [],
    }

    t0 = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, target_idx)

        test_loss, test_mse, test_mae, test_r2, _, _ = evaluate(
            model, test_loader, criterion, target_min, target_max, target_idx)

        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_mae"].append(test_mae); history["test_mae"].append(test_mae)
        history["train_mse"].append(test_mse); history["test_mse"].append(test_mse)
        history["train_r2"].append(test_r2);   history["test_r2"].append(test_r2)

        print(
            f"Epoch [{epoch:3d}/{epochs}]  "
            f"Train Loss: {train_loss:.6f}  Test Loss: {test_loss:.6f}  "
            f"Test MAE: {test_mae:.4f}  Test R²: {test_r2:.4f}"
        )

    train_time_sec = time.time() - t0

    # 标准协议：训练跑满后保存最终模型，测试集只在这里评估一次
    torch.save(model.state_dict(), paths["model_pth"])
    _, _, _, _, all_preds, all_targets = evaluate(
        model, test_loader, criterion, target_min, target_max, target_idx)

    metrics = write_full_report(
        paths=paths, history=history, all_preds=all_preds, all_targets=all_targets,
        raw_test_target=raw_test_target, test_timestamps=test_timestamps,
        lookback_len=lookback_len, pred_len=pred_len, loss_ylim=loss_plot_ylim,
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
        "model": model_name, "des": des,
        "dataset": dataset_name, "year": year,
        "num_samples": int(num_samples),
        "pred_len": pred_len, "label_len": label_len,
        "lookback_len": lookback_len, "num_variates": num_variates,
        "target_idx": target_idx,
        "auto_d_model": auto_d_model, "auto_n_heads": auto_n_heads,
        "auto_e_layers": auto_e_layers, "auto_d_layers": auto_d_layers,
        "auto_d_ff": auto_d_ff, "auto_dropout": auto_dropout,
        "auto_activation": auto_activation, "auto_factor": auto_factor,
        "auto_moving_avg": auto_moving_avg,
        "auto_embed": auto_embed, "auto_freq": auto_freq,
        "batch_size": batch_size, "learning_rate": learning_rate,
        "epochs": epochs, "train_ratio": train_ratio,
        "device": str(device),
    }
    save_args_json(paths["args_json"], config, metrics)
    summary_path = append_run_summary(config=config, metrics=metrics, paths=paths)
    print(f"\nResults saved to: {paths['save_dir']}")
    print(f"Summary appended to: {summary_path}")


if __name__ == "__main__":
    main()
