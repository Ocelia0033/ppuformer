import os
import re
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from iTransformer import iTransformer
from utils import create_save_paths, save_args_json, write_full_report, append_run_summary
from data_provider.split_utils import strict_chronological_split


# ========================== 配置区（你只需要改这里）==========================

# ---------- 实验身份（影响保存路径） ----------
model_name = "iTransformer"   # 第1级目录名。换模型时改这个
des = "baseline"              # 实验描述，仅写入 args.json 用于区分（baseline / ablation_xx / tune1 ...）

# ---------- 数据集 ----------
dataset_name = "pv2017"       # dataset/ 下的 csv 文件名（不含 .csv）。pv2017 / pv2018 / pv2019
year = None                   # 留 None 自动从 dataset_name 抽（pv2017→2017）；要手动覆盖就填整数

# ---------- 预测任务（影响第2级目录 pl{pred_len}） ----------
pred_len = 4                  # 1=1h预测  4=4h预测  8=8h预测  24=24h预测
lookback_len = 168            # 回看窗口（小时数），168 = 7天
num_variates = 5              # 输入特征数（CSV 里第 2-6 列共 5 个）

# ---------- 模型超参 ----------
dim = 128                     # 隐藏维度
depth = 5                     # transformer 层数
heads = 2                     # 注意力头数
dim_head = 32                 # 每个 head 的维度

# ---------- 训练超参 ----------
# iTransformer 由于 RevIN 反归一化的特性，对 batch_size 比其它 baseline 敏感。
# 历史所有 R²≈0.95 的成功跑分（train1/4/5）都是 batch=32 拿到的；
# 改成 128 会导致每个 epoch 的梯度步数只有 ~62 次（vs 246 次），训不出来。
# 这是 iTransformer 的官方推荐配置（原论文 / TSlib scripts 在 ETT 类数据集都用 batch=16~32）。
epochs = 100
batch_size = 64
learning_rate = 0.000190

# ---------- 随机种子 ----------
seed = 35040                  # 固定种子，保证可复现

# ---------- 数据划分 ----------
train_ratio = 0.8             # 训练集占比

# ---------- 输出 ----------
results_dir = "results"       # 顶层结果目录
loss_plot_ylim = (0, 1)           # loss-zoom 纵轴范围

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import random
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# ========================== 数据集 ==========================


class TimeSeriesDataset(Dataset):
    def __init__(self, data, timestamps, lookback_len, pred_len):
        self.data = data
        self.timestamps = timestamps
        self.lookback_len = lookback_len
        self.pred_len = pred_len
        self.length = len(data) - lookback_len - pred_len + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.lookback_len]
        y = self.data[idx + self.lookback_len: idx + self.lookback_len + self.pred_len, -1]
        return torch.FloatTensor(x), torch.FloatTensor(y)


# ========================== 工具函数 ==========================


def infer_year_from_dataset(name: str) -> int:
    """从数据集名字里抽出 4 位年份，例如 pv2017 -> 2017。"""
    match = re.search(r"(20\d{2})", name)
    if match:
        return int(match.group(1))
    raise ValueError(f"无法从 dataset_name='{name}' 推导年份，请显式设置 year 变量")


def load_data(dataset_name):
    csv_path = os.path.join("dataset", f"{dataset_name}.csv")
    df = pd.read_csv(csv_path)
    timestamps = df.iloc[:, 0].values
    features = df.iloc[:, 1:].values.astype(np.float32)
    return features, timestamps


def create_dataloaders(features, timestamps, lookback_len, pred_len, train_ratio, batch_size):
    sp = strict_chronological_split(
        features, timestamps, lookback_len, pred_len, train_ratio,
    )

    target_idx = features.shape[1] - 1
    target_min = sp["scaler"].data_min_[target_idx]
    target_max = sp["scaler"].data_max_[target_idx]

    train_dataset = TimeSeriesDataset(sp["train_data"], sp["train_timestamps"], lookback_len, pred_len)
    test_dataset = TimeSeriesDataset(sp["test_data"], sp["test_timestamps"], lookback_len, pred_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    raw_test_target = sp["test_data_raw"][:, target_idx]

    return (
        train_loader, train_eval_loader, test_loader,
        sp["test_timestamps"], target_min, target_max, raw_test_target,
    )


# ========================== 训练与评估 ==========================


def train_one_epoch(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        pred_active_power = pred[:, :, -1]
        loss = criterion(pred_active_power, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate(model, loader, criterion, target_min, target_max):
    """在 loader 上做完整评估，返回归一化 MSE loss + 反归一化指标 + 反归一化的预测/真实值。"""
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            pred_active_power = pred[:, :, -1]
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


# ========================== 主流程 ==========================


def main():
    global year
    if year is None:
        year = infer_year_from_dataset(dataset_name)

    print(f"Device: {device}")
    print(f"Model: {model_name}  |  Dataset: {dataset_name} (year={year})")
    print(f"Lookback: {lookback_len}, Pred length: {pred_len}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")
    print("-" * 60)

    features, timestamps = load_data(dataset_name)
    num_samples = features.shape[0]
    print(f"Data loaded: {num_samples} samples, {features.shape[1]} features")

    (train_loader, train_eval_loader, test_loader,
     test_timestamps, target_min, target_max, raw_test_target) = create_dataloaders(
        features, timestamps, lookback_len, pred_len, train_ratio, batch_size,
    )
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    model = iTransformer(
        num_variates=num_variates,
        lookback_len=lookback_len,
        dim=dim,
        depth=depth,
        heads=heads,
        dim_head=dim_head,
        pred_length=pred_len,
        num_tokens_per_variate=1,
        use_reversible_instance_norm=True,
        flash_attn=True,
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
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)

        _, train_mse, train_mae, train_r2, _, _ = evaluate(
            model, train_eval_loader, criterion, target_min, target_max,
        )
        test_loss, test_mse, test_mae, test_r2, _, _ = evaluate(
            model, test_loader, criterion, target_min, target_max,
        )

        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_mae"].append(train_mae)
        history["test_mae"].append(test_mae)
        history["train_mse"].append(train_mse)
        history["test_mse"].append(test_mse)
        history["train_r2"].append(train_r2)
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
        model, test_loader, criterion, target_min, target_max,
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
        "lookback_len": lookback_len,
        "num_variates": num_variates,
        "dim": dim,
        "depth": depth,
        "heads": heads,
        "dim_head": dim_head,
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
