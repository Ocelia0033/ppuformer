import os
import re
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from model.iTransformer_PGIA import iTransformerPGIA
from utils import create_save_paths, save_args_json, write_full_report, append_run_summary
from data_provider.split_utils import strict_chronological_split


# ========================== 配置区（你只需要改这里）==========================

# ---------- 实验身份 ----------
# model_name 会根据下面的模块开关自动生成（见 main() 开头），不用手动改。
# 例如：PPU_PGIA / PPU_PGIA_PSG / PPU_Full / iTransformer17 / PPU_PGIA_noRevIN
# des 是自由文字描述，写入 args.json 和 all_runs.txt，方便你备注。
des = ""                      # 留空=自动生成，或手动写如 "test1" "50ep验证" 等

# ---------- 数据集 ----------
dataset_name = "pv2017_ext"   # 17 维扩展特征
year = None                   # 留 None 自动推导

# ---------- 预测任务 ----------
pred_len = 4
lookback_len = 168
num_variates = 17             # pv2017_ext 有 17 列特征
target_idx = 4                # features[:, 4] = Active_Power

# ---------- 模块开关（改 True/False 就行）----------
use_psg  = True               # Physics State Gate
use_wase = True              # Weather Aware Spectral Enhancement
use_dsc  = False              # Depthwise Separable Conv
use_pgia = True              # Physics Guided Instance-Aware
use_ppu  = True              # Progressive Physical Unlocking（γ 从 0 起步）
use_revin = True              # RevIN 可逆实例归一化

# ---------- 模型超参 ----------
dim = 128
depth = 4
heads = 2
dim_head = 32

# ---------- 训练超参 ----------
epochs = 50
batch_size = 32
learning_rate = 0.000190
weight_decay = 0.0
gate_lr_mult = 5              # γ 门参数的学习率倍率

# ---------- 随机种子 ----------
seed = 35040                  # 固定种子，保证可复现

# ---------- 数据划分 ----------
train_ratio = 0.8

# ---------- 输出 ----------
results_dir = "results"
loss_plot_ylim = None

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
        y = self.data[
            idx + self.lookback_len: idx + self.lookback_len + self.pred_len,
            self.target_idx,
        ]
        return torch.FloatTensor(x), torch.FloatTensor(y)


# ========================== 工具函数 ==========================


def infer_year_from_dataset(name: str) -> int:
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


def create_dataloaders(features, timestamps):
    sp = strict_chronological_split(
        features, timestamps, lookback_len, pred_len, train_ratio,
    )

    t_min = sp["scaler"].data_min_[target_idx]
    t_max = sp["scaler"].data_max_[target_idx]

    train_ds = TimeSeriesDataset(sp["train_data"], sp["train_timestamps"],
                                 lookback_len, pred_len, target_idx)
    test_ds = TimeSeriesDataset(sp["test_data"], sp["test_timestamps"],
                                lookback_len, pred_len, target_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    train_eval_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    raw_test_target = sp["test_data_raw"][:, target_idx]

    return (train_loader, train_eval_loader, test_loader,
            sp["test_timestamps"], t_min, t_max, raw_test_target)


# ========================== 训练与评估 ==========================


def train_one_epoch(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        pred_ap = pred[:, :, target_idx]
        loss = criterion(pred_ap, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate(model, loader, criterion, t_min, t_max):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            pred_ap = pred[:, :, target_idx]
            loss = criterion(pred_ap, y)
            total_loss += loss.item()
            all_preds.append(pred_ap.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    avg_loss = total_loss / len(loader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    preds_orig = all_preds * (t_max - t_min) + t_min
    targets_orig = all_targets * (t_max - t_min) + t_min

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

    # 根据开关自动生成 model_name（= 结果目录名）和 tag
    modules_on = []
    if use_psg:  modules_on.append("PSG")
    if use_wase: modules_on.append("WASE")
    if use_dsc:  modules_on.append("DSC")
    if use_pgia: modules_on.append("PGIA")

    if not modules_on:
        model_name = "iTransformer17"
    elif len(modules_on) == 4:
        model_name = "PPU_Full"
    else:
        model_name = "PPU_" + "_".join(modules_on)
    if not use_revin:
        model_name += "_noRevIN"

    tag = "+".join(modules_on) if modules_on else "allOff"
    if not use_revin:
        tag += "_noRevIN"

    # des 留空时自动用 tag
    actual_des = des if des else tag

    print(f"Device: {device}")
    print(f"Model: {model_name}  |  Dataset: {dataset_name} (year={year})")
    print(f"des: {actual_des}  |  Modules: {tag}")
    print(f"RevIN: {use_revin}  |  PPU: {use_ppu}")
    print(f"Lookback: {lookback_len}, Pred length: {pred_len}")
    print(f"Epochs: {epochs}, Batch: {batch_size}, LR: {learning_rate}")
    print("-" * 60)

    features, timestamps = load_data(dataset_name)
    print(f"Data loaded: {features.shape[0]} samples, {features.shape[1]} features")

    (train_loader, train_eval_loader, test_loader,
     test_timestamps, t_min, t_max, raw_test_target) = create_dataloaders(
        features, timestamps,
    )
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    model = iTransformerPGIA(
        num_variates=num_variates,
        lookback_len=lookback_len,
        pred_length=pred_len,
        target_idx=target_idx,
        dim=dim, depth=depth, heads=heads, dim_head=dim_head,
        num_tokens_per_variate=1,
        use_reversible_instance_norm=use_revin,
        flash_attn=True,
        use_psg=use_psg,
        use_wase=use_wase,
        use_dsc=use_dsc,
        use_pgia=use_pgia,
        use_ppu=use_ppu,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print("-" * 60)

    dsc_params, dsc_gamma_params, gate_params, other_params = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "dsc." in name:
            if "gamma" in name:
                dsc_gamma_params.append(p)
            else:
                dsc_params.append(p)
        elif p.numel() == 1 and "gamma" in name:
            gate_params.append(p)
        else:
            other_params.append(p)

    param_groups = [
        {"params": other_params, "lr": learning_rate, "weight_decay": weight_decay},
        {"params": gate_params, "lr": learning_rate * gate_lr_mult, "weight_decay": 0.0},
    ]
    if dsc_params:
        param_groups.append(
            {"params": dsc_params, "lr": learning_rate * 0.5, "weight_decay": weight_decay},
        )
    if dsc_gamma_params:
        param_groups.append(
            {"params": dsc_gamma_params, "lr": learning_rate * 1.0, "weight_decay": 0.0},
        )
    optimizer = torch.optim.AdamW(param_groups)
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
            model, train_eval_loader, criterion, t_min, t_max,
        )
        test_loss, test_mse, test_mae, test_r2, _, _ = evaluate(
            model, test_loader, criterion, t_min, t_max,
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

        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            print(
                f"Epoch [{epoch:3d}/{epochs}]  "
                f"Train Loss: {train_loss:.6f}  Test Loss: {test_loss:.6f}  "
                f"Test MAE: {test_mae:.4f}  Test R²: {test_r2:.4f}"
            )

    train_time_sec = time.time() - t0

    torch.save(model.state_dict(), paths["model_pth"])
    _, _, _, _, all_preds, all_targets = evaluate(
        model, test_loader, criterion, t_min, t_max,
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
    metrics["best_epoch"] = int(epochs)
    metrics["train_time_sec"] = round(train_time_sec, 2)
    metrics["total_params"] = int(total_params)

    print("-" * 60)
    print(f"RMSE: {metrics['RMSE']:.6f}")
    print(f"MAE:  {metrics['MAE']:.6f}")
    print(f"R2:   {metrics['R2']:.6f}")
    print(f"Train time: {metrics['train_time_sec']:.1f}s")

    config = {
        "model": model_name,
        "des": actual_des,
        "modules": tag,
        "dataset": dataset_name,
        "year": year,
        "pred_len": pred_len,
        "lookback_len": lookback_len,
        "num_variates": num_variates,
        "target_idx": target_idx,
        "use_psg": use_psg,
        "use_wase": use_wase,
        "use_dsc": use_dsc,
        "use_pgia": use_pgia,
        "use_ppu": use_ppu,
        "use_revin": use_revin,
        "dim": dim, "depth": depth, "heads": heads, "dim_head": dim_head,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "gate_lr_mult": gate_lr_mult,
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
