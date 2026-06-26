import os
import re
import sys
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
from utils import create_save_paths, save_args_json, write_full_report
from data_provider.split_utils import strict_chronological_split

# ========================== 日志收集 ==========================

_log_lines = []
_original_print = print


def print(*args, **kwargs):
    """重写 print，同时收集所有输出到 _log_lines"""
    msg = " ".join(str(a) for a in args)
    _log_lines.append(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")
    kwargs.setdefault('flush', True)
    _original_print(*args, **kwargs)


# ========================== 固定配置（你只需要改这里）==========================

# ---------- 实验身份 ----------
model_name = "iTransformer"   # 第1级目录名
des = "pso_best"              # 描述，用于区分这是 PSO 搜出来的最优结果

# ---------- 数据集 ----------
dataset_name = "pv2017"       # pv2017 / pv2018 / pv2019
year = None                   # None = 自动从 dataset_name 抽

# ---------- 预测任务 ----------
pred_len = 1                  # 1=1h预测  4=4h预测  8=8h预测  24=24h预测
lookback_len = 168            # 回看窗口（小时）
num_variates = 5              # 输入特征数

# ---------- 训练超参（dim/depth/heads/dim_head/lr 由 PSO 自动搜，不在这里设）----------
epochs = 100
train_ratio = 0.9

# ---------- 输出 ----------
results_dir = "results"
loss_plot_ylim = (0, 20)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================== PSO 搜索空间 ==========================

# 搜索维度: [dim, depth, heads, dim_head, log10(lr)]
PARAM_BOUNDS = np.array([
    [32, 256],    # dim
    [1, 6],       # depth
    [1, 8],       # heads
    [16, 64],     # dim_head
    [-5, -2],     # log10(learning_rate)
])

PARAM_NAMES = ["dim", "depth", "heads", "dim_head", "learning_rate"]

batch_size = 32

PSO_PARTICLES = 8
PSO_ITERATIONS = 15
PSO_W = 0.7
PSO_C1 = 1.5
PSO_C2 = 1.5


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


# ========================== 数据加载 ==========================


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


def create_dataloaders(features, timestamps, lookback_len, pred_len, train_ratio, batch_size,
                       verbose=True):
    sp = strict_chronological_split(
        features, timestamps, lookback_len, pred_len, train_ratio,
        verbose=verbose,
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


# ========================== 模型评估（PSO 搜索阶段，只关心 RMSE）==========================


def train_and_evaluate(dim, depth, heads, dim_head, learning_rate):
    """给定一组超参数，训练模型并返回测试 RMSE（作为 PSO 适应度值）。"""
    try:
        features, timestamps = load_data(dataset_name)
        (train_loader, _, test_loader,
         _, target_min, target_max, _) = create_dataloaders(
            features, timestamps, lookback_len, pred_len, train_ratio, batch_size,
            verbose=False,
        )

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

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        for _ in range(epochs):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred[:, :, -1], y)
                loss.backward()
                optimizer.step()

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                all_preds.append(pred[:, :, -1].cpu().numpy())
                all_targets.append(y.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_preds = all_preds * (target_max - target_min) + target_min
        all_targets = all_targets * (target_max - target_min) + target_min

        rmse = float(np.sqrt(mean_squared_error(all_targets.flatten(), all_preds.flatten())))
        return rmse

    except Exception as e:
        print(f"  [ERROR] 训练失败: {e}")
        return float('inf')


# ========================== 参数解码 ==========================


def decode_params(position):
    """将连续的粒子位置解码为合法的超参数。"""
    raw = position.flatten()
    for i in range(len(raw)):
        raw[i] = np.clip(raw[i], PARAM_BOUNDS[i, 0], PARAM_BOUNDS[i, 1])

    dim_options = [32, 64, 128, 256]
    dim = min(dim_options, key=lambda x: abs(x - raw[0]))
    depth = int(round(raw[1]))
    heads = int(round(raw[2]))
    dim_head_options = [16, 32, 64]
    dim_head = min(dim_head_options, key=lambda x: abs(x - raw[3]))
    learning_rate = 10 ** raw[4]

    return dim, depth, heads, dim_head, learning_rate


# ========================== PSO 实现 ==========================


class Particle:
    def __init__(self, dim):
        self.position = np.array([
            np.random.uniform(PARAM_BOUNDS[i, 0], PARAM_BOUNDS[i, 1])
            for i in range(dim)
        ]).reshape(1, dim)
        max_vel = (PARAM_BOUNDS[:, 1] - PARAM_BOUNDS[:, 0]) * 0.2
        self.velocity = np.random.uniform(-max_vel, max_vel).reshape(1, dim)
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')
        self.fitness = float('inf')


class PSOOptimizer:
    def __init__(self, dim, n_particles, n_iterations, w, c1, c2):
        self.dim = dim
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.global_best_position = np.zeros((1, dim))
        self.global_best_fitness = float('inf')
        self.fitness_history = []
        self.particles = [Particle(dim) for _ in range(n_particles)]

    def optimize(self):
        print("=" * 70)
        print("PSO 超参数优化开始")
        print(f"粒子数: {self.n_particles}, 迭代次数: {self.n_iterations}")
        print(f"每次评估训练 {epochs} 个 epoch")
        print("=" * 70)

        for iteration in range(self.n_iterations):
            print(f"\n{'='*70}")
            print(f"PSO 迭代 [{iteration + 1}/{self.n_iterations}]")
            print(f"{'='*70}")

            for p_idx, particle in enumerate(self.particles):
                params = decode_params(particle.position)
                dim, depth, heads, dim_head, lr = params

                print(f"\n  粒子 {p_idx + 1}/{self.n_particles}: "
                      f"dim={dim}, depth={depth}, heads={heads}, "
                      f"dim_head={dim_head}, lr={lr:.6f}")

                fitness = train_and_evaluate(dim, depth, heads, dim_head, lr)
                particle.fitness = fitness

                print(f"  -> RMSE = {fitness:.6f}")

                if fitness < particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = particle.position.copy()

                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle.position.copy()

            for particle in self.particles:
                r1 = np.random.rand(1, self.dim)
                r2 = np.random.rand(1, self.dim)
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
                  f"lr={best_params[4]:.6f}")

        return self.global_best_position, self.global_best_fitness, self.fitness_history


# ========================== 最终完整训练 ==========================


def train_one_epoch(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred[:, :, -1], y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate(model, loader, criterion, target_min, target_max):
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


def final_train_and_save(best_params, fitness_history):
    """用最优参数做一次完整训练并按 4 级目录结构保存全部产出。"""
    dim, depth, heads, dim_head, lr = best_params

    print("\n" + "=" * 70)
    print("使用最优超参数进行最终训练")
    print(f"dim={dim}, depth={depth}, heads={heads}, dim_head={dim_head}, "
          f"lr={lr:.6f}, batch_size={batch_size}, epochs={epochs}")
    print("=" * 70)

    features, timestamps = load_data(dataset_name)
    num_samples = features.shape[0]

    (train_loader, train_eval_loader, test_loader,
     test_timestamps, target_min, target_max, raw_test_target) = create_dataloaders(
        features, timestamps, lookback_len, pred_len, train_ratio, batch_size
    )

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
    print(f"模型参数量: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    paths = create_save_paths(
        model_name=model_name,
        year=year,
        pred_len=pred_len,
        base_dir=results_dir,
    )
    print(f"Save dir: {paths['save_dir']}  (train{paths['train_id']})")

    history = {
        "epochs": [],
        "train_loss": [], "test_loss": [],
        "train_mae": [],  "test_mae": [],
        "train_mse": [],  "test_mse": [],
        "train_r2": [],   "test_r2": [],
    }

    t0 = time.time()
    best_test_mse = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        _, train_mse, train_mae, train_r2, _, _ = evaluate(
            model, train_eval_loader, criterion, target_min, target_max
        )
        test_loss, test_mse, test_mae, test_r2, _, _ = evaluate(
            model, test_loader, criterion, target_min, target_max
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

        if test_mse < best_test_mse:
            best_test_mse = test_mse
            torch.save(model.state_dict(), paths["model_pth"])

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch [{epoch}/{epochs}]  "
                  f"Train Loss: {train_loss:.6f}  Test Loss: {test_loss:.6f}  "
                  f"Test MAE: {test_mae:.4f}  Test R²: {test_r2:.4f}")

    train_time_sec = time.time() - t0

    model.load_state_dict(torch.load(paths["model_pth"], map_location=device))
    _, _, _, _, all_preds, all_targets = evaluate(
        model, test_loader, criterion, target_min, target_max
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
    metrics["best_epoch"] = int(np.argmin(history["test_mse"]) + 1)
    metrics["train_time_sec"] = round(train_time_sec, 2)
    metrics["total_params"] = int(total_params)

    pso_curve_png = os.path.join(paths["save_dir"], "pso_convergence.png")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(fitness_history) + 1), fitness_history, "b-o", linewidth=2)
    plt.xlabel("PSO Iteration")
    plt.ylabel("Best RMSE")
    plt.title("PSO Convergence Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(pso_curve_png, dpi=150)
    plt.close()

    pso_log_path = os.path.join(paths["save_dir"], "pso_run.log")
    with open(pso_log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(_log_lines))
        f.write("\n")

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
        "learning_rate": float(lr),
        "epochs": epochs,
        "train_ratio": train_ratio,
        "device": str(device),
        "pso": {
            "particles": PSO_PARTICLES,
            "iterations": PSO_ITERATIONS,
            "w": PSO_W,
            "c1": PSO_C1,
            "c2": PSO_C2,
            "fitness_history": [float(x) for x in fitness_history],
        },
    }
    save_args_json(paths["args_json"], config, metrics)

    print(f"\n最终结果:")
    print(f"  RMSE: {metrics['RMSE']:.6f}")
    print(f"  MAE:  {metrics['MAE']:.6f}")
    print(f"  R2:   {metrics['R2']:.6f}")
    print(f"  Best epoch: {metrics['best_epoch']}")
    print(f"\n所有结果已保存到: {paths['save_dir']}")


# ========================== 主流程 ==========================


def main():
    global year
    if year is None:
        year = infer_year_from_dataset(dataset_name)

    print(f"Device: {device}")
    print(f"Model: {model_name}  |  Dataset: {dataset_name} (year={year})")
    print(f"Lookback: {lookback_len}, Pred length: {pred_len}")
    print(f"固定训练 Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"PSO 粒子数: {PSO_PARTICLES}, 迭代次数: {PSO_ITERATIONS}")
    print()

    pso = PSOOptimizer(
        dim=len(PARAM_BOUNDS),
        n_particles=PSO_PARTICLES,
        n_iterations=PSO_ITERATIONS,
        w=PSO_W,
        c1=PSO_C1,
        c2=PSO_C2,
    )

    best_position, best_fitness, fitness_history = pso.optimize()
    best_params = decode_params(best_position)
    dim, depth, heads, dim_head, lr = best_params

    print("\n" + "=" * 70)
    print("PSO 搜索完成!")
    print(f"最优 RMSE: {best_fitness:.6f}")
    print(f"最优超参数: dim={dim}, depth={depth}, heads={heads}, "
          f"dim_head={dim_head}, lr={lr:.6f}")
    print("=" * 70)

    final_train_and_save(best_params, fitness_history)


if __name__ == "__main__":
    main()
