# -*- coding: utf-8 -*-
"""
scripts/label_shuffle_sanity.py
================================
标签打乱 sanity check：训练标签随机打乱，测试标签不变。
若 R² 仍然很高，说明可能存在数据泄漏或评价错误。

Usage:
    python scripts/label_shuffle_sanity.py
"""

import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.iTransformer_PGIA import iTransformerPGIA
from data_provider.split_utils import strict_chronological_split
from run_ppuformer import (
    load_data, evaluate, device, seed, lookback_len, pred_len,
    target_idx, num_variates, train_ratio, batch_size, learning_rate,
    dim, depth, heads, dim_head,
)

EPOCHS = 50


class ShuffledLabelDataset(Dataset):
    """输入 x 不变，标签 y 在训练集内随机打乱。"""

    def __init__(self, data, timestamps, lookback_len, pred_len, target_idx, seed=35040):
        self.data = data
        self.timestamps = timestamps
        self.lookback_len = lookback_len
        self.pred_len = pred_len
        self.target_idx = target_idx
        self.length = len(data) - lookback_len - pred_len + 1

        rng = np.random.RandomState(seed)
        n = self.length
        self.shuffled_y = []
        for idx in range(n):
            y = data[idx + lookback_len: idx + lookback_len + pred_len, target_idx]
            self.shuffled_y.append(y.copy())
        flat = np.stack(self.shuffled_y)
        perm = rng.permutation(n)
        self.shuffled_y = flat[perm]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.lookback_len]
        y = self.shuffled_y[idx]
        return torch.FloatTensor(x), torch.FloatTensor(y)


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)[:, :, target_idx]
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def main():
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("=" * 60)
    print("  LABEL SHUFFLE SANITY CHECK")
    print(f"  epochs={EPOCHS}, modules=PSG+WASE+PGIA (no DSC)")
    print("=" * 60)

    features, timestamps = load_data("pv2017_ext")
    sp = strict_chronological_split(
        features, timestamps, lookback_len, pred_len, train_ratio,
    )
    t_min = sp["scaler"].data_min_[target_idx]
    t_max = sp["scaler"].data_max_[target_idx]

    train_ds = ShuffledLabelDataset(
        sp["train_data"], sp["train_timestamps"],
        lookback_len, pred_len, target_idx, seed=seed + 999,
    )
    test_ds = __import__("run_ppuformer", fromlist=["TimeSeriesDataset"]).TimeSeriesDataset(
        sp["test_data"], sp["test_timestamps"],
        lookback_len, pred_len, target_idx,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = iTransformerPGIA(
        num_variates=num_variates, lookback_len=lookback_len, pred_length=pred_len,
        target_idx=target_idx, dim=dim, depth=depth, heads=heads, dim_head=dim_head,
        use_psg=True, use_wase=True, use_dsc=False, use_pgia=True, use_ppu=True,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        test_loss, _, test_mae, test_r2, _, _ = evaluate(
            model, test_loader, criterion, t_min, t_max,
        )
        if epoch == 1 or epoch == EPOCHS or epoch % 10 == 0:
            print(
                f"Epoch [{epoch:2d}/{EPOCHS}]  train_loss={train_loss:.4f}  "
                f"test_loss={test_loss:.4f}  test_R2={test_r2:.4f}"
            )

    print("-" * 60)
    if test_r2 > 0.5:
        print(f"FAIL: shuffled labels but test R2={test_r2:.4f} still high — check leakage/eval")
    elif test_r2 > 0.0:
        print(f"WARN: shuffled labels but test R2={test_r2:.4f} mildly positive")
    else:
        print(f"PASS: shuffled labels => test R2={test_r2:.4f} (expected poor performance)")
    print("=" * 60)


if __name__ == "__main__":
    main()
