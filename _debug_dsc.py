# -*- coding: utf-8 -*-
"""
_debug_dsc.py
=============
单独诊断 DSC 模块（不训练、不画曲线、不接 PSO）。

目的（对照 audit 第 5 条）：
    - 输入 x 的 shape / mean / std / min / max
    - DSC 输出 y 的 shape / mean / std / min / max
    - residual = y - x 的 mean / std / max_abs
    - DSC.gamma 当前数值
    - 同一 batch 在 model.train() vs model.eval() 下 DSC 输出是否异常背离
    - 检查 time 轴 / variate 轴是否被处理错（kernel 是否作用在 time 维而非 variate 维）
    - 检查 DSC 与 RevIN / Embedding 前后的尺度冲突：
        * RevIN 通常出现在 backbone 的 stem 层，把每个变量 z-score 化
        * DSC 在 backbone 之前（PSG/WASE 之后）作用于原始 17 维输入
        * 因此 DSC 的输入实际仍是 MinMaxScaler 归一化后的 [0,1] 区间数据
        * 如果 DSC 输出脱离 [0,1] 太多，后续 RevIN/Embedding 会被压崩

调用：
    python -u _debug_dsc.py

不修改模型源码，本脚本只做观测。修复策略由观测结果决定。
"""

import os
import sys
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 复用主脚本的常量与构造逻辑
import run_ablation_module_v1 as M
from layers.dsc_local import DSCLocalBranch

DEVICE = M.device
OUT_DIR = "results/_debug_dsc"
os.makedirs(OUT_DIR, exist_ok=True)


def _set_seed(seed: int = M.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _tensor_stats(t: torch.Tensor, name: str) -> dict:
    t = t.detach()
    return {
        "name":  name,
        "shape": tuple(t.shape),
        "mean":  float(t.mean()),
        "std":   float(t.std(unbiased=False)),
        "min":   float(t.min()),
        "max":   float(t.max()),
        "abs_max": float(t.abs().max()),
    }


def _print_row(stats: dict):
    print(f"  {stats['name']:<28}  shape={str(stats['shape']):<18}  "
          f"mean={stats['mean']:+.4f}  std={stats['std']:.4f}  "
          f"min={stats['min']:+.4f}  max={stats['max']:+.4f}")


# ============================================================================
# Step A：单独跑 DSCLocalBranch，看它在原始归一化输入上的行为
# ============================================================================


def debug_dsc_only():
    print("=" * 80)
    print("  Step A: 独立 DSC 模块（不接 backbone）")
    print("=" * 80)

    # 拿 inner_train / val 的 dataloader
    splits = M.load_inner_train_and_val(
        M.DATASET_NAME, M.LOOKBACK_LEN, M.PRED_LEN,
        M.TRAIN_RATIO_OUTER, M.TRAIN_RATIO_INNER,
    )
    val_ds = splits["val_ds"]
    loader = DataLoader(val_ds, batch_size=16, shuffle=False, drop_last=False)
    x, y = next(iter(loader))           # x:[B,L,N]=[16,168,17]
    x = x.to(DEVICE)

    print(f"\n[输入 x] (反归一化后是原始 AP，但这里是 MinMaxScaler 归一化空间 [0,1])")
    _print_row(_tensor_stats(x, "x_input (B,L,N)"))

    # 构造一份 DSC（与 run_ablation_module_v1.py 相同参数）
    _set_seed(M.SEED)
    dsc = DSCLocalBranch(
        num_variates=M.NUM_VARIATES,
        kernels=M.DSC_KERNELS,
        dropout=M.DSC_DROPOUT,
        use_ppu=M.USE_PPU,
    ).to(DEVICE)
    print(f"\n[DSC 模块] params={sum(p.numel() for p in dsc.parameters()):,}")
    print(f"  γ_dsc 初始值       = {float(dsc.gamma.detach().cpu()):+.6f}  (PPU 软启动起点)")
    print(f"  PWConv weight |·| = {float(dsc.pw.weight.detach().abs().sum().cpu()):.6f}  (期望接近 0，因为零初始化)")
    print(f"  PWConv bias |·|   = {float(dsc.pw.bias.detach().abs().sum().cpu()):.6f}    (期望严格 0)")
    print(f"  kernels           = {dsc.kernels}")

    # ---- 在 train() 与 eval() 下分别跑一次 ----
    rows = []
    for mode in ("train", "eval"):
        getattr(dsc, mode)()
        with torch.no_grad():
            y = dsc(x)
            r = y - x
            stats_x = _tensor_stats(x, f"x_input ({mode})")
            stats_y = _tensor_stats(y, f"y_DSC ({mode})")
            stats_r = _tensor_stats(r, f"residual y-x ({mode})")
        print(f"\n[mode={mode}]")
        _print_row(stats_x)
        _print_row(stats_y)
        _print_row(stats_r)
        rows.append({"stage": "dsc_only", "mode": mode, **stats_y})

    # ---- 关键诊断：γ_dsc=0.01 时残差应当被压成 ~1% ----
    print(f"\n  → 由于 γ_dsc={float(dsc.gamma):.4f}（PPU 软启动），"
          f"残差 (y-x) 的 abs_max 应当 ≤ 0.01 * (内部分支幅度)；"
          f"如果 >> 1.0，说明 γ 或 pw 初始化失败。")

    # ---- 检查 conv 是否对正确的轴（time 维 L=168）做卷积 ----
    # 期望：dw_convs 的输入是 [B, N, L]，kernel 沿 L 滑动；如果错把 N 当时间，
    # 输出 std 在 L 维方差会异常。
    print(f"\n[Axis sanity]")
    print(f"  DSC 内部 transpose 后 x'.shape = (B, N, L) = ({x.shape[0]}, {x.shape[2]}, {x.shape[1]})")
    print(f"  期望 Conv1d 沿 L 轴卷积；Conv1d 的 kernel_size={dsc.dw_convs[0].kernel_size},"
          f" groups={dsc.dw_convs[0].groups}（=num_variates → depthwise 每个变量自己一份卷积核）")
    print(f"  → 若 groups != N，则 DSC 误把 time/variate 维混在一起，需要修。")
    assert dsc.dw_convs[0].groups == M.NUM_VARIATES, "DSC 不是 depthwise！"

    return rows, dsc


# ============================================================================
# Step B：把 DSC 接进 iTransformerPGIA，对照 iTransformer-17 与 DSC_only 在
#         完全相同初始权重下 forward 一次的输出差异。
# ============================================================================


def debug_dsc_in_model():
    print()
    print("=" * 80)
    print("  Step B: DSC 与 iTransformerPGIA 协同行为对比 (相同 init, 相同 batch)")
    print("=" * 80)

    splits = M.load_inner_train_and_val(
        M.DATASET_NAME, M.LOOKBACK_LEN, M.PRED_LEN,
        M.TRAIN_RATIO_OUTER, M.TRAIN_RATIO_INNER,
    )
    val_ds = splits["val_ds"]
    x, y = val_ds[0]
    x = x.unsqueeze(0).to(DEVICE)       # [1, L, N]

    anchors = M.build_anchors()

    cfg_b = {"name": "iTransformer-17", "use_psg": False, "use_wase": False, "use_dsc": False, "use_pgia": False}
    cfg_d = {"name": "DSC_only",         "use_psg": False, "use_wase": False, "use_dsc": True,  "use_pgia": False}

    _set_seed(M.SEED); m_b = M.build_model(cfg_b); M.load_anchors_into_model(m_b, cfg_b, anchors)
    _set_seed(M.SEED); m_d = M.build_model(cfg_d); M.load_anchors_into_model(m_d, cfg_d, anchors)
    m_b.eval(); m_d.eval()

    # ---- 同一输入在两个模型上跑 ----
    with torch.no_grad():
        out_b = m_b(x)
        out_d = m_d(x)
        # 取 DSC 模块的中间输出
        h_in  = x
        h_dsc = m_d.dsc(x)              # DSC 后的张量，应当 ≈ x（γ=0.01）

    print(f"\n[输入 x] shape={tuple(x.shape)}  mean={float(x.mean()):+.4f}  std={float(x.std(unbiased=False)):.4f}")
    print(f"[DSC 后 h] shape={tuple(h_dsc.shape)}  mean={float(h_dsc.mean()):+.4f}  std={float(h_dsc.std(unbiased=False)):.4f}")
    print(f"  |h_dsc - x|_max = {float((h_dsc - x).abs().max()):.6f}    (γ=0.01 + pw=0 → 期望 ≈ 0)")
    print(f"\n[backbone 输出]  iTransformer-17 vs DSC_only  (init 完全相同，但 DSC_only 经过 DSC 一次)")
    print(f"  out_b: mean={float(out_b.mean()):+.4f}  std={float(out_b.std(unbiased=False)):.4f}")
    print(f"  out_d: mean={float(out_d.mean()):+.4f}  std={float(out_d.std(unbiased=False)):.4f}")
    print(f"  |out_d - out_b|_max = {float((out_d - out_b).abs().max()):.6f}    "
          f"(期望 ~ |γ| 量级；如果 >> 1，说明 PPU 软启动失效)")

    return m_d


# ============================================================================
# Step C：模拟「训练过程中 DSC.gamma 被推大」的副作用
# ============================================================================


def debug_dsc_when_gamma_grows(m_d):
    print()
    print("=" * 80)
    print("  Step C: 人为把 DSC.gamma 调到 0.1 / 0.5 / 1.0，看模型输出会爆吗")
    print("=" * 80)

    splits = M.load_inner_train_and_val(
        M.DATASET_NAME, M.LOOKBACK_LEN, M.PRED_LEN,
        M.TRAIN_RATIO_OUTER, M.TRAIN_RATIO_INNER,
    )
    val_ds = splits["val_ds"]
    loader = DataLoader(val_ds, batch_size=16, shuffle=False, drop_last=False)
    x, _ = next(iter(loader))
    x = x.to(DEVICE)

    m_d.eval()
    for g in (0.01, 0.1, 0.5, 1.0):
        with torch.no_grad():
            m_d.dsc.gamma.data.fill_(g)
            h = m_d.dsc(x)
            out = m_d(x)
        print(f"  γ_dsc={g:.2f}  → h_dsc.std={float(h.std(unbiased=False)):.4f}  "
              f"out.std={float(out.std(unbiased=False)):.4f}  "
              f"out.max={float(out.max()):.4f}  out.min={float(out.min()):+.4f}")
    print("  → 如果 out.std 随 γ 线性增长正常；若 γ=1.0 时 out.max>>1，则 DSC 把数值放大到了"
          " RevIN/embedding 难以吸收的量级。")


def main():
    print(f"Device: {DEVICE}\n")

    _set_seed(M.SEED)
    rows_a, _ = debug_dsc_only()
    m_d = debug_dsc_in_model()
    debug_dsc_when_gamma_grows(m_d)

    print("\n" + "=" * 80)
    print("  Debug 完成。请基于上述输出判断 DSC 是否实现/尺度有 bug。")
    print("=" * 80)
    print("  关键阈值检查（人工读结果）：")
    print("    A. γ_dsc 初始值应 ≈ 0.01，PWConv weight |·| ≈ 0，bias |·| 严格 0；")
    print("    B. DSC 后 h_dsc 与输入 x 的 abs_max 差 < 0.01；")
    print("    C. DSC_only.backbone 输出 vs baseline.backbone 输出 abs_max 差 ~ 0.01 量级；")
    print("    D. γ 从 0.01 增长到 1.0 时，out.std 线性增长且仍在 [-1, 1] 附近，")
    print("       否则说明 DSC 的卷积输出未被 RevIN/embedding 兼容。")


if __name__ == "__main__":
    main()
