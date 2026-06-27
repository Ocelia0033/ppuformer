# -*- coding: utf-8 -*-
"""
run_ablation_module_v1.py
=========================
PPU-Former 模块摸底实验（pv2017, pred_len=4, seed=35040）。

目标：
    在不动后 20% 真 test、不跑 PSO 的前提下，先用 validation 集合
    回答以下问题：
        - Full（PSG+WASE+DSC+PGIA）是否优于 vanilla iTransformer-17？
        - PSG / WASE / DSC / PGIA 单独打开时各自有效吗？
        - 哪些模块可能拖后腿？

数据划分（关键，**绝不碰后 20% test**）：
    1. 读 pv2017_ext.csv（17 变量）
    2. 取前 80% 行作为「训练阶段全部可用数据」
    3. 在这 80% 内部再用 strict_chronological_split(train_ratio=0.8)
       切成 inner_train(前 80% × 80% = 64%) + validation(前 80% × 20% = 16%)
    4. 后 20% test 完全不读、不参与任何统计

训练协议：
    - 模型：iTransformerPGIA（baseline 也用它，但 4 个 use_xxx 全 False，
            等价 iTransformer-17，控制变量更干净）
    - 优化器：AdamW（lr=1.9e-4, batch=32），与 run_ppu.py 默认对齐
    - 每个组开始前重置 seed=35040，保证组间公平
    - 每个 epoch 评 validation，按 **best val RMSE** 挑 best_epoch 报告该轮 metrics

使用：
    python -u run_ablation_module_v1.py
输出：
    results/_ablation_module_v1/<timestamp>/summary.csv
    results/_ablation_module_v1/<timestamp>/<group>/{loss.png, loss-zoom.png, R².png, ...}
"""

import os
import time
import json
import random
import copy
import hashlib
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from model.iTransformer_PGIA import iTransformerPGIA
from data_provider.split_utils import strict_chronological_split
from utils.reporters import write_full_report, save_raw_target_curve_168h
from utils.run_log import append_run_summary


# ========================== 配置 ==========================

DATASET_NAME = "pv2017_ext"
PRED_LEN = 4
LOOKBACK_LEN = 168
NUM_VARIATES = 17
TARGET_IDX = 4
TRAIN_RATIO_OUTER = 0.8   # 前 80% 作为「训练阶段数据」，后 20% 是真 test，本脚本不动
TRAIN_RATIO_INNER = 0.8   # 在前 80% 内部再切 80/20 → inner_train(64%) + val(16%)

SEED = 35040

# 训练超参（与 run_ppu.py 一致，做摸底所以 epoch 降到 50）
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1.9e-4
GATE_LR_MULT = 5
WEIGHT_DECAY = 0.0

# 模型超参（与 run_ppu.py 一致）
DIM = 128
DEPTH = 4
HEADS = 2
DIM_HEAD = 32
ATTN_DROPOUT = 0.1
FF_DROPOUT = 0.1
PHYS_HIDDEN_DIM = 32
PSG_HIDDEN_DIM = 64
WASE_HIDDEN_DIM = 64
DSC_KERNELS = (3, 5, 7)
DSC_DROPOUT = 0.0
USE_PPU = True             # PPU γ/α 渐进策略全程开启（统一对比）

# 优化器选择（统一记录，6 组完全一致；严禁在 summary 里写两种）
OPTIMIZER_NAME = "AdamW"   # 与 run_ppu.py 默认一致；baseline run.py 用的是 Adam，本脚本不使用

# 本轮实验的角色标签：仅用于 result.json 让后续阅读者一眼看清这是哪一阶段
EXPERIMENT_PHASE = "module_screening_first_round"  # 后续若有希望，改成 long_run_100/200ep 复查

RESULTS_BASE_DIR = "results/_ablation_module_v1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 6 组消融配置（顺序 = 报告顺序）
ABLATIONS: List[Dict] = [
    {"name": "iTransformer-17", "use_psg": False, "use_wase": False, "use_dsc": False, "use_pgia": False},
    {"name": "PSG_only",        "use_psg": True,  "use_wase": False, "use_dsc": False, "use_pgia": False},
    {"name": "WASE_only",       "use_psg": False, "use_wase": True,  "use_dsc": False, "use_pgia": False},
    {"name": "DSC_only",        "use_psg": False, "use_wase": False, "use_dsc": True,  "use_pgia": False},
    {"name": "PGIA_only",       "use_psg": False, "use_wase": False, "use_dsc": False, "use_pgia": True},
    {"name": "Full",            "use_psg": True,  "use_wase": True,  "use_dsc": True,  "use_pgia": True},
]

# 仅跑列表内组名（用于审计/快速回归测试）；None = 跑全部 6 组
RUN_ONLY_NAMES = ["iTransformer-17", "PGIA_only"]   # ← audit 验证阶段
# RUN_ONLY_NAMES = None   # ← 正式跑全部 6 组时改回 None


# ========================== 工具：seed、DataLoader ==========================


def set_seed(seed: int):
    """所有随机源一次性设定，保证 6 组初始化、shuffle 顺序、Dropout mask 完全对齐。

    目的不是让所有模型结果一样（模型结构不同当然结果不同），而是把"随机初始化、
    DataLoader shuffle、Dropout、CUDA 卷积"这些随机因素压成同一份，使各组之间
    可观察到的差异只来自模块开关（PSG / WASE / DSC / PGIA）本身。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn deterministic：让 Conv1d / 卷积反向也走确定算法，
    # 这样 DSC 分支多卷积核的初始化与反向梯度在组间完全可复现。
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_dataloader_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _worker_init_fn(worker_id: int):
    s = SEED + worker_id
    np.random.seed(s)
    random.seed(s)


# ========================== 工具：参数哈希 / 模块状态打印 ==========================


def _state_dict_hash(sd: Dict[str, torch.Tensor],
                     exclude_substrs: tuple = ()) -> str:
    """对一个 state_dict 计算稳定 SHA256 哈希，用于跨组断言初始权重一致。

    exclude_substrs : 跳过包含任一子串的 key（典型用法：剔除 'phys_bias' 后求
        「shared backbone」hash，让 use_pgia=True / False 的组也能直接对齐共享层）。
    """
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        if any(sub in k for sub in exclude_substrs):
            continue
        v = sd[k].detach().cpu().contiguous().float().numpy().tobytes()
        h.update(k.encode("utf-8"))
        h.update(v)
    return h.hexdigest()[:16]


def module_status(model: nn.Module) -> Dict[str, bool]:
    """返回该组模型中 4 个前端模块的存在性 + forward 中是否会执行。

    关闭的模块在 iTransformerPGIA 中既被设为 None，又在 forward 里被 `if self.use_xxx`
    跳过——只要"不参与 forward"这一条成立，is None 与否不强制要求。
    """
    return {
        "has_psg":         model.psg is not None,
        "psg_in_forward":  bool(model.use_psg),
        "has_wase":        model.wase is not None,
        "wase_in_forward": bool(model.use_wase),
        "has_dsc":         model.dsc is not None,
        "dsc_in_forward":  bool(model.use_dsc),
        "pgia_in_forward": bool(model.use_pgia),
    }


def sanity_check_forward_paths(splits: dict, anchors: Dict[str, Dict[str, torch.Tensor]]) -> None:
    """构造一份 baseline（全关）模型和一份 PSG_only 模型，做以下校验：

    (a) baseline 的 4 个 use_xxx 标志确实为 False，PSG_only 仅 use_psg=True。
    (b) 跑一次前向，两者输出 shape 一致；并通过 forward hook 验证：
        baseline 的 PSG/WASE/DSC 在 forward 期间**未被调用**；
        PSG_only 的 PSG 被调用，WASE/DSC 未被调用。
    """
    print("=" * 72)
    print("  Step 2: sanity-check —— 关闭的模块确实不参与 forward")
    print("=" * 72)

    def _mk(cfg):
        m = build_model(cfg).to(device)
        load_anchors_into_model(m, cfg, anchors)
        m.eval()
        return m

    cfg_baseline = {"name": "iTransformer-17", "use_psg": False, "use_wase": False, "use_dsc": False, "use_pgia": False}
    cfg_psg = {"name": "PSG_only", "use_psg": True, "use_wase": False, "use_dsc": False, "use_pgia": False}

    set_seed(SEED); mb = _mk(cfg_baseline)
    set_seed(SEED); mp = _mk(cfg_psg)

    # 注册 forward hook，记录哪些子模块被实际调用
    called: Dict[str, set] = {"baseline": set(), "psg_only": set()}

    def _hook_for(tag, modname):
        def _h(mod, inp, out):
            called[tag].add(modname)
        return _h

    for tag, m in (("baseline", mb), ("psg_only", mp)):
        if m.psg  is not None: m.psg.register_forward_hook(_hook_for(tag, "psg"))
        if m.wase is not None: m.wase.register_forward_hook(_hook_for(tag, "wase"))
        if m.dsc  is not None: m.dsc.register_forward_hook(_hook_for(tag, "dsc"))

    # 借一批 val 数据走一次前向（不更新权重）
    x_val = torch.stack(
        [splits["val_ds"][i][0] for i in range(min(8, len(splits["val_ds"])))]
    ).to(device)
    with torch.no_grad():
        _ = mb(x_val)
        _ = mp(x_val)

    status_b = module_status(mb)
    status_p = module_status(mp)
    print(f"  baseline.module_status = {status_b}")
    print(f"  baseline.modules_called_in_forward = {sorted(called['baseline'])}  (期望: 空集)")
    print(f"  PSG_only.module_status = {status_p}")
    print(f"  PSG_only.modules_called_in_forward = {sorted(called['psg_only'])}  (期望: ['psg'])")

    assert status_b["psg_in_forward"] is False
    assert status_b["wase_in_forward"] is False
    assert status_b["dsc_in_forward"] is False
    assert status_b["pgia_in_forward"] is False
    assert called["baseline"] == set(), \
        f"baseline 中竟有模块被 forward 调用：{called['baseline']}"
    assert status_p["psg_in_forward"] is True
    assert called["psg_only"] == {"psg"}, \
        f"PSG_only 中 forward 调用集合不符合预期：{called['psg_only']}"
    print("  ✓ 关闭的模块确认不参与 forward")


# ========================== Anchor：抽取「公共初始权重」 ==========================


def build_anchors() -> Dict[str, Dict[str, torch.Tensor]]:
    """
    构造 3 个 anchor 模型并提取 state_dict，作为「6 组通用的初始权重池」。

    用途
    ----
    iTransformerPGIA 的构造顺序固定为：
        PSG (if use_psg) → WASE (if use_wase) → DSC (if use_dsc) → backbone (always)
    模块开关不同时，backbone 之前消耗的 RNG 数量不同，会导致 backbone 内部各层
    权重在「不同 cfg」之间存在系统性偏差。直接 set_seed 无法消除这种偏差。

    本函数用 set_seed=SEED 分别构造：
        anchor_no_pgia  : 4 个模块全关 → 抽 backbone(use_pgia=False) 的 state_dict
        anchor_pgia     : 仅 PGIA=True → 抽 backbone(use_pgia=True)  的 state_dict
        anchor_modules  : 4 个模块全开 → 抽 PSG / WASE / DSC 各自的 state_dict

    6 组每组构造模型后再 load_state_dict(anchor, strict=False) 即可让：
        - 共享 backbone 权重在 6 组之间完全一致（按 use_pgia 二分）
        - 每个前端模块在「启用它」的所有组之间完全一致
    """
    print("=" * 72)
    print("  Step 1: 构造 anchor 模型，提取共享初始权重")
    print("  目的：消除「PSG/WASE/DSC 开关导致 backbone 之前 RNG 消耗不同」的偏差")
    print("=" * 72)

    def _mk(use_psg, use_wase, use_dsc, use_pgia) -> iTransformerPGIA:
        return iTransformerPGIA(
            num_variates=NUM_VARIATES,
            lookback_len=LOOKBACK_LEN,
            pred_length=PRED_LEN,
            target_idx=TARGET_IDX,
            dim=DIM, depth=DEPTH, heads=HEADS, dim_head=DIM_HEAD,
            num_tokens_per_variate=1,
            use_reversible_instance_norm=True,
            flash_attn=True,
            attn_dropout=ATTN_DROPOUT, ff_dropout=FF_DROPOUT,
            phys_hidden_dim=PHYS_HIDDEN_DIM,
            psg_hidden_dim=PSG_HIDDEN_DIM,
            wase_hidden_dim=WASE_HIDDEN_DIM,
            dsc_kernels=DSC_KERNELS, dsc_dropout=DSC_DROPOUT,
            use_psg=use_psg, use_wase=use_wase,
            use_dsc=use_dsc, use_pgia=use_pgia,
            use_ppu=USE_PPU,
        )

    set_seed(SEED)
    a0 = _mk(False, False, False, False)
    bb_no_pgia_sd = copy.deepcopy(a0.backbone.state_dict())
    print(f"  [anchor_no_pgia] backbone params: "
          f"{sum(p.numel() for p in a0.backbone.parameters()):,}  "
          f"hash={_state_dict_hash(bb_no_pgia_sd)}")

    set_seed(SEED)
    a1 = _mk(False, False, False, True)
    bb_pgia_sd = copy.deepcopy(a1.backbone.state_dict())
    print(f"  [anchor_pgia]    backbone params: "
          f"{sum(p.numel() for p in a1.backbone.parameters()):,}  "
          f"hash={_state_dict_hash(bb_pgia_sd)}")

    set_seed(SEED)
    af = _mk(True, True, True, True)
    psg_sd = copy.deepcopy(af.psg.state_dict())
    wase_sd = copy.deepcopy(af.wase.state_dict())
    dsc_sd = copy.deepcopy(af.dsc.state_dict())
    print(f"  [anchor_psg]   params: {sum(p.numel() for p in af.psg.parameters()):,}  "
          f"hash={_state_dict_hash(psg_sd)}")
    print(f"  [anchor_wase]  params: {sum(p.numel() for p in af.wase.parameters()):,}  "
          f"hash={_state_dict_hash(wase_sd)}")
    print(f"  [anchor_dsc]   params: {sum(p.numel() for p in af.dsc.parameters()):,}  "
          f"hash={_state_dict_hash(dsc_sd)}")

    del a0, a1, af
    return {
        "backbone_no_pgia": bb_no_pgia_sd,
        "backbone_pgia":    bb_pgia_sd,
        "psg":  psg_sd,
        "wase": wase_sd,
        "dsc":  dsc_sd,
    }


def load_anchors_into_model(model: iTransformerPGIA, cfg: Dict,
                            anchors: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, str]:
    """
    把 anchor 权重 load 进当前组的模型。返回各部位 hash，用于校验。

    - backbone：按 use_pgia 选 anchor_backbone_no_pgia / anchor_backbone_pgia。
      use_pgia=True 时 backbone 多出 phys_bias 子层，本就在 anchor_pgia 里，所以
      load_state_dict(strict=True) 应当严格匹配；为了对极少数键名差异容错，仍传 strict=False。
    - PSG/WASE/DSC：仅当该模块在本组启用时加载。
    """
    bb_anchor = anchors["backbone_pgia"] if cfg["use_pgia"] else anchors["backbone_no_pgia"]
    missing, unexpected = model.backbone.load_state_dict(bb_anchor, strict=False)
    if missing or unexpected:
        # 不应触发，但若结构对不上要立刻知道
        print(f"    ! backbone state_dict mismatch  missing={len(missing)}  unexpected={len(unexpected)}")

    if cfg["use_psg"]:
        model.psg.load_state_dict(anchors["psg"])
    if cfg["use_wase"]:
        model.wase.load_state_dict(anchors["wase"])
    if cfg["use_dsc"]:
        model.dsc.load_state_dict(anchors["dsc"])

    bb_sd = model.backbone.state_dict()
    bb_full_hash    = _state_dict_hash(bb_sd)
    # 剔除 phys_bias 后的「共享 backbone」hash，可让 use_pgia=True / False 两类组直接对齐
    bb_shared_hash  = _state_dict_hash(bb_sd, exclude_substrs=("phys_bias",))
    psg_hash  = _state_dict_hash(model.psg.state_dict())  if cfg["use_psg"]  else "-"
    wase_hash = _state_dict_hash(model.wase.state_dict()) if cfg["use_wase"] else "-"
    dsc_hash  = _state_dict_hash(model.dsc.state_dict())  if cfg["use_dsc"]  else "-"
    return {
        "backbone_full":   bb_full_hash,
        "backbone_shared": bb_shared_hash,
        "psg":  psg_hash,
        "wase": wase_hash,
        "dsc":  dsc_hash,
    }


# ========================== 数据 ==========================


class TimeSeriesDataset(Dataset):
    """与 run_ppu.py 完全一致：x=[L, 17], y=[H] (target=AP)。"""

    def __init__(self, data, lookback_len, pred_len, target_idx):
        self.data = data
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


def load_inner_train_and_val(dataset_name: str,
                             lookback_len: int,
                             pred_len: int,
                             train_ratio_outer: float,
                             train_ratio_inner: float):
    """
    返回 inner_train / validation 的 (loader, raw_target, timestamps_for_val,
    target_min, target_max, num_samples_total, val_samples)。

    严格不碰后 20% 真 test。
    """
    csv_path = os.path.join("dataset", f"{dataset_name}.csv")
    df = pd.read_csv(csv_path)
    timestamps = df.iloc[:, 0].values
    features = df.iloc[:, 1:].values.astype(np.float32)
    num_total = features.shape[0]
    assert features.shape[1] == NUM_VARIATES, (
        f"CSV 列数 {features.shape[1]} 与 NUM_VARIATES={NUM_VARIATES} 不一致"
    )

    # 步骤 1：截取前 80% 作为「训练阶段数据」，后 20% 留给真 test，本脚本不用
    outer_split = int(num_total * train_ratio_outer)
    feats_phase = features[:outer_split]
    ts_phase = timestamps[:outer_split]
    print(f"[Split] 总样本 {num_total} → 前 {train_ratio_outer*100:.0f}% "
          f"({outer_split} 行) 作为本次训练阶段数据；后 {num_total-outer_split} 行（真 test）不使用")

    # 步骤 2：对这 80% 再调用 strict_chronological_split → inner_train + val
    # strict_chronological_split 内部约定：scaler 仅 fit 在 features_phase[:inner_split]
    # = inner_train_raw 区间；features_phase[inner_split-lookback_len:] = val 区间
    # 仅 transform。features[outer_split:] = 真 test，本脚本根本没有传入它，自然 0 接触。
    sp = strict_chronological_split(
        feats_phase, ts_phase,
        lookback_len=lookback_len,
        pred_len=pred_len,
        train_ratio=train_ratio_inner,
        verbose=True,
    )

    # 明确审计 scaler 的来源（防止任何"scaler 偷偷见过 val/test"的可能性）
    inner_split_idx = int(len(feats_phase) * train_ratio_inner)
    print("[Scaler 审计] 仅 fit 在 inner_train 区间 feats_phase[:%d]，validation 只 transform，"
          "真 test 完全未被读取。" % inner_split_idx)
    print(f"  fit 数据形状      = {sp['split_info']['scaler_fit_range']}  "
          f"(对应原 CSV 行 [0, {inner_split_idx - 1}])")
    print(f"  val 数据来源     = features_phase[{inner_split_idx - lookback_len}:]  (含 lookback 上下文)")
    print(f"  test 区间        = features[{outer_split}:{num_total}]  ← 当前脚本未读取")

    target_min = float(sp["scaler"].data_min_[TARGET_IDX])
    target_max = float(sp["scaler"].data_max_[TARGET_IDX])

    inner_train_ds = TimeSeriesDataset(sp["train_data"], lookback_len, pred_len, TARGET_IDX)
    val_ds = TimeSeriesDataset(sp["test_data"], lookback_len, pred_len, TARGET_IDX)

    return {
        "inner_train_ds": inner_train_ds,
        "val_ds": val_ds,
        "val_timestamps": sp["test_timestamps"],
        "raw_val_target": sp["test_data_raw"][:, TARGET_IDX],
        "target_min": target_min,
        "target_max": target_max,
        "outer_total": num_total,
        "phase_total": outer_split,
        "inner_train_samples": len(inner_train_ds),
        "val_samples": len(val_ds),
    }


def build_loaders(splits: dict, batch_size: int, seed: int):
    g = make_dataloader_generator(seed)
    train_loader = DataLoader(
        splits["inner_train_ds"], batch_size=batch_size, shuffle=True,
        drop_last=True, generator=g, worker_init_fn=_worker_init_fn,
    )
    train_eval_loader = DataLoader(
        splits["inner_train_ds"], batch_size=batch_size, shuffle=False,
        drop_last=False, worker_init_fn=_worker_init_fn,
    )
    val_loader = DataLoader(
        splits["val_ds"], batch_size=batch_size, shuffle=False,
        drop_last=False, worker_init_fn=_worker_init_fn,
    )
    return train_loader, train_eval_loader, val_loader


# ========================== 训练 / 评估 ==========================


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        pred_ap = pred[:, :, TARGET_IDX]
        loss = criterion(pred_ap, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, target_min, target_max):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            pred_ap = pred[:, :, TARGET_IDX]
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
    rmse = float(np.sqrt(mse))
    return {
        "loss": avg_loss, "rmse": rmse, "mse": mse, "mae": mae, "r2": r2,
        "preds": preds_orig, "targets": targets_orig,
    }


def build_model(cfg: Dict) -> nn.Module:
    model = iTransformerPGIA(
        num_variates=NUM_VARIATES,
        lookback_len=LOOKBACK_LEN,
        pred_length=PRED_LEN,
        target_idx=TARGET_IDX,
        dim=DIM, depth=DEPTH, heads=HEADS, dim_head=DIM_HEAD,
        num_tokens_per_variate=1,
        use_reversible_instance_norm=True,
        flash_attn=True,
        attn_dropout=ATTN_DROPOUT, ff_dropout=FF_DROPOUT,
        phys_hidden_dim=PHYS_HIDDEN_DIM,
        psg_hidden_dim=PSG_HIDDEN_DIM,
        wase_hidden_dim=WASE_HIDDEN_DIM,
        dsc_kernels=DSC_KERNELS, dsc_dropout=DSC_DROPOUT,
        use_psg=cfg["use_psg"], use_wase=cfg["use_wase"],
        use_dsc=cfg["use_dsc"], use_pgia=cfg["use_pgia"],
        use_ppu=USE_PPU,
    ).to(device)
    return model


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    """与 run_ppu 相同的分组学习率：γ 标量门走 gate_lr_mult × lr。"""
    gate_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.numel() == 1 and "gamma" in name:
            gate_params.append(p)
        else:
            other_params.append(p)
    return torch.optim.AdamW([
        {"params": other_params, "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY},
        {"params": gate_params,  "lr": LEARNING_RATE * GATE_LR_MULT, "weight_decay": 0.0},
    ])


# ========================== 单组实验 ==========================


def run_one_group(cfg: Dict, splits: dict, group_dir: str,
                  anchors: Dict[str, Dict[str, torch.Tensor]]) -> Dict:
    print()
    print("=" * 72)
    print(f"  组: {cfg['name']}   use_psg={cfg['use_psg']}  use_wase={cfg['use_wase']}  "
          f"use_dsc={cfg['use_dsc']}  use_pgia={cfg['use_pgia']}")
    print("=" * 72)

    os.makedirs(group_dir, exist_ok=True)

    # ---------- 每组开始前 set seed，建 DataLoader ----------
    # 对齐：np / random / torch CPU / torch CUDA 的全部 RNG state
    set_seed(SEED)
    print(f"[seed] set_seed({SEED}) before DataLoader build")
    train_loader, train_eval_loader, val_loader = build_loaders(splits, BATCH_SIZE, SEED)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ---------- 模型创建前再 set 一次，再 load anchor ----------
    # build_loaders 用独立 generator，本不影响全局 RNG，但稳妥起见再 set 一次；
    # 然后用 anchors 把 backbone 与各前端模块覆盖成统一的初始权重 —— 这才是真正
    # 让 6 组「初始权重池」对齐的关键步骤（直接靠 set_seed 是做不到的，因为
    # iTransformerPGIA 内部构造顺序使各组在 backbone 之前消耗的 RNG 不同）。
    set_seed(SEED)
    print(f"[seed] set_seed({SEED}) before model build  (cfg={cfg['name']})")
    model = build_model(cfg)
    total_params = sum(p.numel() for p in model.parameters())

    init_hashes = load_anchors_into_model(model, cfg, anchors)
    print(f"Model parameters: {total_params:,}")
    print(f"[init hash] backbone_full   = {init_hashes['backbone_full']}")
    print(f"[init hash] backbone_shared = {init_hashes['backbone_shared']}  (剔除 phys_bias)")
    print(f"[init hash] psg={init_hashes['psg']}  wase={init_hashes['wase']}  dsc={init_hashes['dsc']}")
    print(f"[module_status] {module_status(model)}")

    optimizer = build_optimizer(model)
    criterion = nn.MSELoss()

    history = {
        "epochs": [],
        "train_loss": [], "test_loss": [],
        "train_mae": [],  "test_mae": [],
        "train_mse": [],  "test_mse": [],
        "train_r2": [],   "test_r2": [],
    }
    val_rmse_hist: List[float] = []
    best = {"epoch": 0, "rmse": float("inf"), "mse": 0.0, "mae": 0.0, "r2": -float("inf"),
            "preds": None, "targets": None}

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_eval = evaluate(model, val_loader, criterion,
                            splits["target_min"], splits["target_max"])

        # ★ 协议统一：MAE / RMSE / R² **每个 epoch** 都在「反归一化后的原始 AP 空间」
        #   上计算 train 与 val，二者尺度严格一致；只有 loss（MSE）保留在归一化空间。
        #   修复了上一版「train_mae 在 epoch 2 之后被 0.0 占位」导致的图表误导。
        tr_eval = evaluate(model, train_eval_loader, criterion,
                           splits["target_min"], splits["target_max"])

        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)            # 归一化空间 MSE
        history["test_loss"].append(val_eval["loss"])       # 归一化空间 MSE
        history["train_mae"].append(tr_eval["mae"])         # 反归一化空间
        history["test_mae"].append(val_eval["mae"])         # 反归一化空间
        history["train_mse"].append(tr_eval["mse"])         # 反归一化空间
        history["test_mse"].append(val_eval["mse"])         # 反归一化空间
        history["train_r2"].append(tr_eval["r2"])           # 反归一化空间
        history["test_r2"].append(val_eval["r2"])           # 反归一化空间

        val_rmse_hist.append(val_eval["rmse"])

        # 按 best val RMSE 选 best epoch
        if val_eval["rmse"] < best["rmse"]:
            best = {
                "epoch": epoch,
                "rmse": val_eval["rmse"],
                "mse": val_eval["mse"],
                "mae": val_eval["mae"],
                "r2": val_eval["r2"],
                "preds": val_eval["preds"], "targets": val_eval["targets"],
            }

        if epoch == 1 or epoch == EPOCHS or epoch % 5 == 0:
            print(f"  Epoch [{epoch:3d}/{EPOCHS}]  TrLoss={train_loss:.6f}  "
                  f"Val Loss={val_eval['loss']:.6f}  Val RMSE={val_eval['rmse']:.4f}  "
                  f"Val MAE={val_eval['mae']:.4f}  Val R²={val_eval['r2']:.4f}  "
                  f"(best@{best['epoch']} RMSE={best['rmse']:.4f})")

    elapsed = time.time() - t0

    # 写完整图（线性 y 轴、自适应 0.01/0.1/1/10 格式，由 reporters.py 保障）
    paths = {
        "loss_csv": os.path.join(group_dir, "loss.csv"),
        "loss_png": os.path.join(group_dir, "loss.png"),
        "loss_zoom_png": os.path.join(group_dir, "loss-zoom.png"),
        "mae_csv": os.path.join(group_dir, "mae.csv"),
        "mae_png": os.path.join(group_dir, "mae.png"),
        "mse_csv": os.path.join(group_dir, "mse.csv"),
        "mse_png": os.path.join(group_dir, "mse.png"),
        "r2_csv":  os.path.join(group_dir, "R2.csv"),
        "r2_png":  os.path.join(group_dir, "R2.png"),
        "overall_csv": os.path.join(group_dir, "Overall_indicators.csv"),
        "best_csv": os.path.join(group_dir, "Best.csv"),
        "best_png": os.path.join(group_dir, "Best.png"),
        "all_csv": os.path.join(group_dir, "ALL.csv"),
        "all_scatter_png": os.path.join(group_dir, "ALL-scatter.png"),
        "all_error_png": os.path.join(group_dir, "ALL-error.png"),
        "predictions_csv": os.path.join(group_dir, "predictions.csv"),
        "prediction_curve_png": os.path.join(group_dir, "prediction_curve.png"),
        "prediction_curve_168h_png": os.path.join(group_dir, "prediction_curve_168h.png"),
        "prediction_curve_168h_csv": os.path.join(group_dir, "prediction_curve_168h.csv"),
        "best_part_csv": os.path.join(group_dir, "Best-Part.csv"),
        "best_part_png": os.path.join(group_dir, "Best-Part.png"),

        # raw sanity：与模型无关，每组目录里都写一份，方便单独审计
        "raw_val_curve_168h_png": os.path.join(group_dir, "raw_validation_AP_168h.png"),
        "raw_val_curve_168h_csv": os.path.join(group_dir, "raw_validation_AP_168h.csv"),
    }
    # write_full_report 期望 all_preds=[N,H]、all_targets=[N,H]
    # best 中存的是反归一化后的 [N_val, pred_len] 数组
    # phase="val" → 所有图标题、CSV 列名都自动带 "Validation"
    write_full_report(
        paths=paths,
        history=history,
        all_preds=best["preds"],
        all_targets=best["targets"],
        raw_test_target=splits["raw_val_target"],
        test_timestamps=splits["val_timestamps"],
        lookback_len=LOOKBACK_LEN,
        pred_len=PRED_LEN,
        phase="val",
        skip_epochs_for_zoom=10,
    )

    # 单组结果落盘（独立 json，明确记录所有超参 & init hash）
    group_result = {
        "name": cfg["name"],
        "experiment_phase": EXPERIMENT_PHASE,
        "config": {k: cfg[k] for k in cfg if k != "name"},
        "module_status": module_status(model),
        "init_hashes": init_hashes,

        "val_best_epoch":  int(best["epoch"]),
        "val_rmse":  best["rmse"],
        "val_mae":   best["mae"],
        "val_mse":   best["mse"],
        "val_r2":    best["r2"],
        "val_rmse_last": val_rmse_hist[-1],
        "val_rmse_history": val_rmse_hist,
        "train_time_sec": round(elapsed, 1),
        "total_params": int(total_params),

        # ---------- 完整训练协议（避免 summary 和 result.json 出现含糊不清的 Adam/AdamW） ----------
        "dataset": DATASET_NAME,
        "seed": SEED,
        "optimizer": OPTIMIZER_NAME,
        "learning_rate": LEARNING_RATE,
        "gate_lr_mult": GATE_LR_MULT,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "pred_len": PRED_LEN,
        "lookback_len": LOOKBACK_LEN,
        "num_variates": NUM_VARIATES,
        "target_idx": TARGET_IDX,
        "dim": DIM, "depth": DEPTH, "heads": HEADS, "dim_head": DIM_HEAD,
        "attn_dropout": ATTN_DROPOUT, "ff_dropout": FF_DROPOUT,
        "phys_hidden_dim": PHYS_HIDDEN_DIM,
        "psg_hidden_dim": PSG_HIDDEN_DIM,
        "wase_hidden_dim": WASE_HIDDEN_DIM,
        "dsc_kernels": list(DSC_KERNELS),
        "dsc_dropout": DSC_DROPOUT,
        "use_ppu": USE_PPU,

        "split": {
            "outer_total": splits["outer_total"],
            "phase_total": splits["phase_total"],
            "inner_train_samples": splits["inner_train_samples"],
            "val_samples": splits["val_samples"],
            "train_ratio_outer": TRAIN_RATIO_OUTER,
            "train_ratio_inner": TRAIN_RATIO_INNER,
        },
    }
    with open(os.path.join(group_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(group_result, f, ensure_ascii=False, indent=2)

    # ★ 追加到全局 all_runs.txt（恢复 run.py / run_ppu.py 的旧习惯）
    #   注意：这一行写的是 **validation** 上的 RMSE/MAE/R²，不是 test；
    #   des 字段明确写 "module_screen_v1_val:<group>"，避免和 test 结果混淆。
    log_config = {
        "model": "iTransformerPGIA_ablation",
        "year":  int(2017 if "2017" in DATASET_NAME else 0),
        "pred_len": PRED_LEN,
        "des":   f"module_screen_v1_val:{cfg['name']}",
    }
    log_metrics = {
        "RMSE": best["rmse"],
        "MAE":  best["mae"],
        "R2":   best["r2"],
        "best_epoch": int(best["epoch"]),
        "train_time_sec": round(elapsed, 1),
    }
    log_paths = {"train_id": cfg["name"]}
    summary_path = append_run_summary(config=log_config, metrics=log_metrics, paths=log_paths)

    print(f"  → Best epoch={best['epoch']}  Val RMSE={best['rmse']:.4f}  "
          f"MAE={best['mae']:.4f}  R²={best['r2']:.4f}  耗时 {elapsed:.1f}s")
    print(f"  → 追加到 {summary_path}")
    return group_result


# ========================== 入口 ==========================


def main():
    print(f"Device: {device}")
    print(f"Phase: {EXPERIMENT_PHASE}  (50ep 摸底；正式实验需 100/200ep 复查)")
    print(f"Dataset: {DATASET_NAME} | pred_len={PRED_LEN} | seed={SEED} | epochs={EPOCHS}")
    print(f"训练协议: optimizer={OPTIMIZER_NAME} lr={LEARNING_RATE} gate_lr_mult={GATE_LR_MULT} "
          f"batch={BATCH_SIZE} weight_decay={WEIGHT_DECAY}")
    print(f"模型超参: dim={DIM} depth={DEPTH} heads={HEADS} dim_head={DIM_HEAD}")
    print("-" * 72)

    # ---------- 数据切分（前 80% 内部再切 80/20；后 20% test 完全不读取） ----------
    splits = load_inner_train_and_val(
        DATASET_NAME, LOOKBACK_LEN, PRED_LEN,
        TRAIN_RATIO_OUTER, TRAIN_RATIO_INNER,
    )
    print(f"[Split] inner_train_samples={splits['inner_train_samples']}  "
          f"val_samples={splits['val_samples']}  "
          f"(test={splits['outer_total']-splits['phase_total']} 行，当前阶段不读取、不评估)")
    print("-" * 72)

    # ---------- 实验输出目录 ----------
    stamp = time.strftime("%Y%m%d-%H%M%S")
    exp_dir = os.path.join(RESULTS_BASE_DIR, f"run_{stamp}")
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Experiment dir: {exp_dir}")

    # ---------- Raw sanity：与模型无关，先把 raw validation AP 的 168h 曲线 ----------
    #             夜间 PV 真值应当 ≈ 0；若 raw 曲线本身夜间出现峰，
    #             说明 dataset 时间戳 / target 列 / 预处理本身有 bug，必须先修数据。
    raw_png = os.path.join(exp_dir, "raw_validation_AP_168h.png")
    raw_csv = os.path.join(exp_dir, "raw_validation_AP_168h.csv")
    save_raw_target_curve_168h(
        png_path=raw_png,
        csv_path=raw_csv,
        raw_target=splits["raw_val_target"],
        timestamps=splits["val_timestamps"],
        lookback_len=LOOKBACK_LEN,
        phase="val",
        series_name="Active Power",
    )
    print(f"[Raw sanity] 已生成: {raw_png}  /  {raw_csv}")
    print(f"             请先打开 PNG 查看夜间是否 ≈ 0；若否，先查 dataset/timestamp/target，再谈模型。")

    # ---------- Step 1: 构造 anchors（共享初始权重池） ----------
    anchors = build_anchors()

    # ---------- Step 2: 模块关闭确实不参与 forward 的 sanity check ----------
    sanity_check_forward_paths(splits, anchors)

    # ---------- Step 3: 跑实验（按 RUN_ONLY_NAMES 过滤） ----------
    if RUN_ONLY_NAMES is not None:
        running = [c for c in ABLATIONS if c["name"] in RUN_ONLY_NAMES]
        print(f"\n[Filter] RUN_ONLY_NAMES={RUN_ONLY_NAMES} → 只跑 {len(running)} 组: "
              f"{[c['name'] for c in running]}")
    else:
        running = ABLATIONS
        print(f"\n[Filter] 跑全部 {len(running)} 组")

    summary_rows: List[Dict] = []
    all_init_hashes: List[Dict] = []
    for cfg in running:
        group_dir = os.path.join(exp_dir, cfg["name"])
        result = run_one_group(cfg, splits, group_dir, anchors)
        all_init_hashes.append({
            "name":  result["name"],
            "use_pgia": cfg["use_pgia"],
            "backbone_full":   result["init_hashes"]["backbone_full"],
            "backbone_shared": result["init_hashes"]["backbone_shared"],
            "psg":  result["init_hashes"]["psg"],
            "wase": result["init_hashes"]["wase"],
            "dsc":  result["init_hashes"]["dsc"],
        })
        summary_rows.append({
            "name": result["name"],
            "use_psg":  cfg["use_psg"],
            "use_wase": cfg["use_wase"],
            "use_dsc":  cfg["use_dsc"],
            "use_pgia": cfg["use_pgia"],
            "val_best_epoch": result["val_best_epoch"],
            "val_rmse": result["val_rmse"],
            "val_mae":  result["val_mae"],
            "val_r2":   result["val_r2"],
            "val_rmse_last": result["val_rmse_last"],
            "train_time_sec": result["train_time_sec"],
            "params": result["total_params"],

            # ★ 训练协议字段（每行都明确写出，避免歧义）
            "experiment_phase": EXPERIMENT_PHASE,
            "dataset": DATASET_NAME,
            "seed": SEED,
            "optimizer": OPTIMIZER_NAME,
            "learning_rate": LEARNING_RATE,
            "gate_lr_mult": GATE_LR_MULT,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "pred_len": PRED_LEN,
            "lookback_len": LOOKBACK_LEN,
            "dim": DIM, "depth": DEPTH, "heads": HEADS, "dim_head": DIM_HEAD,
        })

    df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(exp_dir, "module_screening_summary.csv")
    df.to_csv(summary_csv, index=False)

    # 单独把 init hash 表落盘，便于事后追溯各组起点是否一致
    hashes_csv = os.path.join(exp_dir, "init_hashes.csv")
    pd.DataFrame(all_init_hashes).to_csv(hashes_csv, index=False)

    # ---------- 终端总表 + hash 校验表 + 分析 ----------
    print()
    print("=" * 90)
    print(f"  模块摸底实验汇总  ({DATASET_NAME}, pred_len={PRED_LEN}, seed={SEED}, "
          f"optimizer={OPTIMIZER_NAME}, epochs={EPOCHS})")
    print("=" * 90)
    hdr = f"{'group':<18}{'val_RMSE':>12}{'val_MAE':>12}{'val_R²':>12}{'best_ep':>10}{'params':>14}"
    print(hdr); print("-" * len(hdr))
    for r in summary_rows:
        print(f"{r['name']:<18}{r['val_rmse']:>12.4f}{r['val_mae']:>12.4f}"
              f"{r['val_r2']:>12.4f}{r['val_best_epoch']:>10d}{r['params']:>14,}")
    print("-" * len(hdr))

    # ----- init hash 校验 -----
    print()
    print("[init-hash 校验]  6 组初始权重对齐情况")
    h_hdr = f"{'group':<18}{'use_pgia':>10}{'bb_full':>20}{'bb_shared':>20}{'psg':>20}{'wase':>20}{'dsc':>20}"
    print(h_hdr); print("-" * len(h_hdr))
    for r in all_init_hashes:
        print(f"{r['name']:<18}{str(r['use_pgia']):>10}"
              f"{r['backbone_full']:>20}{r['backbone_shared']:>20}"
              f"{r['psg']:>20}{r['wase']:>20}{r['dsc']:>20}")
    print("-" * len(h_hdr))

    # 自动断言：剔除 phys_bias 的 backbone_shared 在 6 组之间应当全部相同
    shared_bb_hashes = {r["backbone_shared"] for r in all_init_hashes}
    print(f"  → backbone_shared 唯一值数 = {len(shared_bb_hashes)}   "
          f"{'✓ 6 组共享 backbone 对齐' if len(shared_bb_hashes) == 1 else '✗ 共享 backbone 仍存在偏差'}")

    # 同 use_pgia 的组 backbone_full 应当相同（4 组 use_pgia=False / 2 组 use_pgia=True）
    bb_full_no_pgia = {r["backbone_full"] for r in all_init_hashes if not r["use_pgia"]}
    bb_full_pgia    = {r["backbone_full"] for r in all_init_hashes if r["use_pgia"]}
    print(f"  → backbone_full(use_pgia=False) 唯一值数 = {len(bb_full_no_pgia)}   "
          f"{'✓' if len(bb_full_no_pgia) == 1 else '✗'}")
    print(f"  → backbone_full(use_pgia=True)  唯一值数 = {len(bb_full_pgia)}   "
          f"{'✓' if len(bb_full_pgia) == 1 else '✗'}")

    # 启用 PSG/WASE/DSC 的组之间，对应模块 hash 应当相同
    for mod, groups in (("psg",  ("PSG_only", "Full")),
                        ("wase", ("WASE_only", "Full")),
                        ("dsc",  ("DSC_only", "Full"))):
        hs = {r[mod] for r in all_init_hashes if r["name"] in groups}
        print(f"  → 模块 {mod} 在 {groups} 间一致性: "
              f"{'✓ 一致' if len(hs) == 1 else '✗ 不一致'}  ({hs})")

    # ----- 模块效果分析 -----
    def _get_by_name(name):
        for r in summary_rows:
            if r["name"] == name:
                return r
        return None

    base = _get_by_name("iTransformer-17")
    full = _get_by_name("Full")

    def delta(a, b):
        return a - b

    if base is not None and full is not None:
        print()
        print("[Full vs iTransformer-17]")
        print(f"  ΔRMSE = {delta(full['val_rmse'], base['val_rmse']):+.4f}   "
              f"ΔMAE = {delta(full['val_mae'], base['val_mae']):+.4f}   "
              f"ΔR²  = {delta(full['val_r2'], base['val_r2']):+.4f}")
        print(f"  结论: {'Full 优于 baseline' if full['val_rmse'] < base['val_rmse'] else 'Full 未优于 baseline'}"
              f"  (按 val RMSE 比较)")

    if base is not None:
        print()
        print("[单模块 vs iTransformer-17]")
        for name in ["PSG_only", "WASE_only", "DSC_only", "PGIA_only"]:
            r = _get_by_name(name)
            if r is None:
                continue
            d = delta(r["val_rmse"], base["val_rmse"])
            tag = "+ 有效"   if d < -1e-4 else ("× 拖后腿" if d > 1e-4 else "~ 无明显差异")
            print(f"  {name:<10}  ΔRMSE={d:+.4f}  ΔMAE={delta(r['val_mae'], base['val_mae']):+.4f}  "
                  f"ΔR²={delta(r['val_r2'], base['val_r2']):+.4f}    [{tag}]")

    print()
    print(f"Summary CSV: {summary_csv}")
    print(f"InitHashes:  {hashes_csv}")
    print(f"Exp dir:     {exp_dir}")
    print(f"Phase tag:   {EXPERIMENT_PHASE}  ← 这是首轮摸底；后续如有希望，请用 100/200ep 复查")


if __name__ == "__main__":
    main()
