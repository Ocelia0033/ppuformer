# -*- coding: utf-8 -*-
"""
data_provider/split_utils.py
==============================
公共时间序列严格切分工具。

所有训练脚本必须调用本模块的 strict_chronological_split() 函数，
确保训练标签与测试标签在时间轴上零重叠，且 scaler 仅在训练区间 fit。

切分协议
--------
    split_idx = int(len(features) * train_ratio)

    训练集原始区间 : features[:split_idx]
        训练标签最大索引 = split_idx - 1

    测试集原始区间 : features[split_idx - lookback_len :]
        测试标签最小索引 = split_idx
        前 lookback_len 个点仅用作输入上下文，不属于泄露。

    Scaler : 仅 fit features[:split_idx]
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def strict_chronological_split(features, timestamps, lookback_len, pred_len,
                               train_ratio, time_feats=None, verbose=True):
    """
    严格按原始时间点切分，保证训练标签与测试标签零重叠。

    Parameters
    ----------
    features    : ndarray [T, N]    原始特征矩阵
    timestamps  : ndarray [T]       时间戳向量
    lookback_len: int               回看窗口长度
    pred_len    : int               预测步长
    train_ratio : float             训练集占比 (0, 1)
    time_feats  : ndarray [T, F] 或 None   时间特征矩阵（供 Informer 等使用）
    verbose     : bool              是否打印审计信息（PSO 搜索时设 False）

    Returns
    -------
    dict  包含以下键:
        train_data, test_data           : 归一化后的 ndarray
        train_data_raw, test_data_raw   : 原始 ndarray
        train_timestamps, test_timestamps
        train_time_feats, test_time_feats  (仅当 time_feats 不为 None)
        scaler                          : 已 fit 的 MinMaxScaler
        split_info                      : 审计信息 dict
    """
    T = len(features)
    split_idx = int(T * train_ratio)

    # ---- 原始数据切分 ----
    train_data_raw = features[:split_idx]
    test_data_raw = features[split_idx - lookback_len:]

    train_timestamps = timestamps[:split_idx]
    test_timestamps = timestamps[split_idx - lookback_len:]

    # ---- Scaler: 仅 fit 训练区间 ----
    scaler = MinMaxScaler()
    scaler.fit(train_data_raw)                    # fit 范围 = [0, split_idx-1]
    train_data = scaler.transform(train_data_raw)
    test_data = scaler.transform(test_data_raw)

    # ---- 时间特征（可选）----
    train_time_feats = None
    test_time_feats = None
    if time_feats is not None:
        train_time_feats = time_feats[:split_idx]
        test_time_feats = time_feats[split_idx - lookback_len:]

    # ---- 样本计数 ----
    train_samples = split_idx - lookback_len - pred_len + 1
    test_samples = T - split_idx - pred_len + 1

    # ---- 标签索引范围（原始空间） ----
    train_label_min = lookback_len
    train_label_max = split_idx - 1
    test_label_min = split_idx
    test_label_max = T - 1
    test_context_start = split_idx - lookback_len
    scaler_fit_end = split_idx - 1

    # ---- 审计信息 ----
    split_info = {
        "raw_total_len":     T,
        "split_idx":         split_idx,
        "train_raw_range":   f"[0, {split_idx - 1}]",
        "train_label_range": f"[{train_label_min}, {train_label_max}]",
        "test_context_range": f"[{test_context_start}, {split_idx - 1}]",
        "test_label_range":  f"[{test_label_min}, {test_label_max}]",
        "train_samples":     train_samples,
        "test_samples":      test_samples,
        "label_overlap":     False,
        "scaler_fit_range":  f"[0, {scaler_fit_end}]",
    }

    if verbose:
        print("=" * 60)
        print("  严格时间序列切分审计 (Strict Chronological Split Audit)")
        print("=" * 60)
        for k, v in split_info.items():
            print(f"  {k:25s} = {v}")
        print("=" * 60)

    # ---- 断言 ----
    assert train_label_max < test_label_min, (
        f"标签重叠！train_label_max={train_label_max} >= "
        f"test_label_min={test_label_min}"
    )
    assert scaler_fit_end <= split_idx - 1, (
        f"Scaler 拟合越界！scaler_fit_end={scaler_fit_end} > "
        f"split_idx-1={split_idx - 1}"
    )
    assert test_context_start == split_idx - lookback_len, (
        f"测试上下文起点错误！{test_context_start} != "
        f"{split_idx - lookback_len}"
    )
    assert train_samples > 0, f"训练样本数不足: {train_samples}"
    assert test_samples > 0, f"测试样本数不足: {test_samples}"

    result = {
        "train_data":       train_data,
        "test_data":        test_data,
        "train_data_raw":   train_data_raw,
        "test_data_raw":    test_data_raw,
        "train_timestamps": train_timestamps,
        "test_timestamps":  test_timestamps,
        "scaler":           scaler,
        "split_info":       split_info,
    }
    if time_feats is not None:
        result["train_time_feats"] = train_time_feats
        result["test_time_feats"] = test_time_feats

    return result
