# -*- coding: utf-8 -*-
"""
run_baseline_with_pso_params.py
================================
用 PSO 搜出来的 backbone 参数（dim, depth, heads, dim_head, lr）重跑 iTransformer baseline。
保证"公平对比"——所有模型使用相同骨干超参数。

使用方法：
    1. PSO 搜索完后，把最优参数填入下方 BEST_PARAMS
    2. python -u run_baseline_with_pso_params.py

4 个 pred_len 各跑一次，每次 300 epoch。
"""

import run  # 复用 run.py 的完整训练流程


# ★★★ PSO 搜出来的 backbone 参数填在这里（PSO 跑完后更新）★★★
BEST_PARAMS = {
    "dim": 128,       # PSO 搜出来的值
    "depth": 5,       # PSO 搜出来的值
    "heads": 1,       # PSO 搜出来的值
    "dim_head": 32,   # PSO 搜出来的值
    "learning_rate": 0.000190,  # PSO 搜出来的值
}

PRED_LENS = [1, 4, 8, 24]
DATASET_NAME = "pv2017"


def main():
    total = len(PRED_LENS)
    done = 0

    for pl in PRED_LENS:
        done += 1
        print()
        print("=" * 70)
        print(f"  [{done}/{total}] iTransformer baseline (PSO params) | {DATASET_NAME} | pred_len={pl}")
        print(f"  dim={BEST_PARAMS['dim']}, depth={BEST_PARAMS['depth']}, "
              f"heads={BEST_PARAMS['heads']}, dim_head={BEST_PARAMS['dim_head']}, "
              f"lr={BEST_PARAMS['learning_rate']:.6f}")
        print("=" * 70)

        run.dataset_name = DATASET_NAME
        run.year = None
        run.pred_len = pl
        run.model_name = "iTransformer"
        run.des = "pso_ppu_backbone"
        run.dim = BEST_PARAMS["dim"]
        run.depth = BEST_PARAMS["depth"]
        run.heads = BEST_PARAMS["heads"]
        run.dim_head = BEST_PARAMS["dim_head"]
        run.learning_rate = BEST_PARAMS["learning_rate"]

        run.main()

    print()
    print("=" * 70)
    print(f"All {total} baseline runs done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
