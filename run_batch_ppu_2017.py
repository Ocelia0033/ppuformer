# -*- coding: utf-8 -*-
"""
批量跑 PPU-Former pv2017 消融实验
==================================
6 组配置 × 4 个 pred_len = 24 次跑分

----------------------------------------------------------------------
消融组：
    full          : 完整 PPU-Former                          （PAD+WASE+DSC+PGIA+PPU）
    wo_pad        : 去掉 PAD                                  （证明 PAD 有用）
    wo_wase_dsc   : 去掉 WASE 和 DSC（时序增强分支整体去掉）  （证明时序增强有用）
    wo_pgia       : 去掉 PGIA                                 （证明 PGIA 有用）
    wo_ppu        : 保留所有模块但关掉渐进解锁                （证明 PPU 渐进策略有用）
----------------------------------------------------------------------
使用方法（在 iTransformer-main 目录下）：

    python run_batch_ppu_2017.py

"""

import run_ppu


# (name,         model_name,                       use_psg, use_wase, use_dsc, use_pgia, use_ppu)
ABLATIONS = [
    ("full",        "iTransformer_PGIA",             True,  True,  True,  True,  True),
    ("wo_psg",      "iTransformer_PGIA_woPSG",       False, True,  True,  True,  True),
]

PRED_LENS = [4]                     # 先只跑 4h 消融

DATASET_NAME = "pv2017_ext"


def main():
    total = len(ABLATIONS) * len(PRED_LENS)
    done = 0

    for name, model_name, use_psg, use_wase, use_dsc, use_pgia, use_ppu in ABLATIONS:
        for pl in PRED_LENS:
            done += 1
            print()
            print("=" * 70)
            print(f"  [{done}/{total}] PPU-Former [{name}] | {DATASET_NAME} | pred_len={pl} | epochs={run_ppu.epochs}")
            print(f"  use_psg={use_psg}  use_wase={use_wase}  use_dsc={use_dsc}  "
                  f"use_pgia={use_pgia}  use_ppu={use_ppu}")
            print("=" * 70)

            run_ppu.dataset_name = DATASET_NAME
            run_ppu.year = None
            run_ppu.pred_len = pl
            run_ppu.model_name = model_name
            run_ppu.des = f"ablation_{name}"
            run_ppu.use_psg = use_psg
            run_ppu.use_wase = use_wase
            run_ppu.use_dsc = use_dsc
            run_ppu.use_pgia = use_pgia
            run_ppu.use_ppu = use_ppu

            run_ppu.main()

    print()
    print("=" * 70)
    print(f"All {total} runs done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
