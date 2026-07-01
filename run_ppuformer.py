import argparse

import torch

from model.iTransformer_PGIA import iTransformerPGIA
from trainers import run_with_early_stopping


# ========================== 配置区（后续主要改这里）==========================

# ---------- 可选实验版本 ----------
variant = "PPU_Full"

# ---------- 实验身份 ----------
model_name = "PPU_Full"
des = "frozen_robust_pso_top1"

# ---------- 数据集 ----------
dataset_name = "pv2017_ext"
year = None
input_type = "extended_17_features"

# ---------- 预测任务 ----------
pred_len = 4
lookback_len = 168
label_len = 48
num_variates = 17
target_idx = 4

# ---------- PPU-Former 结构超参 ----------
dim = 192
depth = 2
heads = 2
dim_head = 16
dropout = 0.16817512929497844
attn_dropout = 0.16817512929497844
ff_dropout = 0.16817512929497844

# ---------- PPU 专用优化超参 ----------
gate_lr_mult = 5.324064377906677
dsc_lr_mult = 0.6302068101206094
dsc_gamma_lr_mult = 1.024075658413647
dsc_gamma_bound = 0.009999999776482582

# ---------- PPU 模块开关 ----------
use_ppu = True
use_psg = True
use_wase = True
use_dsc = True
use_pgia = True
use_revin = True

# ---------- 训练超参 ----------
max_epochs = 500
batch_size = 128
learning_rate = 0.0001273536110892301
seed = 35040
weight_decay = 0.0

# ---------- 固定实验协议参数 ----------
train_ratio = 0.7
val_end_ratio = 0.8
patience = 30
min_delta = 1e-5

# ---------- 输出 ----------
results_dir = "results"
loss_plot_ylim = None


PPU_VARIANTS = {
    "PPU_Full": {
        "model_name": "PPU_Full",
        "use_ppu": True,
        "use_psg": True,
        "use_wase": True,
        "use_dsc": True,
        "use_pgia": True,
    },
    "PPU_NoPSG": {
        "model_name": "PPU_NoPSG",
        "use_ppu": True,
        "use_psg": False,
        "use_wase": True,
        "use_dsc": True,
        "use_pgia": True,
    },
    "PPU_NoWASE": {
        "model_name": "PPU_NoWASE",
        "use_ppu": True,
        "use_psg": True,
        "use_wase": False,
        "use_dsc": True,
        "use_pgia": True,
    },
    "PPU_NoDSC": {
        "model_name": "PPU_NoDSC",
        "use_ppu": True,
        "use_psg": True,
        "use_wase": True,
        "use_dsc": False,
        "use_pgia": True,
    },
    "PPU_NoPGIA": {
        "model_name": "PPU_NoPGIA",
        "use_ppu": True,
        "use_psg": True,
        "use_wase": True,
        "use_dsc": True,
        "use_pgia": False,
    },
}


def build_model(spec):
    h = spec.model_hparams
    flags = {
        "use_ppu": spec.extra_args.get("use_ppu", True),
        "use_psg": spec.extra_args.get("use_psg", True),
        "use_wase": spec.extra_args.get("use_wase", True),
        "use_dsc": spec.extra_args.get("use_dsc", True),
        "use_pgia": spec.extra_args.get("use_pgia", True),
    }
    return iTransformerPGIA(
        num_variates=spec.num_variates,
        lookback_len=spec.lookback_len,
        pred_length=spec.pred_len,
        target_idx=spec.target_idx,
        dim=h["dim"],
        depth=h["depth"],
        heads=h["heads"],
        dim_head=h["dim_head"],
        num_tokens_per_variate=1,
        use_reversible_instance_norm=h["use_revin"],
        flash_attn=True,
        attn_dropout=h["attn_dropout"],
        ff_dropout=h["ff_dropout"],
        phys_hidden_dim=32,
        psg_hidden_dim=32,
        wase_hidden_dim=64,
        dsc_kernels=(3, 5, 7),
        dsc_dropout=0.0,
        dsc_gamma_bound=h["dsc_gamma_bound"],
        **flags,
    )


def build_optimizer(model, spec):
    h = spec.model_hparams
    dsc_params = []
    dsc_gamma_params = []
    gate_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "dsc." in name:
            if name.endswith("gamma") or name == "dsc.gamma":
                dsc_gamma_params.append(param)
            else:
                dsc_params.append(param)
        elif param.numel() == 1 and "gamma" in name:
            gate_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {"params": other_params, "lr": spec.learning_rate},
        {"params": gate_params, "lr": spec.learning_rate * h["gate_lr_mult"]},
    ]
    if dsc_params:
        param_groups.append(
            {"params": dsc_params, "lr": spec.learning_rate * h["dsc_lr_mult"]}
        )
    if dsc_gamma_params:
        param_groups.append(
            {
                "params": dsc_gamma_params,
                "lr": spec.learning_rate * h["dsc_gamma_lr_mult"],
            }
        )
    return torch.optim.Adam(param_groups, weight_decay=spec.weight_decay)


def get_run_config(selected_variant: str | None = None):
    if selected_variant is None:
        active_variant = variant
        variant_cfg = {
            "model_name": model_name,
            "use_ppu": use_ppu,
            "use_psg": use_psg,
            "use_wase": use_wase,
            "use_dsc": use_dsc,
            "use_pgia": use_pgia,
        }
    else:
        active_variant = selected_variant
        variant_cfg = PPU_VARIANTS[active_variant]
    return {
        "variant": active_variant,
        "model_name": variant_cfg["model_name"],
        "des": des,
        "dataset_name": dataset_name,
        "year": year,
        "input_type": input_type,
        "pred_len": pred_len,
        "lookback_len": lookback_len,
        "label_len": label_len,
        "num_variates": num_variates,
        "target_idx": target_idx,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_epochs": max_epochs,
        "patience": patience,
        "min_delta": min_delta,
        "train_ratio": train_ratio,
        "val_end_ratio": val_end_ratio,
        "seed": seed,
        "results_dir": results_dir,
        "loss_plot_ylim": loss_plot_ylim,
        "use_time_features": False,
        "weight_decay": weight_decay,
        "model_hparams": {
            "dim": dim,
            "depth": depth,
            "heads": heads,
            "dim_head": dim_head,
            "dropout": dropout,
            "attn_dropout": attn_dropout,
            "ff_dropout": ff_dropout,
            "use_revin": use_revin,
            "gate_lr_mult": gate_lr_mult,
            "dsc_lr_mult": dsc_lr_mult,
            "dsc_gamma_lr_mult": dsc_gamma_lr_mult,
            "dsc_gamma_bound": dsc_gamma_bound,
        },
        "extra_config": {
            "variant": active_variant,
            "dim": dim,
            "depth": depth,
            "heads": heads,
            "dim_head": dim_head,
            "dropout": dropout,
            "attn_dropout": attn_dropout,
            "ff_dropout": ff_dropout,
            "use_revin": use_revin,
            "gate_lr_mult": gate_lr_mult,
            "dsc_lr_mult": dsc_lr_mult,
            "dsc_gamma_lr_mult": dsc_gamma_lr_mult,
            "dsc_gamma_bound": dsc_gamma_bound,
            "use_ppu": variant_cfg["use_ppu"],
            "use_psg": variant_cfg["use_psg"],
            "use_wase": variant_cfg["use_wase"],
            "use_dsc": variant_cfg["use_dsc"],
            "use_pgia": variant_cfg["use_pgia"],
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run PPU-Former final protocol")
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        choices=list(PPU_VARIANTS.keys()),
        help="可选；不传时使用顶部配置区，传入时覆盖为对应消融 variant",
    )
    parser.add_argument("--smoke", action="store_true", help="3 epochs smoke test")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_run_config(args.variant)
    run_with_early_stopping(
        model_builder=build_model,
        model_name=cfg["model_name"],
        des=cfg["des"],
        dataset_name=cfg["dataset_name"],
        year=cfg["year"],
        num_variates=cfg["num_variates"],
        input_type=cfg["input_type"],
        target_idx=cfg["target_idx"],
        lookback_len=cfg["lookback_len"],
        pred_len=cfg["pred_len"],
        label_len=cfg["label_len"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        max_epochs=cfg["max_epochs"],
        patience=cfg["patience"],
        min_delta=cfg["min_delta"],
        train_ratio=cfg["train_ratio"],
        val_end_ratio=cfg["val_end_ratio"],
        seed=cfg["seed"],
        results_dir=cfg["results_dir"],
        loss_plot_ylim=cfg["loss_plot_ylim"],
        extra_config=cfg["extra_config"],
        model_hparams=cfg["model_hparams"],
        use_time_features=cfg["use_time_features"],
        weight_decay=cfg["weight_decay"],
        optimizer_builder=build_optimizer,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
