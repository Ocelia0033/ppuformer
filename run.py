import argparse

from iTransformer import iTransformer
from trainers import run_with_early_stopping


# ========================== 配置区（后续主要改这里）==========================

# ---------- 可选实验版本 ----------
variant = "iTransformer-5"

# ---------- 实验身份 ----------
model_name = "iTransformer_5"
des = "itransformer_baseline"

# ---------- 数据集 ----------
dataset_name = "pv2017"
year = None
input_type = "original_5_features"

# ---------- 预测任务 ----------
pred_len = 4
lookback_len = 168
label_len = 48
num_variates = 5
target_idx = 4

# ---------- iTransformer 模型超参 ----------
dim = 128
depth = 3
heads = 4
dim_head = 32
dropout = 0.1
attn_dropout = 0.1
ff_dropout = 0.1
use_revin = True

# ---------- 训练超参 ----------
max_epochs = 500
batch_size = 64
learning_rate = 0.00019
seed = 35040

# ---------- 固定实验协议参数 ----------
train_ratio = 0.7
val_end_ratio = 0.8
patience = 30
min_delta = 1e-5

# ---------- 输出 ----------
results_dir = "results"
loss_plot_ylim = (0, 20)


ITRANSFORMER_VARIANTS = {
    "iTransformer-5": {
        "model_name": "iTransformer_5",
        "dataset_name": "pv2017",
        "num_variates": 5,
        "input_type": "original_5_features",
    },
    "iTransformer-17": {
        "model_name": "iTransformer_17",
        "dataset_name": "pv2017_ext",
        "num_variates": 17,
        "input_type": "extended_17_features",
    },
}


def build_model(spec):
    h = spec.model_hparams
    return iTransformer(
        num_variates=spec.num_variates,
        lookback_len=spec.lookback_len,
        pred_length=spec.pred_len,
        dim=h["dim"],
        depth=h["depth"],
        heads=h["heads"],
        dim_head=h["dim_head"],
        num_tokens_per_variate=1,
        use_reversible_instance_norm=h["use_revin"],
        flash_attn=True,
        attn_dropout=h["attn_dropout"],
        ff_dropout=h["ff_dropout"],
    )


def get_run_config(selected_variant: str | None = None):
    if selected_variant is None:
        active_variant = variant
        variant_cfg = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "num_variates": num_variates,
            "input_type": input_type,
        }
    else:
        active_variant = selected_variant
        variant_cfg = ITRANSFORMER_VARIANTS[active_variant]
    return {
        "variant": active_variant,
        "model_name": variant_cfg["model_name"],
        "des": des,
        "dataset_name": variant_cfg["dataset_name"],
        "year": year,
        "input_type": variant_cfg["input_type"],
        "pred_len": pred_len,
        "lookback_len": lookback_len,
        "label_len": label_len,
        "num_variates": variant_cfg["num_variates"],
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
        "model_hparams": {
            "dim": dim,
            "depth": depth,
            "heads": heads,
            "dim_head": dim_head,
            "dropout": dropout,
            "attn_dropout": attn_dropout,
            "ff_dropout": ff_dropout,
            "use_revin": use_revin,
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
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run iTransformer final protocol")
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        choices=list(ITRANSFORMER_VARIANTS.keys()),
        help="可选；不传时使用顶部配置区，传入时覆盖为对应 variant",
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
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
