import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run import build_model as build_itransformer_model
from run import get_run_config as get_itransformer_config
from run_ppuformer import build_model as build_ppu_model
from run_ppuformer import build_optimizer as build_ppu_optimizer
from run_ppuformer import get_run_config as get_ppu_config
from trainers import run_with_early_stopping


SUPPORTED_MODELS = {
    "iTransformer-5": {
        "config_loader": lambda: get_itransformer_config("iTransformer-5"),
        "model_builder": build_itransformer_model,
        "optimizer_builder": None,
    },
    "iTransformer-17": {
        "config_loader": lambda: get_itransformer_config("iTransformer-17"),
        "model_builder": build_itransformer_model,
        "optimizer_builder": None,
    },
    "PPU_Full": {
        "config_loader": lambda: get_ppu_config("PPU_Full"),
        "model_builder": build_ppu_model,
        "optimizer_builder": build_ppu_optimizer,
    },
    "PPU_NoPSG": {
        "config_loader": lambda: get_ppu_config("PPU_NoPSG"),
        "model_builder": build_ppu_model,
        "optimizer_builder": build_ppu_optimizer,
    },
    "PPU_NoWASE": {
        "config_loader": lambda: get_ppu_config("PPU_NoWASE"),
        "model_builder": build_ppu_model,
        "optimizer_builder": build_ppu_optimizer,
    },
    "PPU_NoDSC": {
        "config_loader": lambda: get_ppu_config("PPU_NoDSC"),
        "model_builder": build_ppu_model,
        "optimizer_builder": build_ppu_optimizer,
    },
    "PPU_NoPGIA": {
        "config_loader": lambda: get_ppu_config("PPU_NoPGIA"),
        "model_builder": build_ppu_model,
        "optimizer_builder": build_ppu_optimizer,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Unified early stopping runner")
    parser.add_argument(
        "--model",
        type=str,
        default="PPU_Full",
        choices=list(SUPPORTED_MODELS.keys()),
        help="选择要运行的模型配置",
    )
    parser.add_argument("--smoke", action="store_true", help="3 epochs smoke test")
    return parser.parse_args()


def main():
    args = parse_args()
    bundle = SUPPORTED_MODELS[args.model]
    cfg = bundle["config_loader"]()
    run_with_early_stopping(
        model_builder=bundle["model_builder"],
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
        extra_config=cfg.get("extra_config", cfg["model_hparams"]),
        model_hparams=cfg["model_hparams"],
        use_time_features=cfg["use_time_features"],
        weight_decay=cfg.get("weight_decay", 0.0),
        optimizer_builder=bundle["optimizer_builder"],
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
