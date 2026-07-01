import argparse

from models.lstm_wrapper import LSTMWrapper
from trainers import run_with_early_stopping


# ========================== 配置区（后续主要改这里）==========================

# ---------- 实验身份 ----------
model_name = "LSTM"
des = "lstm_baseline"

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

# ---------- LSTM 模型超参 ----------
lstm_hidden_size = 128
lstm_num_layers = 2
lstm_dropout = 0.1
lstm_bidirectional = False

# ---------- 训练超参 ----------
max_epochs = 500
batch_size = 64
learning_rate = 0.0003
seed = 35040

# ---------- 固定实验协议参数 ----------
train_ratio = 0.7
val_end_ratio = 0.8
patience = 30
min_delta = 1e-5

# ---------- 输出 ----------
results_dir = "results"
loss_plot_ylim = (0, 20)


def build_model(spec):
    h = spec.model_hparams
    return LSTMWrapper(
        num_variates=spec.num_variates,
        seq_len=spec.lookback_len,
        label_len=spec.label_len,
        pred_len=spec.pred_len,
        hidden_size=h["lstm_hidden_size"],
        num_layers=h["lstm_num_layers"],
        dropout=h["lstm_dropout"],
        bidirectional=h["lstm_bidirectional"],
    )


def get_run_config():
    return {
        "model_name": model_name,
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
        "model_hparams": {
            "lstm_hidden_size": lstm_hidden_size,
            "lstm_num_layers": lstm_num_layers,
            "lstm_dropout": lstm_dropout,
            "lstm_bidirectional": lstm_bidirectional,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run LSTM final protocol")
    parser.add_argument("--smoke", action="store_true", help="3 epochs smoke test")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_run_config()
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
        extra_config=cfg["model_hparams"],
        model_hparams=cfg["model_hparams"],
        use_time_features=cfg["use_time_features"],
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
