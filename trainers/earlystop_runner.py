from __future__ import annotations

import copy
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset

from data_provider.split_utils import chronological_70_10_20_split
from utils import append_run_summary, create_save_paths, save_args_json
from utils.reporters import (
    save_all,
    save_best,
    save_curve,
    save_curve_after_epoch,
    save_overall_indicators,
    save_prediction_curve,
)

TRAIN_RATIO = 0.7
VAL_END_RATIO = 0.8
DEFAULT_MAX_EPOCHS = 500
DEFAULT_PATIENCE = 30
DEFAULT_MIN_DELTA = 1e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ExperimentSpec:
    model_name: str
    des: str
    dataset_name: str
    year: Optional[int]
    num_variates: int
    input_type: str
    build_model: Callable[["ExperimentSpec"], nn.Module]
    model_hparams: dict[str, Any]
    batch_size: int
    learning_rate: float
    lookback_len: int = 168
    pred_len: int = 4
    target_idx: int = 4
    label_len: int = 48
    train_ratio: float = TRAIN_RATIO
    val_end_ratio: float = VAL_END_RATIO
    use_time_features: bool = False
    seed: int = 35040
    results_dir: str = "results"
    loss_plot_ylim: Optional[tuple[float, float]] = None
    weight_decay: float = 0.0
    optimizer_name: str = "Adam"
    scheduler_name: Optional[str] = None
    loss_name: str = "MSE"
    max_epochs: int = DEFAULT_MAX_EPOCHS
    patience: int = DEFAULT_PATIENCE
    min_delta: float = DEFAULT_MIN_DELTA
    extra_args: dict[str, Any] = field(default_factory=dict)
    optimizer_factory: Optional[
        Callable[[nn.Module, "ExperimentSpec"], torch.optim.Optimizer]
    ] = None


class WindowDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        timestamps: np.ndarray,
        lookback_len: int,
        pred_len: int,
        target_idx: int,
        time_feats: Optional[np.ndarray] = None,
        label_len: int = 48,
    ):
        self.data = data
        self.timestamps = timestamps
        self.lookback_len = lookback_len
        self.pred_len = pred_len
        self.target_idx = target_idx
        self.time_feats = time_feats
        self.label_len = label_len
        self.length = len(data) - lookback_len - pred_len + 1
        assert self.length > 0, (
            f"样本不足: len={len(data)}, lookback={lookback_len}, pred_len={pred_len}"
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        s_begin = idx
        s_end = idx + self.lookback_len
        x = self.data[s_begin:s_end]
        y = self.data[s_end:s_end + self.pred_len, self.target_idx]

        if self.time_feats is None:
            return torch.FloatTensor(x), torch.FloatTensor(y)

        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len
        x_mark = self.time_feats[s_begin:s_end]
        y_mark = self.time_feats[r_begin:r_end]
        return (
            torch.FloatTensor(x),
            torch.FloatTensor(y),
            torch.FloatTensor(x_mark),
            torch.FloatTensor(y_mark),
        )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def infer_year_from_dataset(name: str) -> int:
    match = re.search(r"(20\d{2})", name)
    if match:
        return int(match.group(1))
    raise ValueError(f"无法从 dataset_name='{name}' 推导年份")


def make_time_features(timestamps_str) -> np.ndarray:
    dt = pd.to_datetime(pd.Series(timestamps_str))
    hour_of_day = dt.dt.hour / 23.0 - 0.5
    day_of_week = dt.dt.dayofweek / 6.0 - 0.5
    day_of_month = (dt.dt.day - 1) / 30.0 - 0.5
    day_of_year = (dt.dt.dayofyear - 1) / 365.0 - 0.5
    return np.stack(
        [
            hour_of_day.values,
            day_of_week.values,
            day_of_month.values,
            day_of_year.values,
        ],
        axis=1,
    ).astype(np.float32)


def load_dataset(dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
    csv_path = os.path.join("dataset", f"{dataset_name}.csv")
    df = pd.read_csv(csv_path)
    timestamps = df.iloc[:, 0].values
    features = df.iloc[:, 1:].values.astype(np.float32)
    return features, timestamps


def build_label_hours(
    timestamps: np.ndarray,
    lookback_len: int,
    pred_len: int,
) -> np.ndarray:
    ts_dt = pd.to_datetime(timestamps)
    sample_count = len(timestamps) - lookback_len - pred_len + 1
    rows = []
    for i in range(sample_count):
        rows.append(
            ts_dt[lookback_len + i : lookback_len + i + pred_len].hour.to_numpy()
        )
    return np.asarray(rows, dtype=np.int64)


def create_loaders_and_bundle(spec: ExperimentSpec) -> dict[str, Any]:
    features, timestamps = load_dataset(spec.dataset_name)
    time_feats = make_time_features(timestamps) if spec.use_time_features else None

    bundle = chronological_70_10_20_split(
        features=features,
        timestamps=timestamps,
        lookback_len=spec.lookback_len,
        pred_len=spec.pred_len,
        train_ratio=spec.train_ratio,
        val_end_ratio=spec.val_end_ratio,
        target_idx=spec.target_idx,
        time_feats=time_feats,
        verbose=True,
    )

    train_dataset = WindowDataset(
        data=bundle["train_data"],
        timestamps=bundle["train_timestamps"],
        lookback_len=spec.lookback_len,
        pred_len=spec.pred_len,
        target_idx=spec.target_idx,
        time_feats=bundle.get("train_time_feats"),
        label_len=spec.label_len,
    )
    val_dataset = WindowDataset(
        data=bundle["val_data"],
        timestamps=bundle["val_timestamps"],
        lookback_len=spec.lookback_len,
        pred_len=spec.pred_len,
        target_idx=spec.target_idx,
        time_feats=bundle.get("val_time_feats"),
        label_len=spec.label_len,
    )
    test_dataset = WindowDataset(
        data=bundle["test_data"],
        timestamps=bundle["test_timestamps"],
        lookback_len=spec.lookback_len,
        pred_len=spec.pred_len,
        target_idx=spec.target_idx,
        time_feats=bundle.get("test_time_feats"),
        label_len=spec.label_len,
    )

    bundle["features_shape"] = features.shape
    bundle["loaders"] = {
        "train_loader": DataLoader(
            train_dataset,
            batch_size=spec.batch_size,
            shuffle=True,
            drop_last=True,
        ),
        "train_eval_loader": DataLoader(
            train_dataset,
            batch_size=spec.batch_size,
            shuffle=False,
            drop_last=False,
        ),
        "val_loader": DataLoader(
            val_dataset,
            batch_size=spec.batch_size,
            shuffle=False,
            drop_last=False,
        ),
        "test_loader": DataLoader(
            test_dataset,
            batch_size=spec.batch_size,
            shuffle=False,
            drop_last=False,
        ),
    }
    bundle["train_label_hours"] = build_label_hours(
        bundle["train_timestamps"], spec.lookback_len, spec.pred_len
    )
    bundle["val_label_hours"] = build_label_hours(
        bundle["val_timestamps"], spec.lookback_len, spec.pred_len
    )
    bundle["test_label_hours"] = build_label_hours(
        bundle["test_timestamps"], spec.lookback_len, spec.pred_len
    )
    return bundle


def build_optimizer(
    model: nn.Module,
    spec: ExperimentSpec,
) -> torch.optim.Optimizer:
    if spec.optimizer_factory is not None:
        return spec.optimizer_factory(model, spec)
    return torch.optim.Adam(
        model.parameters(),
        lr=spec.learning_rate,
        weight_decay=spec.weight_decay,
    )


def _move_batch_to_device(batch):
    return tuple(item.to(device) for item in batch)


def _forward_model(model: nn.Module, batch):
    if len(batch) == 2:
        x, _ = batch
        return model(x)
    if len(batch) == 4:
        x, _, x_mark, y_mark = batch
        return model(x, x_mark, y_mark)
    raise ValueError(f"不支持的 batch 格式: len={len(batch)}")


def _target_from_batch(batch):
    return batch[1]


def _target_scale(target_min: float, target_max: float) -> float:
    scale = float(target_max - target_min)
    return scale if scale != 0.0 else 1.0


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    target_idx: int,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = _move_batch_to_device(batch)
        y = _target_from_batch(batch)
        optimizer.zero_grad()
        pred = _forward_model(model, batch)
        pred_target = pred[:, :, target_idx]
        loss = criterion(pred_target, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    target_idx: int,
    target_min: float,
    target_max: float,
    label_hours: np.ndarray,
) -> dict[str, Any]:
    model.eval()
    sse = 0.0
    count = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch)
            y = _target_from_batch(batch)
            pred = _forward_model(model, batch)
            pred_target = pred[:, :, target_idx]
            sse += torch.sum((pred_target - y) ** 2).item()
            count += y.numel()
            all_preds.append(pred_target.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    avg_loss = sse / count
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    scale = _target_scale(target_min, target_max)
    preds_orig = all_preds * scale + target_min
    targets_orig = all_targets * scale + target_min

    preds_flat = preds_orig.flatten()
    targets_flat = targets_orig.flatten()
    rmse = float(np.sqrt(mean_squared_error(targets_flat, preds_flat)))
    mae = float(mean_absolute_error(targets_flat, preds_flat))
    r2 = float(r2_score(targets_flat, preds_flat))
    step1_r2 = float(r2_score(targets_orig[:, 0], preds_orig[:, 0]))

    deep_mask = (label_hours >= 0) & (label_hours <= 5)
    deep_pred = preds_orig[deep_mask]
    deep_true = targets_orig[deep_mask]
    if deep_true.size > 0:
        deep_night_rmse = float(np.sqrt(mean_squared_error(deep_true, deep_pred)))
        deep_night_pred_abs_max = float(np.max(np.abs(deep_pred)))
    else:
        deep_night_rmse = 0.0
        deep_night_pred_abs_max = 0.0

    return {
        "avg_loss": float(avg_loss),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "step1_r2": step1_r2,
        "deep_night_rmse": deep_night_rmse,
        "deep_night_pred_abs_max": deep_night_pred_abs_max,
        "preds_orig": preds_orig,
        "targets_orig": targets_orig,
    }


def save_training_artifacts(
    paths: dict[str, str],
    history: dict[str, list[Any]],
    best_epoch: int,
    loss_plot_ylim: Optional[tuple[float, float]],
):
    epochs = history["epoch"]
    df = pd.DataFrame(
        {
            "epoch": epochs,
            "train_step_loss": history["train_step_loss"],
            "train_eval_loss": history["train_eval_loss"],
            "val_loss": history["val_loss"],
            "train_MAE": history["train_MAE"],
            "val_MAE": history["val_MAE"],
            "train_MSE": history["train_MSE"],
            "val_MSE": history["val_MSE"],
            "train_RMSE": history["train_RMSE"],
            "val_RMSE": history["val_RMSE"],
            "train_R2": history["train_R2"],
            "val_R2": history["val_R2"],
            "learning_rate": history["learning_rate"],
            "is_best": [1 if epoch == best_epoch else 0 for epoch in epochs],
        }
    )
    df.to_csv(paths["loss_csv"], index=False)

    save_curve(
        epochs,
        history["train_eval_loss"],
        history["val_loss"],
        None,
        paths["loss_png"],
        ylabel="Loss",
        title="Train Eval Loss vs Val Loss",
        ylim=loss_plot_ylim,
        series_label="Val",
    )
    save_curve_after_epoch(
        epochs,
        history["train_eval_loss"],
        history["val_loss"],
        paths["loss_zoom_png"],
        ylabel="Loss",
        title="Train Eval Loss vs Val Loss from epoch 10",
        skip_epochs=10,
        series_label="Val",
    )
    save_curve(
        epochs,
        history["train_MAE"],
        history["val_MAE"],
        paths["mae_csv"],
        paths["mae_png"],
        ylabel="MAE",
        title="Train MAE vs Val MAE",
        series_label="Val",
    )
    save_curve(
        epochs,
        history["train_MSE"],
        history["val_MSE"],
        paths["mse_csv"],
        paths["mse_png"],
        ylabel="MSE",
        title="Train MSE vs Val MSE",
        series_label="Val",
    )
    save_curve(
        epochs,
        history["train_R2"],
        history["val_R2"],
        paths["r2_csv"],
        paths["r2_png"],
        ylabel="R²",
        title="Train R² vs Val R²",
        series_label="Val",
    )


def save_test_artifacts(
    paths: dict[str, str],
    test_eval: dict[str, Any],
    test_timestamps: np.ndarray,
    lookback_len: int,
    pred_len: int,
    final_metrics: dict[str, Any],
):
    preds = test_eval["preds_orig"]
    targets = test_eval["targets_orig"]

    save_overall_indicators(
        paths["overall_csv"],
        test_eval["rmse"],
        test_eval["mae"],
        test_eval["r2"],
        extras={
            "step1_R2": final_metrics["step1_R2"],
            "deep_night_RMSE": final_metrics["deep_night_RMSE"],
            "deep_night_pred_abs_max": final_metrics["deep_night_pred_abs_max"],
            "best_epoch": final_metrics["best_epoch"],
            "best_val_loss": final_metrics["best_val_loss"],
            "best_val_RMSE": final_metrics["best_val_RMSE"],
            "stopped_epoch": final_metrics["stopped_epoch"],
            "train_time_sec": final_metrics["train_time_sec"],
        },
    )

    save_best(
        paths["best_csv"],
        paths["best_png"],
        preds,
        targets,
        pred_len,
        test_timestamps,
        lookback_len,
    )
    save_all(
        paths["all_csv"],
        paths["all_scatter_png"],
        paths["all_error_png"],
        preds,
        targets,
        pred_len,
        test_timestamps,
        lookback_len,
        predictions_csv=paths.get("predictions_csv"),
    )
    save_prediction_curve(
        paths["prediction_curve_png"],
        preds,
        targets,
        zoom_168h_png=paths.get("prediction_curve_168h_png"),
        zoom_168h_csv=paths.get("prediction_curve_168h_csv"),
        timestamps=test_timestamps,
        lookback_len=lookback_len,
        phase="test",
    )


def _base_args_payload(
    spec: ExperimentSpec,
    split_info: dict[str, Any],
    model_param_count: int,
    smoke: bool,
) -> dict[str, Any]:
    payload = {
        "model_name": spec.model_name,
        "des": spec.des,
        "dataset_name": spec.dataset_name,
        "year": spec.year,
        "num_variates": spec.num_variates,
        "input_type": spec.input_type,
        "lookback_len": spec.lookback_len,
        "pred_len": spec.pred_len,
        "label_len": spec.label_len,
        "target_idx": spec.target_idx,
        "split_protocol": "chronological_70_10_20",
        "train_ratio": spec.train_ratio,
        "val_end_ratio": spec.val_end_ratio,
        "test_usage": "final_evaluation_only",
        "early_stopping": {
            "enabled": True,
            "max_epochs": 3 if smoke else spec.max_epochs,
            "patience": 2 if smoke else spec.patience,
            "min_delta": spec.min_delta,
            "monitor": "val_loss",
            "mode": "min",
            "restore_best_weights": True,
        },
        "optimizer": spec.optimizer_name,
        "weight_decay": spec.weight_decay,
        "scheduler": spec.scheduler_name,
        "loss": spec.loss_name,
        "batch_size": spec.batch_size,
        "learning_rate": spec.learning_rate,
        "seed": spec.seed,
        "results_dir": spec.results_dir,
        "loss_plot_ylim": spec.loss_plot_ylim,
        "model_hparams": dict(spec.model_hparams),
        "split_info": split_info,
        "total_params": model_param_count,
        "smoke_test": smoke,
    }
    payload.update(spec.extra_args)
    return payload


def run_with_earlystop_protocol(spec: ExperimentSpec, smoke: bool = False) -> dict[str, Any]:
    set_seed(spec.seed)
    year = spec.year if spec.year is not None else infer_year_from_dataset(spec.dataset_name)
    spec.year = year

    print("=" * 70)
    print(f"  Experiment: {spec.model_name}")
    print("  Protocol:   chronological_70_10_20 + early_stopping")
    print(f"  Dataset:    {spec.dataset_name} ({spec.num_variates} vars, {spec.input_type})")
    print(f"  lookback={spec.lookback_len}, pred_len={spec.pred_len}, target_idx={spec.target_idx}")
    print("=" * 70)

    bundle = create_loaders_and_bundle(spec)
    loaders = bundle["loaders"]
    print(
        f"Data loaded: {bundle['features_shape'][0]} samples, "
        f"{bundle['features_shape'][1]} features"
    )
    print(f"Train batches: {len(loaders['train_loader'])}")
    print(f"Val batches:   {len(loaders['val_loader'])}")
    print(f"Test batches:  {len(loaders['test_loader'])}")

    model = spec.build_model(spec).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = build_optimizer(model, spec)
    criterion = nn.MSELoss()

    paths = create_save_paths(
        model_name=spec.model_name,
        year=year,
        pred_len=spec.pred_len,
        base_dir=spec.results_dir,
    )
    print(f"Save dir: {paths['save_dir']}  (train{paths['train_id']})")

    max_epochs = 3 if smoke else spec.max_epochs
    patience = 2 if smoke else spec.patience

    history = {
        "epoch": [],
        "train_step_loss": [],
        "train_eval_loss": [],
        "val_loss": [],
        "train_MAE": [],
        "val_MAE": [],
        "train_MSE": [],
        "val_MSE": [],
        "train_RMSE": [],
        "val_RMSE": [],
        "train_R2": [],
        "val_R2": [],
        "learning_rate": [],
    }

    best_val_loss = float("inf")
    best_val_rmse = float("inf")
    best_epoch = 0
    best_state_dict = None
    patience_counter = 0
    stopped_epoch = max_epochs

    start_time = time.time()
    for epoch in range(1, max_epochs + 1):
        train_step_loss = train_one_epoch(
            model=model,
            loader=loaders["train_loader"],
            optimizer=optimizer,
            criterion=criterion,
            target_idx=spec.target_idx,
        )
        train_eval = evaluate_loader(
            model=model,
            loader=loaders["train_eval_loader"],
            criterion=criterion,
            target_idx=spec.target_idx,
            target_min=bundle["target_min"],
            target_max=bundle["target_max"],
            label_hours=bundle["train_label_hours"],
        )
        val_eval = evaluate_loader(
            model=model,
            loader=loaders["val_loader"],
            criterion=criterion,
            target_idx=spec.target_idx,
            target_min=bundle["target_min"],
            target_max=bundle["target_max"],
            label_hours=bundle["val_label_hours"],
        )

        val_loss = val_eval["avg_loss"]
        lr_current = float(optimizer.param_groups[0]["lr"])
        is_best = False

        if val_loss < best_val_loss - spec.min_delta:
            best_val_loss = float(val_loss)
            best_val_rmse = float(val_eval["rmse"])
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_counter = 0
            is_best = True
        else:
            patience_counter += 1

        history["epoch"].append(epoch)
        history["train_step_loss"].append(float(train_step_loss))
        history["train_eval_loss"].append(float(train_eval["avg_loss"]))
        history["val_loss"].append(float(val_loss))
        history["train_MAE"].append(float(train_eval["mae"]))
        history["val_MAE"].append(float(val_eval["mae"]))
        history["train_MSE"].append(float(train_eval["avg_loss"]))
        history["val_MSE"].append(float(val_eval["avg_loss"]))
        history["train_RMSE"].append(float(train_eval["rmse"]))
        history["val_RMSE"].append(float(val_eval["rmse"]))
        history["train_R2"].append(float(train_eval["r2"]))
        history["val_R2"].append(float(val_eval["r2"]))
        history["learning_rate"].append(lr_current)

        if epoch == 1 or epoch % 10 == 0 or is_best or patience_counter >= patience:
            mark = " *BEST*" if is_best else ""
            print(
                f"Epoch [{epoch:3d}/{max_epochs}]  "
                f"TrainStep: {train_step_loss:.6f}  "
                f"TrainEval: {train_eval['avg_loss']:.6f}  "
                f"Val: {val_loss:.6f}  "
                f"ValRMSE: {val_eval['rmse']:.4f}  "
                f"patience: {patience_counter}/{patience}{mark}"
            )

        if patience_counter >= patience:
            stopped_epoch = epoch
            print(f"\nEarly stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break
    else:
        print(f"\nReached max_epochs={max_epochs}. Best epoch: {best_epoch}")

    train_time_sec = round(time.time() - start_time, 2)

    assert best_state_dict is not None, "训练未产生任何 best checkpoint"
    model.load_state_dict(best_state_dict)
    torch.save(best_state_dict, os.path.join(paths["save_dir"], "model.pth"))

    save_training_artifacts(paths, history, best_epoch, spec.loss_plot_ylim)

    print("\n" + "=" * 70)
    print("  Final Test Evaluation (best checkpoint)")
    print("=" * 70)
    test_eval = evaluate_loader(
        model=model,
        loader=loaders["test_loader"],
        criterion=criterion,
        target_idx=spec.target_idx,
        target_min=bundle["target_min"],
        target_max=bundle["target_max"],
        label_hours=bundle["test_label_hours"],
    )

    final_metrics = {
        "RMSE": float(test_eval["rmse"]),
        "MAE": float(test_eval["mae"]),
        "R2": float(test_eval["r2"]),
        "step1_R2": float(test_eval["step1_r2"]),
        "deep_night_RMSE": float(test_eval["deep_night_rmse"]),
        "deep_night_pred_abs_max": float(test_eval["deep_night_pred_abs_max"]),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_val_RMSE": float(best_val_rmse),
        "stopped_epoch": int(stopped_epoch),
        "train_time_sec": train_time_sec,
        "total_params": int(total_params),
    }

    save_test_artifacts(
        paths=paths,
        test_eval=test_eval,
        test_timestamps=bundle["test_timestamps"],
        lookback_len=spec.lookback_len,
        pred_len=spec.pred_len,
        final_metrics=final_metrics,
    )

    args_payload = _base_args_payload(
        spec=spec,
        split_info=bundle["split_info"],
        model_param_count=total_params,
        smoke=smoke,
    )
    save_args_json(paths["args_json"], args_payload, final_metrics)
    summary_path = append_run_summary(config=args_payload, metrics=final_metrics, paths=paths)

    print(f"  RMSE:         {final_metrics['RMSE']:.6f}")
    print(f"  MAE:          {final_metrics['MAE']:.6f}")
    print(f"  R2:           {final_metrics['R2']:.6f}")
    print(f"  step1_R2:     {final_metrics['step1_R2']:.6f}")
    print(f"  best_epoch:   {final_metrics['best_epoch']}")
    print(f"  best_val_loss:{final_metrics['best_val_loss']:.8f}")
    print(f"  stopped_epoch:{final_metrics['stopped_epoch']}")
    print(f"  train_time:   {final_metrics['train_time_sec']:.1f}s")
    print(f"\nResults saved to: {paths['save_dir']}")
    print(f"Summary appended to: {summary_path}")

    return {
        "metrics": final_metrics,
        "paths": paths,
        "args": args_payload,
    }


def run_with_early_stopping(
    *,
    model_builder: Callable[[ExperimentSpec], nn.Module],
    model_name: str,
    des: str,
    dataset_name: str,
    year: Optional[int],
    num_variates: int,
    input_type: str,
    target_idx: int,
    lookback_len: int,
    pred_len: int,
    label_len: int,
    batch_size: int,
    learning_rate: float,
    max_epochs: int,
    patience: int,
    min_delta: float,
    train_ratio: float,
    val_end_ratio: float,
    seed: int,
    results_dir: str,
    loss_plot_ylim: Optional[tuple[float, float]],
    extra_config: Optional[dict[str, Any]] = None,
    model_hparams: Optional[dict[str, Any]] = None,
    use_time_features: bool = False,
    weight_decay: float = 0.0,
    optimizer_name: str = "Adam",
    scheduler_name: Optional[str] = None,
    loss_name: str = "MSE",
    optimizer_builder: Optional[
        Callable[[nn.Module, ExperimentSpec], torch.optim.Optimizer]
    ] = None,
    smoke: bool = False,
) -> dict[str, Any]:
    spec = ExperimentSpec(
        model_name=model_name,
        des=des,
        dataset_name=dataset_name,
        year=year,
        num_variates=num_variates,
        input_type=input_type,
        build_model=model_builder,
        model_hparams=dict(model_hparams or {}),
        batch_size=batch_size,
        learning_rate=learning_rate,
        lookback_len=lookback_len,
        pred_len=pred_len,
        target_idx=target_idx,
        label_len=label_len,
        train_ratio=train_ratio,
        val_end_ratio=val_end_ratio,
        use_time_features=use_time_features,
        seed=seed,
        results_dir=results_dir,
        loss_plot_ylim=loss_plot_ylim,
        weight_decay=weight_decay,
        optimizer_name=optimizer_name,
        scheduler_name=scheduler_name,
        loss_name=loss_name,
        max_epochs=max_epochs,
        patience=patience,
        min_delta=min_delta,
        extra_args=dict(extra_config or {}),
        optimizer_factory=optimizer_builder,
    )
    return run_with_earlystop_protocol(spec, smoke=smoke)
