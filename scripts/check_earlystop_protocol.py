from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_provider.split_utils import chronological_70_10_20_split


def load_module(filename: str, module_name: str):
    path = ROOT / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def check(condition: bool, message: str, failures: list[str]):
    if condition:
        print(f"[PASS] {message}")
    else:
        print(f"[FAIL] {message}")
        failures.append(message)


def main():
    failures: list[str] = []

    runner_source = (ROOT / "trainers" / "earlystop_runner.py").read_text(encoding="utf-8")
    save_paths_source = (ROOT / "utils" / "save_paths.py").read_text(encoding="utf-8")

    run_mod = load_module("run.py", "run_module")
    ppu_mod = load_module("run_ppuformer.py", "run_ppuformer_module")
    lstm_mod = load_module("run_lstm.py", "run_lstm_module")
    trans_mod = load_module("run_transformer.py", "run_transformer_module")
    patch_mod = load_module("run_patchtst.py", "run_patchtst_module")
    informer_mod = load_module("run_informer.py", "run_informer_module")
    auto_mod = load_module("run_autoformer.py", "run_autoformer_module")
    fed_mod = load_module("run_fedformer.py", "run_fedformer_module")

    entry_modules = {
        "run.py": run_mod,
        "run_lstm.py": lstm_mod,
        "run_transformer.py": trans_mod,
        "run_patchtst.py": patch_mod,
        "run_informer.py": informer_mod,
        "run_autoformer.py": auto_mod,
        "run_fedformer.py": fed_mod,
        "run_ppuformer.py": ppu_mod,
    }

    baseline_specs = {
        "run_lstm.py": lstm_mod.get_run_config(),
        "run_transformer.py": trans_mod.get_run_config(),
        "run_patchtst.py": patch_mod.get_run_config(),
        "run_informer.py": informer_mod.get_run_config(),
        "run_autoformer.py": auto_mod.get_run_config(),
        "run_fedformer.py": fed_mod.get_run_config(),
    }
    itransformer_5 = run_mod.get_run_config("iTransformer-5")
    itransformer_17 = run_mod.get_run_config("iTransformer-17")
    ppu_full = ppu_mod.get_run_config("PPU_Full")

    for filename in entry_modules:
        source = (ROOT / filename).read_text(encoding="utf-8")
        check(
            "run_with_early_stopping" in source,
            f"{filename} 调用统一 earlystop runner",
            failures,
        )

    for filename, spec in {
        **baseline_specs,
        "run.py(iTransformer-5)": itransformer_5,
        "run.py(iTransformer-17)": itransformer_17,
        "run_ppuformer.py(PPU_Full)": ppu_full,
    }.items():
        check(spec["pred_len"] == 4, f"{filename} 使用 pred_len=4", failures)
        check(spec["lookback_len"] == 168, f"{filename} 使用 lookback_len=168", failures)
        check(spec["target_idx"] == 4, f"{filename} 使用 target_idx=4", failures)

    check(
        "chronological_70_10_20_split" in runner_source,
        "统一 runner 使用 chronological 70/10/20 划分",
        failures,
    )

    toy_features = np.arange(2000 * 5, dtype=np.float32).reshape(2000, 5)
    toy_timestamps = np.array([f"2017-01-01 {i % 24:02d}:00:00" for i in range(2000)])
    split = chronological_70_10_20_split(
        features=toy_features,
        timestamps=toy_timestamps,
        lookback_len=168,
        pred_len=4,
        target_idx=4,
        verbose=False,
    )

    check(
        split["split_info"]["scaler_fit_range"] == [0, int(2000 * 0.7) - 1],
        "scaler 只 fit train 区间",
        failures,
    )
    check(
        split["split_info"]["val_context_range"][0] == split["split_info"]["train_end"] - 168,
        "val 只向前借 lookback 上下文",
        failures,
    )
    check(
        split["split_info"]["test_context_range"][0] == split["split_info"]["val_end"] - 168,
        "test 只向前借 lookback 上下文",
        failures,
    )
    train_label = split["split_info"]["train_label_range"]
    val_label = split["split_info"]["val_label_range"]
    test_label = split["split_info"]["test_label_range"]
    check(train_label[1] < val_label[0], "train/val label 不重叠", failures)
    check(val_label[1] < test_label[0], "val/test label 不重叠", failures)

    check(
        '"monitor": "val_loss"' in runner_source or "'monitor': 'val_loss'" in runner_source,
        "early stopping 监控 val_loss",
        failures,
    )
    check("test_loss" not in runner_source, "训练过程中没有每轮 test_loss 参与 early stopping", failures)
    check(
        runner_source.count('loaders["test_loader"]') == 1,
        "test 只在训练结束后最终评估一次",
        failures,
    )

    for filename, spec in baseline_specs.items():
        check(spec["dataset_name"] == "pv2017", f"{filename} 保持 pv2017 数据集", failures)
        check(spec["num_variates"] == 5, f"{filename} 保持 5 维输入", failures)
        check(spec["input_type"] == "original_5_features", f"{filename} 记录 original_5_features", failures)

    check(
        set(run_mod.ITRANSFORMER_VARIANTS.keys()) == {"iTransformer-5", "iTransformer-17"},
        "run.py 支持 iTransformer-5 和 iTransformer-17",
        failures,
    )
    check(ppu_full["dataset_name"] == "pv2017_ext", "run_ppuformer.py 使用 pv2017_ext", failures)
    check(ppu_full["num_variates"] == 17, "run_ppuformer.py 使用 17 维输入", failures)
    check(
        set(ppu_mod.PPU_VARIANTS.keys())
        == {"PPU_Full", "PPU_NoPSG", "PPU_NoWASE", "PPU_NoDSC", "PPU_NoPGIA"},
        "run_ppuformer.py 支持 Full / NoPSG / NoWASE / NoDSC / NoPGIA",
        failures,
    )

    for filename in (
        "run_lstm.py",
        "run_transformer.py",
        "run_patchtst.py",
        "run_informer.py",
        "run_autoformer.py",
        "run_fedformer.py",
        "run_ppuformer.py",
        "run.py",
    ):
        source = (ROOT / filename).read_text(encoding="utf-8")
        check("配置区" in source, f"{filename} 保留清晰配置区", failures)
        check("def build_model(" in source, f"{filename} 保留 build_model()", failures)

    check("Train Eval Loss vs Val Loss" in runner_source, "loss.png 画 Train Eval Loss vs Val Loss", failures)
    check("split_protocol" in runner_source, "args.json 记录 split_protocol", failures)
    check("input_type" in runner_source, "args.json 记录 input_type", failures)
    check("early_stopping" in runner_source, "args.json 记录 early_stopping", failures)
    check("test_usage" in runner_source, "args.json 记录 test_usage", failures)
    check("overall_indicators.csv" in save_paths_source, "输出文件包含 overall_indicators.csv", failures)

    if failures:
        print("\nProtocol check failed:")
        for item in failures:
            print(f" - {item}")
        sys.exit(1)

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
