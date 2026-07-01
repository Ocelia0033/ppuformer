# Loss vs Prediction Consistency Audit

**Run directory:** `/root/iTransformer-main/results/PPU_PSG/pl4/y2017/train1`

## Verdict

PASS: loss 与 ALL.csv 基本一致，loss 图可信，预测图问题来自模型输出本身。

## Core Metrics

| Item | Value |
|------|-------|
| final_test_loss | 0.02393089 |
| final_train_eval_loss | 0.00958387 |
| t_min | -0.02440568 |
| t_max | 20.87202756 |
| target_range | 20.89643325 |
| rmse_from_loss | 3.232597 |
| rmse_from_ALL_csv | 3.231669 |
| rmse_from_Overall_indicators | 3.231669 |
| difference_loss_vs_ALL | 0.000928 |
| difference_ALL_vs_Overall | -0.000000 |

## Step-1 Metrics

| Item | Value |
|------|-------|
| step1_rmse (ALL.csv) | 3.013943 |
| step1_mae (ALL.csv) | 2.029268 |
| step1_r2 (ALL.csv) | 0.818672 |
| step1_168h_rmse | 2.612746 |
| step1_168h_mae | 1.842688 |
| step1_168h_r2 | 0.882937 |

## 168h Day/Night Diagnostics

Night: hour_of_day < 6 or > 18. Day: 6 <= hour_of_day <= 18.

| Item | Value |
|------|-------|
| day_168h_rmse | 3.424409 |
| day_168h_mae | 2.721217 |
| night_168h_rmse | 1.017534 |
| night_168h_mae | 0.804426 |
| night_pred_min | -2.883604 |
| night_pred_max | 1.176803 |
| night_pred_abs_max | 2.883604 |
| night_pred_positive_ratio | 0.2338 |
| night_pred_negative_ratio | 0.7662 |

## evaluate() Note

evaluate() 当前对 test_loss / train_eval_loss 使用「按 batch 平均」（total_loss / len(loader)），不是「按样本点加权平均」。最后一个 batch 样本数不足时，会与 ALL.csv 的全体样本 RMSE 产生偏差。若需对齐，可将 evaluate() 改为 sse / count（仅影响记录，不影响训练反传）。

## Config

- dataset: `pv2017_ext`
- target_idx: 4
- train_ratio: 0.8
