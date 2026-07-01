# Loss vs Prediction Consistency Audit

**Run directory:** `/root/iTransformer-main/results/PPU_DSC/pl4/y2017/train2`

## Verdict

PASS: loss 与 ALL.csv 基本一致，loss 图可信，预测图问题来自模型输出本身。

## Core Metrics

| Item | Value |
|------|-------|
| final_test_loss | 0.01184572 |
| final_train_eval_loss | 0.00588857 |
| t_min | -0.02440568 |
| t_max | 20.87202756 |
| target_range | 20.89643325 |
| rmse_from_loss | 2.274327 |
| rmse_from_ALL_csv | 2.260745 |
| rmse_from_Overall_indicators | 2.260745 |
| difference_loss_vs_ALL | 0.013582 |
| difference_ALL_vs_Overall | -0.000000 |

## Step-1 Metrics

| Item | Value |
|------|-------|
| step1_rmse (ALL.csv) | 2.064592 |
| step1_mae (ALL.csv) | 1.461962 |
| step1_r2 (ALL.csv) | 0.914913 |
| step1_168h_rmse | 1.250705 |
| step1_168h_mae | 0.962715 |
| step1_168h_r2 | 0.973175 |

## 168h Day/Night Diagnostics

Night: hour_of_day < 6 or > 18. Day: 6 <= hour_of_day <= 18.

| Item | Value |
|------|-------|
| day_168h_rmse | 1.454135 |
| day_168h_mae | 1.098902 |
| night_168h_rmse | 0.956020 |
| night_168h_mae | 0.801768 |
| night_pred_min | -2.244588 |
| night_pred_max | 0.946649 |
| night_pred_abs_max | 2.244588 |
| night_pred_positive_ratio | 0.1429 |
| night_pred_negative_ratio | 0.8571 |

## evaluate() Note

evaluate() 当前对 test_loss / train_eval_loss 使用「按 batch 平均」（total_loss / len(loader)），不是「按样本点加权平均」。最后一个 batch 样本数不足时，会与 ALL.csv 的全体样本 RMSE 产生偏差。若需对齐，可将 evaluate() 改为 sse / count（仅影响记录，不影响训练反传）。

## Config

- dataset: `pv2017_ext`
- target_idx: 4
- train_ratio: 0.8
