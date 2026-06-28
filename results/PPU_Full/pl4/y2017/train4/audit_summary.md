# Audit Summary

**Run Dir**: `results/PPU_Full/pl4/y2017/train4`

**Verdict**: **WARN**

## Metrics

- RMSE: 1.993657350540161
- MAE: 1.3287523984909058
- R2: 0.9206593036651612

## Issues

- WARN: large negative prediction beyond true data range (pred_min=-2.5052 < -1.0331, true_night_p01=-0.0217)

## Loss Info

- first_train_loss: 1884.404195
- last_train_loss: 0.004630
- first_test_loss: 609.373606
- last_test_loss: 0.009227
- best_test_loss: 0.009227

## R² Info

- final_test_r2: 0.9207
- best_test_r2: 0.9207

## Prediction Audit

- true_min: -0.022337811
- pred_min: -2.505211
- true_negative_ratio: 0.4345
- pred_negative_ratio: 0.1938
- negative_ratio_gap: -0.2407
- true_night_min: -0.021697737
- true_night_p01: -0.021672
- large_negative_threshold: -1.033136
- extreme_negative_threshold: -4.132542
- night_pred_max: 1.022097
- daytime_true_max: 20.662712

## Time Shift

- corr_shift0: 0.9927
- best_shift: 0
- best_corr: 0.9927
- corr_improvement: 0.0
