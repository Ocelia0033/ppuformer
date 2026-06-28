# Audit Summary

**Run Dir**: `results/PPU_Full/pl4/y2017/train5`

**Verdict**: **FAIL**

## Metrics

- RMSE: 2.3099474906921387
- MAE: 1.6157758235931396
- R2: 0.8934878706932068

## Issues

- FAIL: extreme negative prediction (pred_min=-6.9520 < -4.1325)

## Loss Info

- first_train_loss: 1884.630402
- last_train_loss: 0.005095
- first_test_loss: 609.728695
- last_test_loss: 0.012333
- best_test_loss: 0.012333

## R² Info

- final_test_r2: 0.8935
- best_test_r2: 0.8935

## Prediction Audit

- true_min: -0.022337811
- pred_min: -6.952005
- true_negative_ratio: 0.4345
- pred_negative_ratio: 0.2338
- negative_ratio_gap: -0.2007
- true_night_min: -0.021697737
- true_night_p01: -0.021672
- large_negative_threshold: -1.033136
- extreme_negative_threshold: -4.132542
- night_pred_max: 2.3735187
- daytime_true_max: 20.662712

## Time Shift

- corr_shift0: 0.9896
- best_shift: 0
- best_corr: 0.9896
- corr_improvement: 0.0
