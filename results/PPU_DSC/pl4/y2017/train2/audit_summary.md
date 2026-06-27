# Audit Summary

**Run Dir**: `results/PPU_DSC/pl4/y2017/train2`

**Verdict**: **WARN**

## Metrics

- RMSE: 2.2607452869415283
- MAE: 1.57700777053833
- R2: 0.8979769945144653

## Issues

- WARN: large negative prediction beyond true data range (pred_min=-4.1019 < -1.0331, true_night_p01=-0.0217)

## Loss Info

- first_train_loss: 1471.170303
- last_train_loss: 0.005889
- first_test_loss: 471.881877
- last_test_loss: 0.011846
- best_test_loss: 0.011846

## R² Info

- final_test_r2: 0.8980
- best_test_r2: 0.8980

## Prediction Audit

- true_min: -0.022337811
- pred_min: -4.1018596
- true_negative_ratio: 0.4345
- pred_negative_ratio: 0.3522
- negative_ratio_gap: -0.0823
- true_night_min: -0.021697737
- true_night_p01: -0.021672
- large_negative_threshold: -1.033136
- extreme_negative_threshold: -4.132542
- night_pred_max: 0.9466486
- daytime_true_max: 20.662712

## Time Shift

- corr_shift0: 0.99
- best_shift: 0
- best_corr: 0.99
- corr_improvement: 0.0
