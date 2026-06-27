# Audit Summary

**Run Dir**: `results/PPU_Full/pl4/y2017/train2`

**Verdict**: **FAIL**

## Metrics

- RMSE: 2.2090818881988525
- MAE: 1.488168716430664
- R2: 0.9025866389274596

## Issues

- FAIL: extreme negative prediction (pred_min=-8.6855 < -4.1325)

## Loss Info

- first_train_loss: 1879.226865
- last_train_loss: 0.005131
- first_test_loss: 597.477921
- last_test_loss: 0.011301
- best_test_loss: 0.011301

## R² Info

- final_test_r2: 0.9026
- best_test_r2: 0.9026

## Prediction Audit

- true_min: -0.022337811
- pred_min: -8.685452
- true_negative_ratio: 0.4345
- pred_negative_ratio: 0.1641
- negative_ratio_gap: -0.2704
- true_night_min: -0.021697737
- true_night_p01: -0.021672
- large_negative_threshold: -1.033136
- extreme_negative_threshold: -4.132542
- night_pred_max: 1.8363832
- daytime_true_max: 20.662712

## Time Shift

- corr_shift0: 0.9915
- best_shift: 0
- best_corr: 0.9915
- corr_improvement: 0.0
