# Audit Summary

**Run Dir**: `results/PPU_Full/pl4/y2017/train3`

**Verdict**: **WARN**

## Metrics

- RMSE: 1.9429978132247925
- MAE: 1.26383376121521
- R2: 0.9246402382850648

## Issues

- WARN: large negative prediction beyond true data range (pred_min=-2.0355 < -1.0331, true_night_p01=-0.0217)

## Loss Info

- first_train_loss: 1884.494560
- last_train_loss: 0.004628
- first_test_loss: 609.513343
- last_test_loss: 0.008772
- best_test_loss: 0.008772

## R² Info

- final_test_r2: 0.9246
- best_test_r2: 0.9246

## Prediction Audit

- true_min: -0.022337811
- pred_min: -2.035507
- true_negative_ratio: 0.4345
- pred_negative_ratio: 0.1824
- negative_ratio_gap: -0.2521
- true_night_min: -0.021697737
- true_night_p01: -0.021672
- large_negative_threshold: -1.033136
- extreme_negative_threshold: -4.132542
- night_pred_max: 1.0437448
- daytime_true_max: 20.662712

## Time Shift

- corr_shift0: 0.9926
- best_shift: 0
- best_corr: 0.9926
- corr_improvement: 0.0
