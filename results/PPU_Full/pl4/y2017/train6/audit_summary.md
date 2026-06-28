# Audit Summary

**Run Dir**: `results/PPU_Full/pl4/y2017/train6`

**Verdict**: **WARN**

## Metrics

- RMSE: 2.1033449172973637
- MAE: 1.3136155605316162
- R2: 0.9116888046264648

## Issues

- WARN: test_loss rises in later epochs — possible overfitting
- WARN: large negative prediction beyond true data range (pred_min=-1.7696 < -1.0331, true_night_p01=-0.0217)

## Loss Info

- first_train_loss: 0.005592
- last_train_loss: 0.002915
- first_test_loss: 0.010092
- last_test_loss: 0.010250
- best_test_loss: 0.007534

## R² Info

- final_test_r2: 0.9117
- best_test_r2: 0.9352

## Prediction Audit

- true_min: -0.022337811
- pred_min: -1.7696307
- true_negative_ratio: 0.4345
- pred_negative_ratio: 0.1738
- negative_ratio_gap: -0.2607
- true_night_min: -0.021697737
- true_night_p01: -0.021672
- large_negative_threshold: -1.033136
- extreme_negative_threshold: -4.132542
- night_pred_max: 0.7347488
- daytime_true_max: 20.662712

## Time Shift

- corr_shift0: 0.9951
- best_shift: 0
- best_corr: 0.9951
- corr_improvement: 0.0
