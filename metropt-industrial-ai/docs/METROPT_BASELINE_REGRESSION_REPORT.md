# MetroPT Baseline Regression Experiment

## Experimental design

- Forecast horizon: next consecutive one-minute observation.
- Split: chronological 70% training, 15% validation, 15% testing.
- Models: Linear Regression, Decision Tree, HistGradientBoosting.
- Baselines: persistence and training-period mean.
- Features: current sensor values, target lags and rolling averages.
- Recording gaps: lag and rolling windows do not cross gaps.
- Model selection: validation RMSE.

## Results

| Target | Model | Split | N | MAE | RMSE | R² |
|---|---|---|---:|---:|---:|---:|
| Oil_temperature | Persistence | Validation | 26589 | 1.558233 | 2.302204 | 0.693724 |
| Oil_temperature | Persistence | Test | 26590 | 1.629629 | 2.486777 | 0.738100 |
| Oil_temperature | Training Mean | Validation | 26589 | 3.489536 | 4.257200 | -0.047306 |
| Oil_temperature | Training Mean | Test | 26590 | 4.022409 | 4.892142 | -0.013583 |
| Oil_temperature | Linear Regression | Validation | 26589 | 0.705813 | 1.071758 | 0.933623 |
| Oil_temperature | Decision Tree | Validation | 26589 | 0.464359 | 0.787070 | 0.964203 |
| Oil_temperature | HistGradientBoosting | Validation | 26589 | 0.402418 | 0.719896 | 0.970052 |
| Oil_temperature | Linear Regression | Test | 26590 | 0.749293 | 1.247382 | 0.934104 |
| Oil_temperature | Decision Tree | Test | 26590 | 0.617877 | 1.100403 | 0.948718 |
| Oil_temperature | HistGradientBoosting | Test | 26590 | 0.548626 | 1.051173 | 0.953204 |
| TP3 | Persistence | Validation | 26589 | 0.278555 | 0.402409 | 0.560998 |
| TP3 | Persistence | Test | 26590 | 0.252093 | 0.381389 | 0.611385 |
| TP3 | Training Mean | Validation | 26589 | 0.505747 | 0.607488 | -0.000477 |
| TP3 | Training Mean | Test | 26590 | 0.503015 | 0.611803 | -0.000016 |
| TP3 | Linear Regression | Validation | 26589 | 0.130053 | 0.198622 | 0.893049 |
| TP3 | Decision Tree | Validation | 26589 | 0.049384 | 0.120307 | 0.960761 |
| TP3 | HistGradientBoosting | Validation | 26589 | 0.046725 | 0.119773 | 0.961109 |
| TP3 | Linear Regression | Test | 26590 | 0.117081 | 0.198108 | 0.895145 |
| TP3 | Decision Tree | Test | 26590 | 0.064817 | 0.155132 | 0.935704 |
| TP3 | HistGradientBoosting | Test | 26590 | 0.062048 | 0.146671 | 0.942526 |
| Motor_current | Persistence | Validation | 26589 | 0.811665 | 1.524609 | 0.401696 |
| Motor_current | Persistence | Test | 26590 | 0.753392 | 1.467363 | 0.494590 |
| Motor_current | Training Mean | Validation | 26589 | 1.917733 | 2.050040 | -0.081756 |
| Motor_current | Training Mean | Test | 26590 | 1.952024 | 2.072595 | -0.008318 |
| Motor_current | Linear Regression | Validation | 26589 | 0.751700 | 1.195192 | 0.632311 |
| Motor_current | Decision Tree | Validation | 26589 | 0.191594 | 0.492378 | 0.937598 |
| Motor_current | HistGradientBoosting | Validation | 26589 | 0.204889 | 0.483646 | 0.939791 |
| Motor_current | Linear Regression | Test | 26590 | 0.773200 | 1.169370 | 0.679024 |
| Motor_current | Decision Tree | Test | 26590 | 0.243916 | 0.638350 | 0.904350 |
| Motor_current | HistGradientBoosting | Test | 26590 | 0.246507 | 0.611180 | 0.912319 |

## Validation-based model selections

- **Oil_temperature**: HistGradientBoosting
- **TP3**: HistGradientBoosting
- **Motor_current**: HistGradientBoosting

## Limitations

- A single chronological test period does not establish generalization.
- The selected model must be assessed across additional time periods.
- Improvements over persistence must be practically meaningful.
- Prediction uncertainty and drift effects are not yet evaluated.
