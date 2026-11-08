# MetroPT Residual and Prediction Error Analysis

## Experimental design

- HistGradientBoosting with the Phase 2 hyperparameters.
- Three chronological test windows.
- Three seeds: 17, 42 and 73.
- Targets: Oil_temperature, TP3 and Motor_current.
- Residual = actual minus predicted.
- Rolling MAE uses a 60-observation window.
- Rolling calculations reset at timestamp gaps.
- Seed 42 is used for representative visualizations.

## Aggregate residual metrics

| Target | Mean MAE | Mean RMSE | RMSE Std | Mean R² | Mean Bias | Mean P95 Absolute Error | Mean P99 Absolute Error |
|---|---:|---:|---:|---:|---:|---:|---:|
| Motor_current | 0.246787 | 0.577307 | 0.071835 | 0.917444 | -0.029321 | 1.373743 | 2.553468 |
| Oil_temperature | 0.483722 | 0.817500 | 0.174357 | 0.964004 | -0.024704 | 1.323659 | 2.691972 |
| TP3 | 0.065095 | 0.141095 | 0.024893 | 0.942951 | 0.010603 | 0.244743 | 0.450062 |

## Per-window and seed results

| Target | Window | Seed | MAE | RMSE | R² | Mean Bias | P95 Absolute Error | P99 Absolute Error |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Oil_temperature | Window 1 | 17 | 0.511287 | 0.700658 | 0.967043 | -0.126055 | 1.304795 | 1.881762 |
| Oil_temperature | Window 1 | 42 | 0.503291 | 0.690888 | 0.967956 | -0.150912 | 1.290187 | 1.916836 |
| Oil_temperature | Window 1 | 73 | 0.490497 | 0.676909 | 0.969239 | -0.171126 | 1.269858 | 1.843757 |
| Oil_temperature | Window 2 | 17 | 0.399956 | 0.708744 | 0.970973 | 0.005131 | 1.097696 | 1.553604 |
| Oil_temperature | Window 2 | 42 | 0.402418 | 0.719896 | 0.970052 | 0.008914 | 1.100529 | 1.569498 |
| Oil_temperature | Window 2 | 73 | 0.398234 | 0.712251 | 0.970685 | 0.001239 | 1.087935 | 1.540356 |
| Oil_temperature | Window 3 | 17 | 0.549946 | 1.049698 | 0.953335 | 0.073544 | 1.588994 | 4.630163 |
| Oil_temperature | Window 3 | 42 | 0.548626 | 1.051173 | 0.953204 | 0.067747 | 1.585430 | 4.677006 |
| Oil_temperature | Window 3 | 73 | 0.549244 | 1.047288 | 0.953549 | 0.069178 | 1.587507 | 4.614763 |
| TP3 | Window 1 | 17 | 0.103762 | 0.191604 | 0.892418 | 0.049422 | 0.490789 | 0.596748 |
| TP3 | Window 1 | 42 | 0.067239 | 0.119042 | 0.958473 | 0.012288 | 0.232244 | 0.340933 |
| TP3 | Window 1 | 73 | 0.087788 | 0.160527 | 0.924486 | 0.039426 | 0.392307 | 0.489389 |
| TP3 | Window 2 | 17 | 0.047046 | 0.118256 | 0.962088 | -0.004317 | 0.141728 | 0.298560 |
| TP3 | Window 2 | 42 | 0.046725 | 0.119773 | 0.961109 | -0.004910 | 0.142673 | 0.290351 |
| TP3 | Window 2 | 73 | 0.046724 | 0.119643 | 0.961193 | -0.004936 | 0.143061 | 0.294491 |
| TP3 | Window 3 | 17 | 0.062247 | 0.147844 | 0.941603 | 0.002374 | 0.221290 | 0.587887 |
| TP3 | Window 3 | 42 | 0.062048 | 0.146671 | 0.942526 | 0.002904 | 0.219714 | 0.573427 |
| TP3 | Window 3 | 73 | 0.062280 | 0.146499 | 0.942660 | 0.003178 | 0.218876 | 0.578772 |
| Motor_current | Window 1 | 17 | 0.283387 | 0.636605 | 0.900512 | -0.062845 | 1.536800 | 2.932181 |
| Motor_current | Window 1 | 42 | 0.295104 | 0.646429 | 0.897418 | -0.037455 | 1.601579 | 2.736217 |
| Motor_current | Window 1 | 73 | 0.289729 | 0.642874 | 0.898543 | -0.041514 | 1.558405 | 2.884027 |
| Motor_current | Window 2 | 17 | 0.206933 | 0.484587 | 0.939557 | -0.005848 | 1.128581 | 2.131021 |
| Motor_current | Window 2 | 42 | 0.204889 | 0.483646 | 0.939791 | -0.005851 | 1.130725 | 2.123705 |
| Motor_current | Window 2 | 73 | 0.204509 | 0.483535 | 0.939819 | -0.010610 | 1.138729 | 2.117472 |
| Motor_current | Window 3 | 17 | 0.243191 | 0.601065 | 0.915197 | -0.033855 | 1.429328 | 2.663257 |
| Motor_current | Window 3 | 42 | 0.246507 | 0.611180 | 0.912319 | -0.032872 | 1.422466 | 2.682779 |
| Motor_current | Window 3 | 73 | 0.246835 | 0.605844 | 0.913843 | -0.033045 | 1.417070 | 2.710554 |

## Interpretation guidance

- Positive mean residual indicates average underprediction.
- Negative mean residual indicates average overprediction.
- Large P95 or P99 absolute errors indicate a heavier error tail.
- Increasing rolling MAE suggests a period of deteriorating prediction accuracy.
- A residual distribution centered near zero does not by itself establish calibrated uncertainty.
- Associations between errors and operating periods are descriptive, not causal.

## Limitations

- This analysis uses the same chronological test windows as Phase 3.
- HistGradientBoosting was selected using the earlier validation experiment.
- Seed 42 plots are representative visualizations, not a separate evaluation.
- High-error timestamps require further investigation before being labelled anomalies.
- Prediction uncertainty and retraining effectiveness have not yet been established.

## Generated visualizations

- Oil_temperature: `outputs/residual_plots/oil_temperature_residual_analysis.png`
- TP3: `outputs/residual_plots/tp3_residual_analysis.png`
- Motor_current: `outputs/residual_plots/motor_current_residual_analysis.png`