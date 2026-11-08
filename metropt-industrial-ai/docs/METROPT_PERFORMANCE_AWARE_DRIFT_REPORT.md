# MetroPT Performance-Aware Drift Evaluation

## Experimental design

- Initial training: first 55% of prepared observations.
- Validation: next 15%.
- Untouched test: final 30%.
- First half of validation monitoring windows establishes the error baseline.
- Second half evaluates candidate deterioration thresholds.
- Monitoring window: 500 consecutive observations.
- Model: HistGradientBoostingRegressor.
- Targets: Oil_temperature, TP3 and Motor_current.

## Target-level summary

         target  monitoring_windows  baseline_windows  evaluation_windows  baseline_mae  baseline_rmse  evaluation_mean_mae  evaluation_mean_rmse  evaluation_mean_error_ratio  drift_alert_rate_pct  performance_alert_rate_pct  drift_performance_cooccurrences  sustained_performance_alerts  overall_validation_r2
Oil_temperature                  44                22                  22       0.46537        0.64377              0.53855               0.71496                      1.15726                 100.0                     9.09091                                2                             0                0.96796
            TP3                  44                22                  22       0.05450        0.09919              0.07885               0.12670                      1.44684                 100.0                    50.00000                               11                             2                0.95847
  Motor_current                  44                22                  22       0.25865        0.55719              0.32122               0.67384                      1.24190                 100.0                    22.72727                                5                             1                0.89742

## Candidate threshold comparison

         target  threshold_multiplier  baseline_windows  evaluation_windows  baseline_mae  baseline_rmse  performance_alerts  sustained_performance_alerts  drift_alerts  drift_performance_cooccurrences  drift_sustained_cooccurrences  performance_alert_rate_pct  drift_alert_rate_pct  drift_performance_cooccurrence_pct
Oil_temperature                  1.25                22                  22       0.46537        0.64377                   7                             1            22                                7                              1                    31.81818                 100.0                            31.81818
Oil_temperature                  1.50                22                  22       0.46537        0.64377                   2                             0            22                                2                              0                     9.09091                 100.0                             9.09091
Oil_temperature                  2.00                22                  22       0.46537        0.64377                   0                             0            22                                0                              0                     0.00000                 100.0                             0.00000
            TP3                  1.25                22                  22       0.05450        0.09919                  14                             5            22                               14                              5                    63.63636                 100.0                            63.63636
            TP3                  1.50                22                  22       0.05450        0.09919                  11                             2            22                               11                              2                    50.00000                 100.0                            50.00000
            TP3                  2.00                22                  22       0.05450        0.09919                   2                             0            22                                2                              0                     9.09091                 100.0                             9.09091
  Motor_current                  1.25                22                  22       0.25865        0.55719                   8                             2            22                                8                              2                    36.36364                 100.0                            36.36364
  Motor_current                  1.50                22                  22       0.25865        0.55719                   5                             1            22                                5                              1                    22.72727                 100.0                            22.72727
  Motor_current                  2.00                22                  22       0.25865        0.55719                   1                             0            22                                1                              0                     4.54545                 100.0                             4.54545

## Interpretation and limitations

Feature drift and predictive performance deterioration are evaluated separately. Their co-occurrence is descriptive and does not establish that feature drift caused prediction error.

The first half of validation windows is used to estimate the baseline error. Candidate thresholds are evaluated on the remaining validation windows. The final 30% is not used.

Two consecutive deterioration windows are required for the sustained alert. Consecutiveness is reset at recording gaps.

The threshold multipliers are candidate operating settings, not universally optimal values. They must be selected using validation evidence and then frozen before final testing.

The PSI and KS detector compares each monitoring window with the initial training reference. KS tests remain sensitive to temporal dependence. These results do not provide ground-truth false-positive rates.

No retraining is performed in this experiment. It evaluates whether drift signals coincide with predictive deterioration.

The final test period remains untouched and is reserved for the subsequent policy comparison.
