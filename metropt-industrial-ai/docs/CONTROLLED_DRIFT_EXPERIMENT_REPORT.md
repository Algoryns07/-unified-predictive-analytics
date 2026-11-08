# Controlled Drift Experiment Report

## Experimental design

- Training observations: 5000
- Calibration observations: 1000
- Evaluation observations: 4000
- Monitoring window: 200
- Drift injection point: 2000
- Random seeds: [17, 42, 73, 101, 2026]
- Performance threshold: 1.5x calibration MAE
- PSI threshold: 0.2
- KS BH-adjusted q threshold: 0.05
- Sustained performance alert: two consecutive windows.

## Scenario definitions

- No drift: feature distributions and target relationship remain unchanged.
- Covariate drift: two feature distributions shift; the target relationship remains unchanged.
- Concept drift: the target-generating relationship changes; feature distributions remain unchanged.

## Aggregate results

       scenario  experiments  mean_false_drift_rate_pct  mean_false_performance_rate_pct  mean_false_sustained_rate_pct  feature_detection_rate_pct  performance_detection_rate_pct  mean_post_drift_alert_rate_pct  mean_post_performance_alert_rate_pct  mean_feature_detection_delay  mean_performance_detection_delay  mean_post_error_ratio
       no_drift            5                        1.0                              0.0                            0.0                         NaN                             NaN                             NaN                                   NaN                           NaN                               NaN                 0.9948
covariate_drift            5                        0.0                              0.0                            0.0                       100.0                           100.0                           100.0                                  86.0                         200.0                             520.0                 1.6588
  concept_drift            5                        0.0                              0.0                            0.0                        20.0                           100.0                             2.0                                 100.0                        2000.0                             400.0                12.7897

## Per-seed results

       scenario  seed  baseline_mae  pre_injection_windows  post_injection_windows  pre_false_drift_rate_pct  pre_false_performance_rate_pct  pre_false_sustained_rate_pct  post_drift_alert_rate_pct  post_performance_alert_rate_pct  post_sustained_alert_rate_pct  feature_drift_detected  performance_drift_detected  feature_detection_delay_observations  performance_detection_delay_observations  post_mean_mae  post_mean_error_ratio
       no_drift    17        0.3490                     20                      20                       0.0                             0.0                           0.0                        0.0                              0.0                            0.0                   False                       False                                   NaN                                       NaN         0.3487                 0.9992
covariate_drift    17        0.3490                     10                      10                       0.0                             0.0                           0.0                      100.0                             80.0                           60.0                    True                        True                                 200.0                                     400.0         0.5543                 1.5884
  concept_drift    17        0.3490                     10                      10                       0.0                             0.0                           0.0                        0.0                            100.0                           90.0                   False                        True                                   NaN                                     400.0         4.4604                12.7811
       no_drift    42        0.3499                     20                      20                       0.0                             0.0                           0.0                        0.0                              0.0                            0.0                   False                       False                                   NaN                                       NaN         0.3491                 0.9977
covariate_drift    42        0.3499                     10                      10                       0.0                             0.0                           0.0                      100.0                            100.0                           90.0                    True                        True                                 200.0                                     400.0         0.5994                 1.7132
  concept_drift    42        0.3499                     10                      10                       0.0                             0.0                           0.0                        0.0                            100.0                           90.0                   False                        True                                   NaN                                     400.0         4.4392                12.6878
       no_drift    73        0.3346                     20                      20                       0.0                             0.0                           0.0                        0.0                              0.0                            0.0                   False                       False                                   NaN                                       NaN         0.3282                 0.9810
covariate_drift    73        0.3346                     10                      10                       0.0                             0.0                           0.0                      100.0                             90.0                           70.0                    True                        True                                 200.0                                     400.0         0.5507                 1.6460
  concept_drift    73        0.3346                     10                      10                       0.0                             0.0                           0.0                        0.0                            100.0                           90.0                   False                        True                                   NaN                                     400.0         4.3699                13.0617
       no_drift   101        0.3525                     20                      20                       0.0                             0.0                           0.0                        0.0                              0.0                            0.0                   False                       False                                   NaN                                       NaN         0.3478                 0.9869
covariate_drift   101        0.3525                     10                      10                       0.0                             0.0                           0.0                      100.0                             60.0                           30.0                    True                        True                                 200.0                                    1000.0         0.5640                 1.6002
  concept_drift   101        0.3525                     10                      10                       0.0                             0.0                           0.0                        0.0                            100.0                           90.0                   False                        True                                   NaN                                     400.0         4.4037                12.4935
       no_drift  2026        0.3411                     20                      20                       5.0                             0.0                           0.0                        5.0                              0.0                            0.0                   False                       False                                   NaN                                       NaN         0.3441                 1.0090
covariate_drift  2026        0.3411                     10                      10                       0.0                             0.0                           0.0                      100.0                            100.0                           90.0                    True                        True                                 200.0                                     400.0         0.5957                 1.7464
  concept_drift  2026        0.3411                     10                      10                       0.0                             0.0                           0.0                       10.0                            100.0                           90.0                    True                        True                                2000.0                                     400.0         4.4082                12.9242

## Interpretation and limitations

These are controlled synthetic experiments. They test whether monitoring methods respond to known constructed changes; they do not establish the frequency or nature of drift in real MetroPT operations.

Covariate drift changes input distributions while retaining the same target function. Concept drift changes the target function while preserving the input distribution.

The model remains fixed throughout each evaluation stream. No retraining is performed in this experiment.

The performance threshold is calibrated on a separate synthetic calibration stream. Detection delay is measured in observations from the known injection point to the first qualifying alert.

A feature-drift alert is not equivalent to harmful predictive drift. Likewise, a performance alert is not automatically proof of concept drift.

The five seeds provide a small reproducibility check, not a comprehensive statistical guarantee.
