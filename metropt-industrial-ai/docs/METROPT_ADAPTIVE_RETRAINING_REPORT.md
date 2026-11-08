# MetroPT Adaptive Retraining Policy Comparison

## Experimental design

- Dataset: MetroPT minute-level observations.
- Targets: Oil_temperature, TP3, Motor_current.
- Chronological split: 55% training, 15% validation, 30% final test.
- Random seeds: 17, 42, 73.
- Monitoring window: 500 observations.
- Periodic interval: 5,000 observations.
- Performance threshold: 1.5 times validation baseline MAE.
- Performance trigger: two consecutive alert windows.
- Cooldown: 2,000 observations.
- Impact approval: at least 5% observed improvement and benefit-cost score at least 0.05.

## Policy definitions

1. P1: fixed initial model.
2. P2: periodic retraining.
3. P3: sustained predictive-performance deterioration.
4. P4: challenger evaluation on the latest observed window, followed by impact/cost approval.

## Aggregate results

         target                   policy  mean_mae  std_mae  mean_rmse  std_rmse  mean_r2  mean_retraining_events  mean_retraining_runtime_seconds  mean_retraining_training_rows  mean_candidate_acceptances
Oil_temperature         P1_No_Retraining   0.58620  0.02450    0.98048   0.02190  0.95421                 0.00000                          0.00000                            0.0                     0.00000
Oil_temperature              P2_Periodic   0.47416  0.00146    0.89011   0.00183  0.96227                10.00000                          9.39193                      1515820.0                     0.00000
Oil_temperature P3_Performance_Triggered   0.56249  0.02481    0.95033   0.01830  0.95698                 2.33333                          2.44991                       400358.0                     0.00000
Oil_temperature          P4_Impact_Aware   0.56520  0.02574    0.95926   0.02237  0.95617                 0.66667                          7.13138                      1210074.0                     0.66667
            TP3         P1_No_Retraining   0.11779  0.02146    0.20237   0.02595  0.88858                 0.00000                          0.00000                            0.0                     0.00000
            TP3              P2_Periodic   0.05982  0.00239    0.13857   0.00414  0.94830                10.00000                          9.43419                      1515820.0                     0.00000
            TP3 P3_Performance_Triggered   0.05822  0.00100    0.13748   0.00122  0.94914                 3.00000                          2.93550                       477246.0                     0.00000
            TP3          P4_Impact_Aware   0.05838  0.00094    0.13780   0.00124  0.94890                 2.00000                          5.70254                       951992.0                     2.00000
  Motor_current         P1_No_Retraining   0.38777  0.01000    0.82649   0.02189  0.83363                 0.00000                          0.00000                            0.0                     0.00000
  Motor_current              P2_Periodic   0.24113  0.00078    0.57930   0.00277  0.91830                10.00000                          9.43066                      1515820.0                     0.00000
  Motor_current P3_Performance_Triggered   0.24286  0.00057    0.58483   0.00072  0.91674                 3.00000                          2.87564                       477246.0                     0.00000
  Motor_current          P4_Impact_Aware   0.24317  0.00054    0.58587   0.00082  0.91644                 2.00000                          5.64683                       952992.0                     2.00000

## Per-seed results

         target  seed                   policy  test_observations  baseline_validation_mae  test_mae  test_rmse  test_r2  total_absolute_error  retraining_events  candidate_attempts  candidate_acceptances  retraining_runtime_seconds  retraining_training_rows  initial_fit_seconds
Oil_temperature    17         P1_No_Retraining              53179                  0.45838   0.55872    0.95666  0.95642           29712.31306                  0                   0                      0                     0.00000                         0              0.68533
Oil_temperature    17              P2_Periodic              53179                  0.45838   0.47277    0.88879  0.96238           25141.35063                 10                   0                      0                     9.70816                   1515820              0.68533
Oil_temperature    17 P3_Performance_Triggered              53179                  0.45838   0.55553    0.94135  0.95780           29542.74273                  2                   0                      0                     2.22410                    351164              0.68533
Oil_temperature    17          P4_Impact_Aware              53179                  0.45838   0.55671    0.94513  0.95746           29605.20754                  1                   4                      1                     4.90617                    877410              0.68533
Oil_temperature    42         P1_No_Retraining              53179                  0.45954   0.59410    0.98505  0.95379           31593.77385                  0                   0                      0                     0.00000                         0              0.58367
Oil_temperature    42              P2_Periodic              53179                  0.45954   0.47403    0.88933  0.96234           25208.42709                 10                   0                      0                     9.18620                   1515820              0.58367
Oil_temperature    42 P3_Performance_Triggered              53179                  0.45954   0.59004    0.97138  0.95507           31377.89962                  2                   0                      0                     2.11922                    351164              0.58367
Oil_temperature    42          P4_Impact_Aware              53179                  0.45954   0.59410    0.98505  0.95379           31593.77385                  0                   7                      0                     7.36653                   1229074              0.58367
Oil_temperature    73         P1_No_Retraining              53179                  0.45618   0.60576    0.99974  0.95241           32213.94109                  0                   0                      0                     0.00000                         0              0.66329
Oil_temperature    73              P2_Periodic              53179                  0.45618   0.47568    0.89220  0.96209           25296.37201                 10                   0                      0                     9.28142                   1515820              0.66329
Oil_temperature    73 P3_Performance_Triggered              53179                  0.45618   0.54190    0.93826  0.95808           28817.67055                  3                   0                      0                     3.00641                    498746              0.66329
Oil_temperature    73          P4_Impact_Aware              53179                  0.45618   0.54478    0.94761  0.95724           28970.62152                  1                   8                      1                     9.12142                   1523738              0.66329
            TP3    17         P1_No_Retraining              53179                  0.06841   0.14168    0.23012  0.85750            7534.15923                  0                   0                      0                     0.00000                         0              0.67492
            TP3    17              P2_Periodic              53179                  0.06841   0.06253    0.14315  0.94486            3325.12767                 10                   0                      0                     9.16794                   1515820              0.67492
            TP3    17 P3_Performance_Triggered              53179                  0.06841   0.05936    0.13888  0.94810            3156.49257                  3                   0                      0                     2.91973                    477246              0.67492
            TP3    17          P4_Impact_Aware              53179                  0.06841   0.05946    0.13922  0.94784            3161.82807                  2                   4                      2                     5.52417                    951992              0.67492
            TP3    42         P1_No_Retraining              53179                  0.05738   0.10013    0.17869  0.91407            5324.65515                  0                   0                      0                     0.00000                         0              0.57186
            TP3    42              P2_Periodic              53179                  0.05738   0.05800    0.13508  0.95090            3084.46702                 10                   0                      0                     9.34477                   1515820              0.57186
            TP3    42 P3_Performance_Triggered              53179                  0.05738   0.05752    0.13662  0.94977            3058.83112                  3                   0                      0                     2.97758                    477246              0.57186
            TP3    42          P4_Impact_Aware              53179                  0.05738   0.05779    0.13708  0.94944            3073.44357                  2                   4                      2                     5.63360                    951992              0.57186
            TP3    73         P1_No_Retraining              53179                  0.06000   0.11158    0.19831  0.89417            5933.49000                  0                   0                      0                     0.00000                         0              0.65356
            TP3    73              P2_Periodic              53179                  0.06000   0.05892    0.13748  0.94913            3133.12824                 10                   0                      0                     9.78986                   1515820              0.65356
            TP3    73 P3_Performance_Triggered              53179                  0.06000   0.05777    0.13694  0.94954            3072.28386                  3                   0                      0                     2.90917                    477246              0.65356
            TP3    73          P4_Impact_Aware              53179                  0.06000   0.05788    0.13709  0.94943            3078.13700                  2                   4                      2                     5.94985                    951992              0.65356
  Motor_current    17         P1_No_Retraining              53179                  0.28197   0.37644    0.80122  0.84373           20018.93634                  0                   0                      0                     0.00000                         0              0.69970
  Motor_current    17              P2_Periodic              53179                  0.28197   0.24030    0.57620  0.91918           12779.15619                 10                   0                      0                     9.60332                   1515820              0.69970
  Motor_current    17 P3_Performance_Triggered              53179                  0.28197   0.24285    0.58414  0.91693           12914.63125                  3                   0                      0                     2.91551                    477246              0.69970
  Motor_current    17          P4_Impact_Aware              53179                  0.28197   0.24310    0.58504  0.91668           12927.55692                  2                   4                      2                     5.75729                    952992              0.69970
  Motor_current    42         P1_No_Retraining              53179                  0.28895   0.39540    0.83911  0.82859           21027.19537                  0                   0                      0                     0.00000                         0              0.62963
  Motor_current    42              P2_Periodic              53179                  0.28895   0.24185    0.58154  0.91767           12861.27454                 10                   0                      0                     9.42053                   1515820              0.62963
  Motor_current    42 P3_Performance_Triggered              53179                  0.28895   0.24343    0.58477  0.91675           12945.34131                  3                   0                      0                     2.87991                    477246              0.62963
  Motor_current    42          P4_Impact_Aware              53179                  0.28895   0.24373    0.58590  0.91643           12961.54642                  2                   4                      2                     5.70206                    952992              0.62963
  Motor_current    73         P1_No_Retraining              53179                  0.30407   0.39145    0.83914  0.82858           20817.04557                  0                   0                      0                     0.00000                         0              0.62299
  Motor_current    73              P2_Periodic              53179                  0.30407   0.24125    0.58016  0.91806           12829.21988                 10                   0                      0                     9.26813                   1515820              0.62299
  Motor_current    73 P3_Performance_Triggered              53179                  0.30407   0.24228    0.58559  0.91652           12884.46675                  3                   0                      0                     2.83150                    477246              0.62299
  Motor_current    73          P4_Impact_Aware              53179                  0.30407   0.24267    0.58667  0.91621           12904.82547                  2                   4                      2                     5.48115                    952992              0.62299

## Methodological safeguards

- Predictions are generated before the corresponding actual targets are used.
- Retraining decisions affect only future predictions.
- All policies begin with the same initial fitted model.
- Validation data calibrate the performance baseline and are not used to train the initial model.
- The final chronological test stream is not used to tune thresholds.
- Feature drift is recorded as diagnostic evidence and is not a mandatory retraining condition.
- Retraining cost is represented by measured fitting runtime and training-data volume, not invented monetary costs.

## Interpretation limitations

The impact-aware challenger is evaluated on the latest observed window, which is excluded from challenger fitting. After the decision, that window becomes available for subsequent training.

The 5% improvement threshold and benefit-cost score are provisional operational choices, not statistically optimized universal thresholds.

Results describe this dataset, feature construction, model and chronological evaluation period. They do not establish failure prediction or universal retraining superiority.

Retraining policies are compared on the same test stream, but repeated sequential decisions mean the resulting observations are temporally dependent.
