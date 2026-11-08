# MetroPT Adaptive Calibration Assessment

## Objective

Compare fixed split-conformal calibration with rolling residual calibration
while keeping the predictive model unchanged.

## Experimental configuration

- Targets: Oil_temperature, TP3 and Motor_current.
- Model: HistGradientBoostingRegressor.
- Seeds: 17, 42 and 73.
- Three chronological train/calibration/test windows.
- Nominal interval levels: 90% and 95%.
- Calibration quantiles use the finite-sample order statistic.
- Rolling windows: 2,000, 5,000 and 10,000 residuals.
- Rolling quantiles are refreshed every 100 test observations.
- Actual test residuals are appended only after the corresponding prediction.
- The predictive model is not retrained during this experiment.

## Aggregate results

| Target | Method | Level | Experiments | Coverage | Coverage SD | Mean Width | Median Width | Lower Miss | Upper Miss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Motor_current | Fixed | 90% | 9 | 0.8609 | 0.0287 | 1.383438 | 1.383438 | 0.0771 | 0.0620 |
| Motor_current | Fixed | 95% | 9 | 0.9256 | 0.0091 | 2.559254 | 2.559254 | 0.0447 | 0.0297 |
| Motor_current | Rolling_10000 | 90% | 9 | 0.8864 | 0.0088 | 1.728410 | 1.767236 | 0.0639 | 0.0497 |
| Motor_current | Rolling_10000 | 95% | 9 | 0.9415 | 0.0059 | 2.968657 | 2.993145 | 0.0362 | 0.0223 |
| Motor_current | Rolling_2000 | 90% | 9 | 0.8969 | 0.0033 | 1.865101 | 1.767359 | 0.0587 | 0.0443 |
| Motor_current | Rolling_2000 | 95% | 9 | 0.9479 | 0.0014 | 3.150116 | 3.065098 | 0.0328 | 0.0194 |
| Motor_current | Rolling_5000 | 90% | 9 | 0.8924 | 0.0074 | 1.806163 | 1.790523 | 0.0609 | 0.0467 |
| Motor_current | Rolling_5000 | 95% | 9 | 0.9454 | 0.0039 | 3.086200 | 3.036242 | 0.0342 | 0.0204 |
| Oil_temperature | Fixed | 90% | 9 | 0.8495 | 0.0236 | 1.869351 | 1.869351 | 0.0627 | 0.0878 |
| Oil_temperature | Fixed | 95% | 9 | 0.9162 | 0.0238 | 2.383055 | 2.383055 | 0.0305 | 0.0534 |
| Oil_temperature | Rolling_10000 | 90% | 9 | 0.8818 | 0.0122 | 2.101608 | 2.057327 | 0.0471 | 0.0711 |
| Oil_temperature | Rolling_10000 | 95% | 9 | 0.9379 | 0.0126 | 2.665915 | 2.577835 | 0.0215 | 0.0407 |
| Oil_temperature | Rolling_2000 | 90% | 9 | 0.8947 | 0.0040 | 2.301471 | 2.094311 | 0.0410 | 0.0643 |
| Oil_temperature | Rolling_2000 | 95% | 9 | 0.9465 | 0.0042 | 2.883305 | 2.593227 | 0.0186 | 0.0348 |
| Oil_temperature | Rolling_5000 | 90% | 9 | 0.8887 | 0.0085 | 2.189522 | 2.099289 | 0.0434 | 0.0679 |
| Oil_temperature | Rolling_5000 | 95% | 9 | 0.9430 | 0.0081 | 2.773261 | 2.588086 | 0.0197 | 0.0372 |
| TP3 | Fixed | 90% | 9 | 0.8638 | 0.0140 | 0.310090 | 0.310090 | 0.0504 | 0.0858 |
| TP3 | Fixed | 95% | 9 | 0.9232 | 0.0151 | 0.420000 | 0.420000 | 0.0312 | 0.0456 |
| TP3 | Rolling_10000 | 90% | 9 | 0.8849 | 0.0110 | 0.346641 | 0.341547 | 0.0456 | 0.0696 |
| TP3 | Rolling_10000 | 95% | 9 | 0.9402 | 0.0084 | 0.453703 | 0.441217 | 0.0264 | 0.0334 |
| TP3 | Rolling_2000 | 90% | 9 | 0.8921 | 0.0042 | 0.368974 | 0.342269 | 0.0427 | 0.0652 |
| TP3 | Rolling_2000 | 95% | 9 | 0.9439 | 0.0026 | 0.473143 | 0.436417 | 0.0251 | 0.0310 |
| TP3 | Rolling_5000 | 90% | 9 | 0.8886 | 0.0101 | 0.356381 | 0.357412 | 0.0442 | 0.0673 |
| TP3 | Rolling_5000 | 95% | 9 | 0.9417 | 0.0075 | 0.464259 | 0.455505 | 0.0265 | 0.0317 |

## Window-level results

| Target | Window | Method | Level | Coverage | Mean Width | Coverage Gap | Mean Quantile | RMSE |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Motor_current | Window 1 | Fixed | 90% | 0.8248 | 1.051928 | 0.0752 | 0.525964 | 0.648451 |
| Motor_current | Window 1 | Fixed | 95% | 0.9149 | 2.246918 | 0.0351 | 1.123459 | 0.648451 |
| Motor_current | Window 1 | Rolling_10000 | 90% | 0.8804 | 1.687790 | 0.0196 | 0.843895 | 0.648451 |
| Motor_current | Window 1 | Rolling_10000 | 95% | 0.9408 | 2.859589 | 0.0092 | 1.429794 | 0.648451 |
| Motor_current | Window 1 | Rolling_2000 | 90% | 0.8949 | 1.882891 | 0.0051 | 0.941445 | 0.648451 |
| Motor_current | Window 1 | Rolling_2000 | 95% | 0.9469 | 3.044444 | 0.0031 | 1.522222 | 0.648451 |
| Motor_current | Window 1 | Rolling_5000 | 90% | 0.8891 | 1.797400 | 0.0109 | 0.898700 | 0.648451 |
| Motor_current | Window 1 | Rolling_5000 | 95% | 0.9438 | 2.962431 | 0.0062 | 1.481216 | 0.648451 |
| Motor_current | Window 2 | Fixed | 90% | 0.8833 | 1.891269 | 0.0167 | 0.945635 | 0.738537 |
| Motor_current | Window 2 | Fixed | 95% | 0.9354 | 3.180757 | 0.0146 | 1.590379 | 0.738537 |
| Motor_current | Window 2 | Rolling_10000 | 90% | 0.8980 | 2.207297 | 0.0020 | 1.103649 | 0.738537 |
| Motor_current | Window 2 | Rolling_10000 | 95% | 0.9487 | 3.685915 | 0.0013 | 1.842957 | 0.738537 |
| Motor_current | Window 2 | Rolling_2000 | 90% | 0.9013 | 2.275193 | -0.0013 | 1.137596 | 0.738537 |
| Motor_current | Window 2 | Rolling_2000 | 95% | 0.9497 | 3.909871 | 0.0003 | 1.954935 | 0.738537 |
| Motor_current | Window 2 | Rolling_5000 | 90% | 0.9021 | 2.271863 | -0.0021 | 1.135931 | 0.738537 |
| Motor_current | Window 2 | Rolling_5000 | 95% | 0.9504 | 3.868105 | -0.0004 | 1.934052 | 0.738537 |
| Motor_current | Window 3 | Fixed | 90% | 0.8746 | 1.207116 | 0.0254 | 0.603558 | 0.625519 |
| Motor_current | Window 3 | Fixed | 95% | 0.9265 | 2.250087 | 0.0235 | 1.125044 | 0.625519 |
| Motor_current | Window 3 | Rolling_10000 | 90% | 0.8806 | 1.290142 | 0.0194 | 0.645071 | 0.625519 |
| Motor_current | Window 3 | Rolling_10000 | 95% | 0.9351 | 2.360466 | 0.0149 | 1.180233 | 0.625519 |
| Motor_current | Window 3 | Rolling_2000 | 90% | 0.8946 | 1.437219 | 0.0054 | 0.718609 | 0.625519 |
| Motor_current | Window 3 | Rolling_2000 | 95% | 0.9470 | 2.496033 | 0.0030 | 1.248017 | 0.625519 |
| Motor_current | Window 3 | Rolling_5000 | 90% | 0.8861 | 1.349227 | 0.0139 | 0.674613 | 0.625519 |
| Motor_current | Window 3 | Rolling_5000 | 95% | 0.9419 | 2.428063 | 0.0081 | 1.214032 | 0.625519 |
| Oil_temperature | Window 1 | Fixed | 90% | 0.8427 | 2.046216 | 0.0573 | 1.023108 | 0.762016 |
| Oil_temperature | Window 1 | Fixed | 95% | 0.9239 | 2.689500 | 0.0261 | 1.344750 | 0.762016 |
| Oil_temperature | Window 1 | Rolling_10000 | 90% | 0.8938 | 2.404857 | 0.0062 | 1.202429 | 0.762016 |
| Oil_temperature | Window 1 | Rolling_10000 | 95% | 0.9493 | 3.043676 | 0.0007 | 1.521838 | 0.762016 |
| Oil_temperature | Window 1 | Rolling_2000 | 90% | 0.8982 | 2.447700 | 0.0018 | 1.223850 | 0.762016 |
| Oil_temperature | Window 1 | Rolling_2000 | 95% | 0.9498 | 3.034574 | 0.0002 | 1.517287 | 0.762016 |
| Oil_temperature | Window 1 | Rolling_5000 | 90% | 0.8960 | 2.437525 | 0.0040 | 1.218763 | 0.762016 |
| Oil_temperature | Window 1 | Rolling_5000 | 95% | 0.9493 | 3.051069 | 0.0007 | 1.525534 | 0.762016 |
| Oil_temperature | Window 2 | Fixed | 90% | 0.8775 | 1.850536 | 0.0225 | 0.925268 | 0.774740 |
| Oil_temperature | Window 2 | Fixed | 95% | 0.9380 | 2.292974 | 0.0120 | 1.146487 | 0.774740 |
| Oil_temperature | Window 2 | Rolling_10000 | 90% | 0.8853 | 1.892506 | 0.0147 | 0.946253 | 0.774740 |
| Oil_temperature | Window 2 | Rolling_10000 | 95% | 0.9427 | 2.323044 | 0.0073 | 1.161522 | 0.774740 |
| Oil_temperature | Window 2 | Rolling_2000 | 90% | 0.8965 | 1.947944 | 0.0035 | 0.973972 | 0.774740 |
| Oil_temperature | Window 2 | Rolling_2000 | 95% | 0.9488 | 2.378951 | 0.0012 | 1.189475 | 0.774740 |
| Oil_temperature | Window 2 | Rolling_5000 | 90% | 0.8925 | 1.928590 | 0.0075 | 0.964295 | 0.774740 |
| Oil_temperature | Window 2 | Rolling_5000 | 95% | 0.9475 | 2.359477 | 0.0025 | 1.179739 | 0.774740 |
| Oil_temperature | Window 3 | Fixed | 90% | 0.8283 | 1.711302 | 0.0717 | 0.855651 | 1.065051 |
| Oil_temperature | Window 3 | Fixed | 95% | 0.8866 | 2.166692 | 0.0634 | 1.083346 | 1.065051 |
| Oil_temperature | Window 3 | Rolling_10000 | 90% | 0.8663 | 2.007460 | 0.0337 | 1.003730 | 1.065051 |
| Oil_temperature | Window 3 | Rolling_10000 | 95% | 0.9216 | 2.631024 | 0.0284 | 1.315512 | 1.065051 |
| Oil_temperature | Window 3 | Rolling_2000 | 90% | 0.8895 | 2.508769 | 0.0105 | 1.254385 | 1.065051 |
| Oil_temperature | Window 3 | Rolling_2000 | 95% | 0.9410 | 3.236389 | 0.0090 | 1.618195 | 1.065051 |
| Oil_temperature | Window 3 | Rolling_5000 | 90% | 0.8776 | 2.202449 | 0.0224 | 1.101225 | 1.065051 |
| Oil_temperature | Window 3 | Rolling_5000 | 95% | 0.9323 | 2.909237 | 0.0177 | 1.454619 | 1.065051 |
| TP3 | Window 1 | Fixed | 90% | 0.8657 | 0.242765 | 0.0343 | 0.121382 | 0.110526 |
| TP3 | Window 1 | Fixed | 95% | 0.9236 | 0.341374 | 0.0264 | 0.170687 | 0.110526 |
| TP3 | Window 1 | Rolling_10000 | 90% | 0.8828 | 0.260448 | 0.0172 | 0.130224 | 0.110526 |
| TP3 | Window 1 | Rolling_10000 | 95% | 0.9388 | 0.360339 | 0.0112 | 0.180170 | 0.110526 |
| TP3 | Window 1 | Rolling_2000 | 90% | 0.8894 | 0.270717 | 0.0106 | 0.135358 | 0.110526 |
| TP3 | Window 1 | Rolling_2000 | 95% | 0.9430 | 0.358856 | 0.0070 | 0.179428 | 0.110526 |
| TP3 | Window 1 | Rolling_5000 | 90% | 0.8841 | 0.265826 | 0.0159 | 0.132913 | 0.110526 |
| TP3 | Window 1 | Rolling_5000 | 95% | 0.9406 | 0.360833 | 0.0094 | 0.180416 | 0.110526 |
| TP3 | Window 2 | Fixed | 90% | 0.8691 | 0.500235 | 0.0309 | 0.250117 | 0.170346 |
| TP3 | Window 2 | Fixed | 95% | 0.9342 | 0.638881 | 0.0158 | 0.319440 | 0.170346 |
| TP3 | Window 2 | Rolling_10000 | 90% | 0.8983 | 0.569908 | 0.0017 | 0.284954 | 0.170346 |
| TP3 | Window 2 | Rolling_10000 | 95% | 0.9503 | 0.670242 | -0.0003 | 0.335121 | 0.170346 |
| TP3 | Window 2 | Rolling_2000 | 90% | 0.8977 | 0.558742 | 0.0023 | 0.279371 | 0.170346 |
| TP3 | Window 2 | Rolling_2000 | 95% | 0.9472 | 0.664919 | 0.0028 | 0.332460 | 0.170346 |
| TP3 | Window 2 | Rolling_5000 | 90% | 0.9017 | 0.568315 | -0.0017 | 0.284158 | 0.170346 |
| TP3 | Window 2 | Rolling_5000 | 95% | 0.9508 | 0.670364 | -0.0008 | 0.335182 | 0.170346 |
| TP3 | Window 3 | Fixed | 90% | 0.8566 | 0.187271 | 0.0434 | 0.093636 | 0.150275 |
| TP3 | Window 3 | Fixed | 95% | 0.9118 | 0.279744 | 0.0382 | 0.139872 | 0.150275 |
| TP3 | Window 3 | Rolling_10000 | 90% | 0.8736 | 0.209567 | 0.0264 | 0.104783 | 0.150275 |
| TP3 | Window 3 | Rolling_10000 | 95% | 0.9315 | 0.330529 | 0.0185 | 0.165265 | 0.150275 |
| TP3 | Window 3 | Rolling_2000 | 90% | 0.8892 | 0.277463 | 0.0108 | 0.138732 | 0.150275 |
| TP3 | Window 3 | Rolling_2000 | 95% | 0.9415 | 0.395655 | 0.0085 | 0.197827 | 0.150275 |
| TP3 | Window 3 | Rolling_5000 | 90% | 0.8799 | 0.235000 | 0.0201 | 0.117500 | 0.150275 |
| TP3 | Window 3 | Rolling_5000 | 95% | 0.9338 | 0.361580 | 0.0162 | 0.180790 | 0.150275 |

## Method comparison

The fixed method uses the original calibration residuals throughout testing.
Rolling methods use the most recent observed residuals in a bounded buffer.
Quantile recalculation occurs every 100 test observations.

The model remains fixed so that changes in interval performance can be
attributed to calibration updates rather than predictive model retraining.

## Important limitations

- This is a prequential online simulation, not a deployment experiment.
- Rolling calibration uses observed outcomes after they become available.
- The first prediction in each test period uses calibration-period residuals.
- Updating every 100 observations means quantiles can be stale within each block.
- Serial dependence and temporal distribution changes violate the usual
  exchangeability assumptions behind standard conformal coverage guarantees.
- Rolling calibration does not guarantee nominal coverage.
- Repeated seeds do not establish independent machine generalization.
- No model retraining is performed in this phase.

## Interpretation guidance

- Compare coverage against the nominal interval level.
- Examine interval width alongside coverage; wider intervals are not automatically better.
- Check whether rolling calibration improves coverage in difficult windows.
- Check whether improvement requires an unacceptable increase in interval width.
- Inspect lower and upper miss rates for asymmetric error patterns.
- Consider recalibration overhead separately from model retraining cost.

## Next stage

Integrate drift and impact monitoring with cost-sensitive model retraining.
Compare the additional benefit of model updates against adaptive calibration alone.