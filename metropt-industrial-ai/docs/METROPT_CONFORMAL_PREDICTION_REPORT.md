# MetroPT Split Conformal Prediction Assessment

## Experimental design

- Base estimator: HistGradientBoostingRegressor.
- Three targets: Oil_temperature, TP3 and Motor_current.
- Three chronological windows.
- Three random seeds: 17, 42 and 73.
- Training, calibration and test periods are kept separate.
- Calibration uses absolute residuals.
- Finite-sample conformal order statistic is used.
- Prediction intervals are symmetric around point predictions.
- Nominal interval levels: 90% and 95%.

## Aggregate results

| Target | Level | Experiments | Mean Coverage | Mean Width | Mean Median Width | Mean Lower Miss | Mean Upper Miss | Mean RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Motor_current | 90% | 9 | 0.8609 | 1.383438 | 1.383438 | 0.0771 | 0.0620 | 0.670836 |
| Motor_current | 95% | 9 | 0.9256 | 2.559254 | 2.559254 | 0.0447 | 0.0297 | 0.670836 |
| Oil_temperature | 90% | 9 | 0.8495 | 1.869351 | 1.869351 | 0.0627 | 0.0878 | 0.867269 |
| Oil_temperature | 95% | 9 | 0.9162 | 2.383055 | 2.383055 | 0.0305 | 0.0534 | 0.867269 |
| TP3 | 90% | 9 | 0.8638 | 0.310090 | 0.310090 | 0.0504 | 0.0858 | 0.143716 |
| TP3 | 95% | 9 | 0.9232 | 0.420000 | 0.420000 | 0.0312 | 0.0456 | 0.143716 |

## Per-window results

| Target | Window | Seed | Level | Coverage | Mean Width | Calibration Quantile | Lower Miss | Upper Miss | RMSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Oil_temperature | Window 1 | 17 | 90% | 0.8233 | 2.051369 | 1.025684 | 0.0820 | 0.0947 | 0.795938 |
| Oil_temperature | Window 1 | 17 | 95% | 0.9098 | 2.711558 | 1.355779 | 0.0341 | 0.0562 | 0.795938 |
| Oil_temperature | Window 1 | 42 | 90% | 0.8577 | 2.047389 | 1.023695 | 0.0515 | 0.0908 | 0.741803 |
| Oil_temperature | Window 1 | 42 | 95% | 0.9300 | 2.684290 | 1.342145 | 0.0181 | 0.0518 | 0.741803 |
| Oil_temperature | Window 1 | 73 | 90% | 0.8470 | 2.039891 | 1.019946 | 0.0824 | 0.0707 | 0.748308 |
| Oil_temperature | Window 1 | 73 | 95% | 0.9317 | 2.672652 | 1.336326 | 0.0337 | 0.0346 | 0.748308 |
| Oil_temperature | Window 2 | 17 | 90% | 0.8780 | 1.849035 | 0.924518 | 0.0663 | 0.0557 | 0.770620 |
| Oil_temperature | Window 2 | 17 | 95% | 0.9367 | 2.266687 | 1.133343 | 0.0317 | 0.0316 | 0.770620 |
| Oil_temperature | Window 2 | 42 | 90% | 0.8784 | 1.868326 | 0.934163 | 0.0610 | 0.0606 | 0.776787 |
| Oil_temperature | Window 2 | 42 | 95% | 0.9408 | 2.344144 | 1.172072 | 0.0246 | 0.0346 | 0.776787 |
| Oil_temperature | Window 2 | 73 | 90% | 0.8760 | 1.834248 | 0.917124 | 0.0672 | 0.0568 | 0.776815 |
| Oil_temperature | Window 2 | 73 | 95% | 0.9365 | 2.268091 | 1.134046 | 0.0311 | 0.0324 | 0.776815 |
| Oil_temperature | Window 3 | 17 | 90% | 0.8287 | 1.706768 | 0.853384 | 0.0514 | 0.1198 | 1.063163 |
| Oil_temperature | Window 3 | 17 | 95% | 0.8865 | 2.158241 | 1.079121 | 0.0338 | 0.0797 | 1.063163 |
| Oil_temperature | Window 3 | 42 | 90% | 0.8264 | 1.704362 | 0.852181 | 0.0510 | 0.1227 | 1.066470 |
| Oil_temperature | Window 3 | 42 | 95% | 0.8854 | 2.159001 | 1.079501 | 0.0335 | 0.0810 | 1.066470 |
| Oil_temperature | Window 3 | 73 | 90% | 0.8299 | 1.722776 | 0.861388 | 0.0514 | 0.1187 | 1.065522 |
| Oil_temperature | Window 3 | 73 | 95% | 0.8880 | 2.182834 | 1.091417 | 0.0335 | 0.0785 | 1.065522 |
| TP3 | Window 1 | 17 | 90% | 0.8845 | 0.244136 | 0.122068 | 0.0522 | 0.0633 | 0.109278 |
| TP3 | Window 1 | 17 | 95% | 0.9345 | 0.342275 | 0.171138 | 0.0321 | 0.0334 | 0.109278 |
| TP3 | Window 1 | 42 | 90% | 0.8760 | 0.242977 | 0.121489 | 0.0511 | 0.0729 | 0.107742 |
| TP3 | Window 1 | 42 | 95% | 0.9391 | 0.343977 | 0.171988 | 0.0301 | 0.0308 | 0.107742 |
| TP3 | Window 1 | 73 | 90% | 0.8366 | 0.241181 | 0.120590 | 0.0447 | 0.1187 | 0.114559 |
| TP3 | Window 1 | 73 | 95% | 0.8973 | 0.337871 | 0.168936 | 0.0247 | 0.0780 | 0.114559 |
| TP3 | Window 2 | 17 | 90% | 0.8683 | 0.472923 | 0.236462 | 0.0158 | 0.1160 | 0.165700 |
| TP3 | Window 2 | 17 | 95% | 0.9305 | 0.595222 | 0.297611 | 0.0089 | 0.0606 | 0.165700 |
| TP3 | Window 2 | 42 | 90% | 0.8693 | 0.538848 | 0.269424 | 0.0106 | 0.1201 | 0.174254 |
| TP3 | Window 2 | 42 | 95% | 0.9366 | 0.669835 | 0.334918 | 0.0059 | 0.0575 | 0.174254 |
| TP3 | Window 2 | 73 | 90% | 0.8696 | 0.488933 | 0.244466 | 0.0119 | 0.1185 | 0.171086 |
| TP3 | Window 2 | 73 | 95% | 0.9353 | 0.651586 | 0.325793 | 0.0064 | 0.0582 | 0.171086 |
| TP3 | Window 3 | 17 | 90% | 0.8565 | 0.185986 | 0.092993 | 0.0904 | 0.0531 | 0.150176 |
| TP3 | Window 3 | 17 | 95% | 0.9107 | 0.282325 | 0.141162 | 0.0590 | 0.0303 | 0.150176 |
| TP3 | Window 3 | 42 | 90% | 0.8563 | 0.186401 | 0.093200 | 0.0878 | 0.0559 | 0.151251 |
| TP3 | Window 3 | 42 | 95% | 0.9123 | 0.277731 | 0.138866 | 0.0569 | 0.0308 | 0.151251 |
| TP3 | Window 3 | 73 | 90% | 0.8570 | 0.189426 | 0.094713 | 0.0888 | 0.0542 | 0.149397 |
| TP3 | Window 3 | 73 | 95% | 0.9126 | 0.279175 | 0.139587 | 0.0566 | 0.0309 | 0.149397 |
| Motor_current | Window 1 | 17 | 90% | 0.8415 | 1.045325 | 0.522663 | 0.0806 | 0.0779 | 0.637507 |
| Motor_current | Window 1 | 17 | 95% | 0.9178 | 2.223981 | 1.111991 | 0.0394 | 0.0428 | 0.637507 |
| Motor_current | Window 1 | 42 | 90% | 0.8266 | 1.064125 | 0.532062 | 0.0792 | 0.0942 | 0.640753 |
| Motor_current | Window 1 | 42 | 95% | 0.9157 | 2.280990 | 1.140495 | 0.0364 | 0.0480 | 0.640753 |
| Motor_current | Window 1 | 73 | 90% | 0.8064 | 1.046333 | 0.523166 | 0.0788 | 0.1149 | 0.667094 |
| Motor_current | Window 1 | 73 | 95% | 0.9113 | 2.235783 | 1.117892 | 0.0354 | 0.0533 | 0.667094 |
| Motor_current | Window 2 | 17 | 90% | 0.8828 | 1.870684 | 0.935342 | 0.0840 | 0.0331 | 0.740575 |
| Motor_current | Window 2 | 17 | 95% | 0.9362 | 3.164371 | 1.582185 | 0.0531 | 0.0108 | 0.740575 |
| Motor_current | Window 2 | 42 | 90% | 0.8845 | 1.872736 | 0.936368 | 0.0833 | 0.0322 | 0.729291 |
| Motor_current | Window 2 | 42 | 95% | 0.9352 | 3.163585 | 1.581793 | 0.0538 | 0.0110 | 0.729291 |
| Motor_current | Window 2 | 73 | 90% | 0.8826 | 1.930388 | 0.965194 | 0.0841 | 0.0333 | 0.745745 |
| Motor_current | Window 2 | 73 | 95% | 0.9348 | 3.214315 | 1.607158 | 0.0539 | 0.0113 | 0.745745 |
| Motor_current | Window 3 | 17 | 90% | 0.8745 | 1.205657 | 0.602829 | 0.0675 | 0.0580 | 0.619341 |
| Motor_current | Window 3 | 17 | 95% | 0.9257 | 2.221857 | 1.110928 | 0.0440 | 0.0303 | 0.619341 |
| Motor_current | Window 3 | 42 | 90% | 0.8757 | 1.200349 | 0.600174 | 0.0669 | 0.0573 | 0.628503 |
| Motor_current | Window 3 | 42 | 95% | 0.9277 | 2.261453 | 1.130727 | 0.0425 | 0.0298 | 0.628503 |
| Motor_current | Window 3 | 73 | 90% | 0.8734 | 1.215343 | 0.607671 | 0.0697 | 0.0569 | 0.628714 |
| Motor_current | Window 3 | 73 | 95% | 0.9262 | 2.266952 | 1.133476 | 0.0439 | 0.0299 | 0.628714 |

## Interpretation

- Coverage is the observed fraction of test targets inside the prediction interval.
- Mean width measures interval size in the target's original units.
- Lower miss rate measures observations below the lower bound.
- Upper miss rate measures observations above the upper bound.
- Wider intervals are not automatically better; coverage and width must be considered together.
- Nominal 90% and 95% levels are targets, not guaranteed empirical outcomes on chronological data.

## Important methodological limitations

- Standard split conformal finite-sample marginal coverage relies on exchangeability.
- Chronological sensor observations may be dependent and non-stationary.
- Distribution shifts between calibration and testing can reduce empirical coverage.
- Results are specific to this dataset and its chronological windows.
- Symmetric intervals may not be optimal for asymmetric residual distributions.
- Coverage is marginal and does not guarantee coverage for every operating condition.
- Repeated seeds measure estimator variability, not independent machine generalization.

## Next research direction

Compare fixed calibration intervals with rolling or adaptive calibration under controlled drift.
Evaluate whether coverage deterioration can serve as an impact signal for model monitoring and retraining.