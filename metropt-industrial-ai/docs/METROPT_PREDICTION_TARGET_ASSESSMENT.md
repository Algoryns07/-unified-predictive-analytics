# MetroPT Prediction Target Assessment

## Experimental setup

- Minute-level observations: 179,709
- Aggregation: one-minute sensor means.
- Forecast horizon: next consecutive one-minute interval.
- Baseline: persistence prediction.

## Results

| target | valid_consecutive_pairs | persistence_MAE | persistence_RMSE | persistence_R2 | temporal_correlation |
| --- | --- | --- | --- | --- | --- |
| Reservoirs | 179556 | 0.00103 | 0.00220 | 0.99941 | 0.99970 |
| Oil_temperature | 179556 | 1.35101 | 2.11781 | 0.84216 | 0.92108 |
| TP3 | 179556 | 0.23965 | 0.37935 | 0.69542 | 0.84772 |
| Motor_current | 179556 | 0.68555 | 1.40258 | 0.54630 | 0.77314 |
| H1 | 179556 | 1.22106 | 2.61318 | 0.10426 | 0.55217 |
| Flowmeter | 179556 | 1.39710 | 3.24447 | 0.00968 | 0.50486 |
| TP2 | 179556 | 1.19438 | 2.75689 | 0.00931 | 0.50466 |
| DV_pressure | 179556 | 0.00894 | 0.03345 | -0.87317 | 0.06375 |

## Interpretation and limitations

- Persistence is a simple baseline, not a trained predictive model.
- High persistence R? does not independently establish a useful target.
- Short-lived events may be hidden by minute aggregation.
- Long recording gaps are excluded from consecutive-minute comparisons.
- Final target selection requires chronological ML validation.
