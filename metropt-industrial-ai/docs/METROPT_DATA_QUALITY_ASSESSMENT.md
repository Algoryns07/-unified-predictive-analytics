# MetroPT Industrial Dataset: Data Quality Assessment

## 1. Purpose

This document consolidates the initial data-quality investigations
performed on the 2022 Zenodo MetroPT industrial air-compressor dataset.

The purpose is to identify data characteristics, potential measurement
issues, temporal irregularities and modelling constraints before
developing predictive analytics applications.

The assessment is exploratory. Observed anomalies are not automatically
treated as sensor failures or confirmed equipment faults.

## 2. Dataset overview

| Property | Finding |
|---|---|
| Dataset | MetroPT industrial air-compressor dataset |
| Source | Zenodo, 2022 |
| Observations | 10,773,588 |
| Columns | 21 |
| Timestamp range | 2022-01-01 06:00:00 to 2022-06-02 15:49:53 |
| Sampling | Predominantly one-second intervals |
| Missing values | None detected in initial full-dataset scan |
| Invalid timestamps | None detected |
| Invalid numeric values | None detected |
| Constant feature | Pressure_switch |

Source: https://zenodo.org/records/6854240

Scientific Data publication DOI: 10.1038/s41597-022-01877-3

## 3. Timestamp and sampling assessment

The initial full-dataset scan found:

- No adjacent duplicate timestamps.
- No adjacent out-of-order timestamps.
- 10,773,435 one-second intervals.
- 147 intervals of approximately four hours.
- Additional irregular multi-hour and multi-day gaps.

A recurring four-hour recording gap appears between approximately
02:00 and 06:00.

The existence of these gaps must be considered during chronological
validation and lag-feature construction.

Recommendations:

1. Preserve timestamps.
2. Use chronological rather than random train-test splitting.
3. Do not interpolate across long recording gaps.
4. Prevent lag windows from crossing substantial timestamp gaps.
5. Document the chosen resampling frequency.

## 4. Feature-level observations

### 4.1 Constant feature

Pressure_switch was zero across all 10,773,588 observations.

It contains no observed variation in this dataset and should be excluded
from predictive modelling unless a specific analytical justification
emerges.

### 4.2 Rare LPS state

LPS values:

| Value | Observations |
|---|---:|
| 0 | 10,705,915 |
| 1 | 67,673 |

The value 1 represents approximately 0.628% of observations.

This is a rare operating-state signal, not a verified equipment-failure
label. It must not be interpreted as a failure target without additional
evidence.

### 4.3 COMP and DV_eletric

The two signals are nearly complementary:

| Signal combination | Observations |
|---|---:|
| COMP = 1, DV_eletric = 0 | 9,371,865 |
| COMP = 0, DV_eletric = 1 | 1,401,723 |
| COMP = 1, DV_eletric = 1 | 68 |

The 68 simultaneous active observations were investigated separately.

## 5. Signal transition investigation

A total of 68 observations were identified where:

COMP = 1 AND DV_eletric = 1

All 68 observations had immediate one-second neighbours.

The observed transitions were:

| Transition | Count |
|---|---:|
| 1/0 -> 1/1 -> 0/1 | 37 |
| 0/1 -> 1/1 -> 1/0 | 31 |
| Total | 68 |

The simultaneous state appears as a brief intermediate observation
between the two dominant operating-state combinations.

This pattern is consistent with a possible transition overlap,
timing difference or measurement inconsistency. The analysis does
not establish the underlying cause.

Recommendations:

- Preserve all original observations.
- Flag these 68 records for further analysis.
- Do not automatically delete them.
- Do not label them as equipment failures.
- Consider transition indicators during feature engineering.

## 6. GPS speed quality investigation

### 6.1 Scope

The investigation examined observations satisfying:

gpsQuality = 1 AND gpsSpeed > 100 km/h

Results:

| Measure | Result |
|---|---:|
| High-speed observations | 75 |
| High-speed episodes | 38 |
| Valid coordinate comparisons | 62 |
| No valid comparison | 13 |
| One valid comparison | 32 |
| Two valid comparisons | 30 |

The coordinate-derived estimates use adjacent GPS positions and
timestamps, with a maximum adjacent time gap of five seconds.

### 6.2 Agreement analysis

| Absolute difference | Observations within tolerance |
|---|---:|
| 10 km/h | 9 of 62 |
| 25 km/h | 20 of 62 |
| 50 km/h | 30 of 62 |

Pearson correlation between reported speed and the available
coordinate-derived mean was -0.0451 for the investigated subset.

### 6.3 Interpretation and limitations

The reported high speeds are not independently validated by the
coordinate-derived estimates.

However, coordinate-derived speed is not ground truth. GPS positional
jumps, sampling effects and measurement noise may affect both measures.

The negative near-zero correlation applies only to this selected
high-speed subset. It should not be generalized to the complete dataset.

The analysis does not establish whether the observations represent
genuine movement, positional noise or measurement errors.

### 6.4 Recommendations

1. Preserve original GPS fields.
2. Retain gpsQuality as a separate indicator.
3. Flag suspicious speed and positional transitions.
4. Avoid automatically deleting observations based on a speed threshold.
5. Do not treat coordinate-derived speed as ground truth.
6. Evaluate GPS feature usefulness before including it in predictive models.

## 7. Modelling implications

The initial assessment suggests the following precautions:

| Issue | Modelling implication |
|---|---|
| Four-hour and irregular gaps | Time-aware splitting and gap-aware lag construction |
| Constant Pressure_switch | Exclude from predictive features |
| Rare LPS state | Avoid treating as a failure label |
| COMP/DV_eletric overlaps | Preserve and flag transition observations |
| GPS speed inconsistencies | Validate before using GPS features |
| High-frequency observations | Consider justified resampling for computational efficiency |

Potential continuous prediction targets include oil temperature,
TP3 and Reservoirs. The final target must be selected after examining
its temporal behaviour, predictability and operational relevance.

Features must only use information available at prediction time.
Lag construction and validation must avoid temporal leakage.

## 8. Limitations

This assessment is based on exploratory data-quality analysis.

It does not establish:

- A verified catastrophic failure label.
- The physical cause of GPS inconsistencies.
- The physical cause of COMP/DV_eletric overlaps.
- General predictive performance.
- Operational benefits of a particular retraining policy.

These claims require separate experiments and supporting evidence.

## 9. Conclusion

The MetroPT dataset provides a substantial high-frequency industrial
sensor record suitable for further predictive analytics investigation.

The initial assessment identifies temporal gaps, a constant feature,
rare operating-state observations, brief signal-transition overlaps
and GPS consistency concerns.

The findings provide a documented basis for cautious preprocessing,
feature selection, chronological validation and subsequent model
development.

The original dataset should remain unchanged. All cleaning,
transformation and feature-engineering decisions should be performed
in reproducible processing pipelines.
