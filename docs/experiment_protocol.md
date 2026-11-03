# Experiment protocol

## Required scenarios
- No-shift control.
- Shift in one feature.
- Shift in multiple features.
- Several shift magnitudes.
- Repeated random seeds.

## Required policy baselines
1. Always retrain.
2. Retrain whenever drift is detected.
3. Retrain when measured performance falls below a threshold.
4. Proposed impact-aware, cost-sensitive rule.

## Metrics
Predictive: accuracy, precision, recall, F1, ROC-AUC.  
Drift: number of detected windows, false alarms on no-shift controls, drifted feature count.  
Operational: number of retraining events and actual training time.

## Fairness of comparison
Use the same data windows, initial model and information availability for every policy. Freeze thresholds before final test experiments. Record unsuccessful experiments and avoid reporting only favourable results.

## Comparative drift-metric experiment

An initial controlled experiment compares PSI, the Kolmogorov-Smirnov
(KS) test, and Wasserstein distance under five synthetic distribution
scenarios.

### Experimental setup

- Random seed: 42.
- Reference sample size: 2,000.
- Current sample size per scenario: 2,000.
- Reference distribution: Normal distribution with mean 0 and standard deviation 1.
- Current distributions: Normal distributions with means 0, 0.2, 0.5, 1.0, and 2.0, each with standard deviation 1.
- Metrics: PSI, KS statistic, KS p-value, Wasserstein distance, and normalized Wasserstein distance.

### Initial observations

The experiment shows increasing PSI and Wasserstein distance as the
magnitude of the distribution shift increases. The KS test also
identifies statistically significant differences for the tested
small, moderate, large, and extreme shifts.

The no-shift scenario does not produce a statistically significant
KS result at the 0.05 significance level.

### Limitations

This is an initial diagnostic experiment using one random seed and
synthetic normally distributed data. It is not a multi-seed benchmark
or evidence that one drift metric is universally superior.

Further experiments should evaluate multiple random seeds,
distribution shapes, sample sizes, and real-world datasets.
Threshold selection and false-alarm behaviour also require further
validation.

The experiment output is stored in
`outputs/drift_metric_comparison.csv`.
