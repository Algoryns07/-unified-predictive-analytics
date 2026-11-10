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

## Extended Detection Evaluation

The drift detection evaluation includes metrics designed to assess
performance under class imbalance:

- Accuracy: overall proportion of correct predictions.
- Precision: proportion of detected drift cases that are genuine drift cases.
- Recall: proportion of genuine drift cases successfully detected.
- Specificity: proportion of non-drift cases correctly identified.
- F1-score: harmonic mean of precision and recall.
- Balanced accuracy: average of sensitivity and specificity.
- Matthews correlation coefficient (MCC): a balanced measure
  incorporating all four confusion-matrix outcomes.
- Negative predictive value (NPV): proportion of negative predictions
  that are genuinely negative.
- False discovery rate (FDR): proportion of positive predictions
  that are false discoveries.

No single metric is sufficient for evaluating drift detection.
Interpretation should consider false alarms, missed drift events,
class imbalance, and the operational consequences of retraining.

## Input Validation

The drift detection module validates PSI bin counts and smoothing
parameters, statistical significance levels, and non-negative drift
thresholds. It also checks that the reference and current inputs to
frame-level comparisons are pandas DataFrames.

The retraining policy validates performance scores and policy
thresholds before making a decision. These checks are intended to
prevent invalid configurations from silently affecting results.

## Reproducibility and Evidence Limitations

The experiment runner supports repeated runs with different random
seeds and summarizes the resulting metrics using mean and standard
deviation.

The current experiments use controlled synthetic feature shifts on
the breast cancer dataset. They are useful for testing implementation
behavior under defined scenarios, but they do not establish
real-world drift detection performance.

Further validation should include additional datasets, realistic
temporal evaluation, false-alarm analysis, and comparisons of
retraining policies under operational constraints.

## Random-seed input validation

The multi-seed experiment runner accepts non-negative integer seeds.
It rejects Boolean values, fractional values, negative values, empty
seed collections, and duplicate seeds before starting experiments.
Accepted integer seeds are normalized to standard Python integers
before experiment execution and metadata serialization.

This validation improves experiment configuration reliability.
It does not replace evaluation across additional datasets, realistic
temporal windows, or operational false-alarm conditions.

## Multi-seed reproducibility

The multi-seed experiment runner uses five default random seeds: 11, 42, 73, 101 and 202. Results are aggregated using mean and sample standard deviation. Individual seed results are retained for inspection. Run the experiments using python -m src.run_multiseed from the impact-aware application directory.

