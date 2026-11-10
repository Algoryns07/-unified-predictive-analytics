# Drift Detection: Interpretation and Limitations

## Monitoring methods

The project evaluates statistical feature drift using:

- Population Stability Index (PSI)
- Kolmogorov-Smirnov (KS) testing
- Wasserstein distance
- Multiple-testing correction for statistical comparisons

## Interpretation

A statistical drift alert indicates a change in the observed feature
distribution. It does not independently establish that predictive
performance has deteriorated.

Feature drift and predictive-performance deterioration should therefore
be monitored and interpreted separately.

## Evaluation considerations

- Examine false alarms under no-drift conditions.
- Evaluate detection under controlled covariate and concept drift.
- Consider multiple-testing effects.
- Report alert rates separately from detection accuracy.
- Avoid claiming real-world detection accuracy without verified labels.

## Limitation

Real-world drift windows without ground-truth labels cannot establish
whether every alert represents a genuine operational problem.

## Interpreting Drift Detection Results

A drift detection alert indicates a change in feature distributions.
It does not automatically establish that predictive performance
has deteriorated.

### Important evaluation considerations

- Monitor false-positive alerts on no-shift control data.
- Evaluate missed detections under intentional feature shifts.
- Examine predictive performance alongside statistical drift metrics.
- Consider the operational cost of unnecessary retraining.
- Validate detection thresholds before final evaluation.

### Research limitation

Results from controlled synthetic feature shifts should not be
treated as evidence of real-world detection performance.
Additional validation on independent datasets and realistic
monitoring windows is necessary.

