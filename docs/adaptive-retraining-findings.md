# Adaptive Retraining: Experimental Findings

## Policies evaluated

The controlled experiments compare:

1. No retraining
2. Periodic retraining
3. Performance-triggered retraining
4. Impact-aware retraining

## Key observations

### Covariate drift

Performance-triggered and impact-aware policies achieved comparable
prediction errors while requiring fewer retraining events than periodic
retraining in the evaluated experiments.

### Concept drift

Performance-triggered retraining achieved the lowest observed error.
Impact-aware retraining used fewer retraining events, illustrating a
trade-off between predictive performance and retraining burden.

### No drift

Periodic retraining did not provide a meaningful advantage sufficient
to justify its additional retraining activity in the evaluated setup.

## Interpretation

Retraining policies should be compared using both predictive error
and retraining frequency. Lower prediction error alone does not describe
the operational cost of a policy.

These observations are specific to the controlled experimental setup
and should not be presented as universal guarantees.
