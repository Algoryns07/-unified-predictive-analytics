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
