# Research log

## 2026-10-31 — Course-mapped repository baseline
- Mapped implementation modules to all six supplied CSE570 units.
- Kept Random Forest in classification, matching Unit IV.
- Added a separate Unit III regression comparison.
- Added a binary classification drift-monitoring prototype.
- Limitation: simulated drift only; real-world benchmark experiments remain to be conducted.

## Entry template
### YYYY-MM-DD — Experiment name
- Question:
- Dataset/version:
- Split and seed:
- Model and configuration:
- Results:
- Interpretation:
- Limitations / next action:

## 2026-11-07 — Evaluation, Validation and Reproducibility Improvements

### Changes
- Extended binary drift evaluation with balanced accuracy, MCC,
  negative predictive value, and false discovery rate.
- Added validation for cost-sensitive retraining policy inputs.
- Added validation for PSI parameters and frame-level drift thresholds.
- Expanded edge-case and invalid-input test coverage.
- Retained the existing PSI, KS, Wasserstein, and Benjamini-Hochberg
  false discovery rate correction workflow.

### Verification
The complete automated test suite passed with 52 tests after these
changes.

### Interpretation and limitations
The additional metrics improve the breadth of evaluation, particularly
when drift scenarios are imbalanced. Parameter validation improves
reliability of the implementation.

These software and testing improvements do not, by themselves,
demonstrate improved real-world drift detection. Existing experiments
remain controlled synthetic-shift experiments on the breast cancer
dataset. Evaluation on additional datasets and realistic temporal
conditions remains future work.
