# Intelligent Predictive Analytics Platform

A reusable machine-learning platform for predictive analytics, uncertainty estimation, data drift monitoring, and cost-sensitive adaptive model retraining.

The current application uses the MetroPT industrial sensor dataset.

## Project objectives

- Industrial sensor-based predictive regression.
- Chronological model evaluation and temporal robustness.
- Residual analysis and prediction uncertainty estimation.
- Statistical feature drift and performance monitoring.
- Cost-sensitive adaptive model retraining.
- Controlled experiments under synthetic drift.

## Dataset

MetroPT — A Benchmark Dataset for Predictive Maintenance.

- Official dataset: https://zenodo.org/records/6854240
- Research paper: https://www.nature.com/articles/s41597-022-01877-3

Place dataset_train.csv inside data/raw/.

The raw dataset is excluded from Git because of its size. See docs/DATASET.md.

## Methodology

1. Data quality assessment and sensor investigation.
2. Predictive regression using lag and rolling-window features.
3. Chronological validation and temporal robustness experiments.
4. Residual analysis and split conformal prediction.
5. Adaptive prediction-interval calibration.
6. Feature drift and prediction-performance monitoring.
7. Comparison of adaptive retraining policies.
8. Controlled experiments involving no drift, covariate drift and concept drift.

## Representative results

### Baseline regression

| Target | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Oil temperature | 0.5486 | 1.0512 | 0.9532 |
| TP3 | 0.0621 | 0.1467 | 0.9425 |
| Motor current | 0.2465 | 0.6112 | 0.9123 |

### Temporal robustness

| Target | RMSE reduction versus persistence |
|---|---:|
| Oil temperature | 64.54% |
| TP3 | 63.76% |
| Motor current | 61.50% |

### Adaptive retraining

Four policies are compared:
- P1: No retraining.
- P2: Periodic retraining.
- P3: Performance-triggered retraining.
- P4: Impact-aware challenger-based retraining.

Results indicate that no single policy is universally best across all tested drift scenarios.

## Repository structure

- dashboard/: Interactive results dashboard.
- data/: Dataset storage.
- docs/: Dataset and experiment documentation.
- models/: Generated model files.
- notebooks/: Exploratory notebooks.
- outputs/: Generated experiment results.
- src/: Implementation and experiment scripts.
- tests/: Automated tests.

## Installation

Create a virtual environment and install dependencies:

    py -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt

## Run tests

    python -m pytest -v

## Dashboard

    streamlit run dashboard/app.py

## Limitations

- MetroPT does not provide a verified ground-truth failure label for this project.
- Feature drift does not necessarily mean predictive performance has deteriorated.
- Synthetic drift experiments are separate from real MetroPT experiments.
- Temporal dependence can affect statistical tests and conformal prediction coverage.
- Results should be interpreted within the tested experimental settings.

See docs/ for detailed experimental reports.
