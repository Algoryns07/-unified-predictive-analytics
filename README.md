# Unified Predictive Analytics Platform

An extensible machine-learning platform for predictive analytics,
prediction uncertainty, data-quality assessment, statistical drift
monitoring and impact-aware cost-sensitive model retraining.

## Applications

- MetroPT: primary industrial predictive analytics application.
- Appliances Energy Prediction: planned chronological regression benchmark.
- NASA MERRA-2: planned advanced environmental-data application.

Each application will be trained, validated and evaluated independently.
Unrelated datasets will never be combined into a single training dataset.

## Core Capabilities

1. Data loading and quality validation.
2. Feature engineering.
3. Regression model comparison.
4. Chronological model evaluation.
5. Prediction uncertainty and conformal calibration.
6. Statistical data drift monitoring.
7. Predictive performance monitoring.
8. Impact-aware adaptive retraining.
9. Controlled experiments and multi-seed evaluation.
10. Interactive Streamlit dashboard.

## Methodological Principles

- Use time-aware validation for time-dependent datasets.
- Distinguish feature drift from predictive performance deterioration.
- Separate controlled synthetic experiments from real observations.
- Require labelled observations to verify predictive performance.
- Evaluate challenger models before approving model replacement.
- Report assumptions, limitations and uncertainty transparently.

## Installation

Installation instructions will be finalised after dependency compatibility
and Python-version checks.

## Run Tests

    python -m pytest -v

## Launch Dashboard

    streamlit run dashboard/app.py

## Project Status

Initial repository structure created. Integration and validation are pending.

## Data

Raw datasets and large generated artifacts are not tracked in Git.
See `data/README.md` for dataset placement instructions.
