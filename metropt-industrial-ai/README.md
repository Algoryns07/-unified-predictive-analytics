# Intelligent Predictive Analytics with Data Drift Monitoring and Cost-Sensitive Adaptive Model Retraining

### An Interactive Machine Learning Framework for Industrial Predictive Analytics, Prediction Uncertainty and Model Reliability

## Overview

Industrial systems continuously generate sensor measurements. Predictive machine-learning models trained on historical observations may become less reliable when incoming data changes.

This project develops an interactive predictive analytics application using the MetroPT industrial sensor dataset. Its central focus is statistical data drift monitoring and impact-aware, cost-sensitive model retraining.

The application connects predictive regression, uncertainty estimation, data-quality assessment, statistical drift measurements and adaptive retraining policies in one workflow.

## Project Objectives

- Develop machine-learning models for industrial sensor-based predictive regression.
- Evaluate predictive performance using chronological validation.
- Estimate prediction uncertainty and examine prediction-interval calibration.
- Monitor incoming data for statistical feature drift.
- Investigate drift using multiple statistical measurements.
- Distinguish feature drift from actual predictive performance degradation.
- Evaluate adaptive retraining policies under controlled drift scenarios.
- Support retraining decisions using model impact and estimated retraining costs.

## Dataset

**MetroPT — A Benchmark Dataset for Predictive Maintenance**

- Official dataset: https://zenodo.org/records/6854240
- Research paper: https://www.nature.com/articles/s41597-022-01877-3

Place `dataset_train.csv` inside `data/raw/`.

The raw dataset is excluded from Git because of its size. Refer to `docs/DATASET.md` for dataset preparation details.

## System Methodology

1. **Data quality assessment:** Examine missing values, duplicates, sensor characteristics and data quality.
2. **Predictive modelling:** Build regression models using sensor measurements and engineered temporal features.
3. **Chronological evaluation:** Assess predictive performance under temporal validation.
4. **Uncertainty estimation:** Apply residual analysis, split conformal prediction and adaptive prediction-interval calibration.
5. **Statistical data drift monitoring:** Investigate changes using Population Stability Index (PSI), Kolmogorov–Smirnov (KS) statistics and Wasserstein distance.
6. **Impact-aware drift assessment:** Examine drift evidence alongside prediction-performance changes instead of treating every detected shift as a reason to retrain.
7. **Cost-sensitive retraining:** Compare alternative retraining policies and their performance and estimated costs.
8. **Controlled experiments:** Investigate no-drift, covariate-drift and concept-drift scenarios.

## Data Drift Monitoring

Data drift monitoring is a central component of this project.

The application evaluates changes in incoming feature distributions using multiple statistical measurements:

- **Population Stability Index (PSI):** Measures distributional differences between reference and incoming observations.
- **Kolmogorov–Smirnov statistic:** Assesses differences between numeric feature distributions.
- **Wasserstein distance:** Measures the distance between probability distributions.
- **Multiple-testing correction:** Adjusts statistical significance assessments when multiple features are examined.

Drift measurements are interpreted alongside data quality and predictive performance. A detected feature shift does not automatically establish model degradation.

## Impact-Aware Adaptive Retraining

The project compares four retraining policies:

| Policy | Description |
|---|---|
| P1 | No retraining |
| P2 | Periodic retraining |
| P3 | Performance-triggered retraining |
| P4 | Impact-aware challenger-based retraining |

The objective is to investigate when retraining is justified, considering predictive performance and estimated retraining costs.

Retraining recommendations should be supported by labelled incoming observations and measured model-performance degradation. Drift alone is not sufficient evidence.

## Representative Experimental Results

### Baseline Regression

| Target | MAE | RMSE | R² |
|---|---:|---:|---:|
| Oil temperature | 0.5486 | 1.0512 | 0.9532 |
| TP3 | 0.0621 | 0.1467 | 0.9425 |
| Motor current | 0.2465 | 0.6112 | 0.9123 |

### Temporal Robustness

| Target | RMSE reduction versus persistence |
|---|---:|
| Oil temperature | 64.54% |
| TP3 | 63.76% |
| Motor current | 61.50% |

### Adaptive Retraining

Four policies are compared under the tested experimental scenarios. Results indicate that no single policy is universally best across all tested drift scenarios.

These results should be interpreted within their respective experimental settings.

## Interactive Dashboard

The Streamlit application provides:

- Dataset upload and profiling.
- Predictive modelling and prediction results.
- Prediction uncertainty analysis.
- Feature drift assessment.
- Whole-dataset drift investigation.
- Impact-aware retraining recommendations.
- Estimated retraining cost comparison.
- Previously generated experiment results.
- Methodology and limitations.

## Repository Structure

- `dashboard/` — interactive Streamlit application.
- `data/` — dataset storage.
- `docs/` — dataset and experimental reports.
- `models/` — generated model files.
- `notebooks/` — exploratory notebooks.
- `outputs/` — generated experiment results.
- `src/` — implementation and experiment scripts.
- `tests/` — automated tests.

## Installation

Create and activate a virtual environment:

    py -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt

## Run Tests

    python -m pytest -v

## Launch Dashboard

    streamlit run dashboard/app.py

## Limitations

- MetroPT does not provide a verified ground-truth failure label for this project.
- Feature drift does not necessarily mean predictive performance has deteriorated.
- Synthetic drift experiments are separate from real MetroPT experiments.
- Temporal dependence can affect statistical tests and conformal prediction coverage.
- Retraining decisions require appropriate labelled observations for performance evaluation.
- Results should be interpreted within the tested experimental settings.

## Conclusion

This project brings together industrial predictive analytics, statistical data drift monitoring, prediction uncertainty and cost-sensitive adaptive model retraining.

Its central objective is to move beyond simply detecting changes in incoming data and investigate how those changes, their predictive impact and retraining costs can inform model-management decisions.

See `docs/` for detailed dataset, methodology and experimental reports.
