# Impact-Aware Data Drift Assessment and Predictive Analytics Platform

An interactive machine learning platform for predictive analytics, data drift monitoring, impact assessment, and cost-sensitive model retraining.

## Live Dashboard

**Open the live application:** https://impact-aware-ml.streamlit.app/

**GitHub repository:** https://github.com/Algoryns07/-unified-predictive-analytics

---

## 1. Project Overview

Machine learning models may become less reliable when incoming data changes over time. Monitoring these changes and deciding when model retraining is necessary are important challenges in machine learning systems.

This project provides an interactive platform to explore predictive analytics, data drift measurement, impact assessment, and cost-sensitive retraining decisions.

The platform integrates two application components:

1. Impact-Aware Data Drift Monitoring.
2. MetroPT Industrial Predictive Analytics

Both components are accessible through a unified dashboard. Their datasets and experimental results remain separate.

## 2. Main Features

- Upload supported datasets and explore predictions.
- Perform regression-based predictive analytics.
- Measure changes in incoming data distributions.
- Evaluate feature-level data drift.
- Assess the potential impact of detected changes.
- Compare cost-sensitive model retraining policies.
- Review saved experimental results.
- Explore results through an interactive dashboard.

## 3. Machine Learning Methods

### Linear Regression

Linear Regression is used in the regression-based predictive analytics workflow. It estimates continuous target values using input features and provides an interpretable baseline for understanding relationships between variables.

### Data Drift Detection

The monitoring component uses statistical and distance-based measurements to identify changes between reference data and incoming data.

### Impact-Aware Assessment

The platform considers the potential significance of detected changes rather than treating every statistical difference as equally important.

### Cost-Sensitive Retraining

Different retraining policies are evaluated by considering potential model benefits and associated retraining costs.

## 4. Dashboard Sections

| Section | Description |
| --- | --- |
| Upload and Predict | Upload supported data and explore predictions. |
| Existing Experiments | Review saved experiment outputs and comparisons. |
| Method and Limitations | Understand methods, assumptions, and limitations. |
| Impact-Aware Monitoring | Explore drift measurements and retraining analysis. |

## 5. Project Workflow

1. Data input and validation.
2. Data preparation.
3. Model prediction.
4. Data drift measurement.
5. Impact assessment.
6. Cost-sensitive retraining decision.
7. Evaluation and experiment review.

## 6. Technology Stack

- Python
- Streamlit
- NumPy
- pandas
- scikit-learn
- SciPy
- statsmodels
- Matplotlib
- Seaborn
- Plotly
- XGBoost
- pytest

Refer to `requirements.txt` for the dependency list.

## 7. Repository Structure

The main dashboard entry point is:

`applications/metropt/dashboard/app.py`

The repository also contains application source code, configuration files, documentation, automated tests, and experiment outputs for the individual application components.

## 8. Run Locally

### Requirements

- Python 3.12
- Git

### Clone the repository

```bash
git clone https://github.com/Algoryns07/-unified-predictive-analytics.git
cd -unified-predictive-analytics
```

### Create and activate a virtual environment

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Run the dashboard

```bash
streamlit run applications/metropt/dashboard/app.py
```

Open the local address provided by Streamlit, usually `http://localhost:8501`.

## 9. Deployment

The application is hosted using Streamlit Community Cloud.

- Dashboard: https://impact-aware-ml.streamlit.app/
- Repository: https://github.com/Algoryns07/-unified-predictive-analytics
- Deployment branch: `cloud-preview`
- Application entry point: `applications/metropt/dashboard/app.py`

## 10. Experimental Limitations

- Simulated drift experiments are controlled tests and should not be presented as evidence of naturally occurring real-world drift.
- A drift alert does not automatically mean that model performance has deteriorated.
- False-positive alerts and no-shift control scenarios must be considered.
- Retraining decisions depend on the evaluation protocol and cost assumptions.
- Results from separate datasets should be interpreted independently.
- Further validation is required before using experimental results for operational decisions.

## 11. Reproducibility

For meaningful comparisons, record the dataset, preprocessing steps, model settings, detector settings, random seed, evaluation period, and retraining-policy assumptions.

## 12. License and Usage

This repository is maintained as an academic machine learning project.

No open-source license has been provided. Reuse and redistribution are not granted by default.

---

## Quick Links

- Live Dashboard: https://impact-aware-ml.streamlit.app/
- GitHub Repository: https://github.com/Algoryns07/-unified-predictive-analytics
