# Unified Predictive Analytics Platform

An interactive machine-learning platform for predictive analytics, industrial monitoring, and data-drift-aware model management. It brings together two independently developed application tracks under one dashboard while keeping their datasets, experiments, and evaluation results separate.

> **[Open the live dashboard](https://impact-aware-ml.streamlit.app/)**

---

## Overview

The platform demonstrates an end-to-end machine-learning workflow, from data input and prediction to monitoring, impact assessment, and model-retraining decisions.

It currently contains two application tracks:

1. **MetroPT Industrial Predictive Analytics** — predictive analytics using industrial sensor data in the MetroPT application.
2. **Impact-Aware Data Drift Monitoring** — drift measurement, impact assessment, and cost-sensitive retraining-policy experiments.

These tracks are integrated through a common Streamlit dashboard. They remain separate at the data and experiment level: unrelated datasets are not merged into a single training dataset, and results are interpreted within the application and dataset that produced them.

## Live Dashboard

### [Launch the Unified Predictive Analytics Platform](https://impact-aware-ml.streamlit.app/)

The dashboard is organized into four main areas:

| Section | Description |
|---|---|
| **Upload & Predict** | Upload supported data and explore model predictions. |
| **Existing Experiments** | Review available experiment outputs and comparison results. |
| **Method & Limitations** | Understand the methods, assumptions, and interpretation boundaries. |
| **Impact-Aware Monitoring** | Explore drift monitoring, impact assessment, and cost-sensitive retraining results. |

The live application may take a short time to wake up if it has been idle.

## Main Capabilities

- **Predictive analytics:** apply machine-learning models to supported industrial or tabular datasets.
- **Data-quality-aware workflow:** organize the process from input data through prediction and evaluation.
- **Drift measurement:** examine changes in feature distributions using statistical and distance-based measures.
- **Impact-aware assessment:** consider the potential importance of detected changes rather than treating every statistical difference as equally consequential.
- **Cost-sensitive retraining:** compare retraining policies using experiment outputs and associated costs.
- **Experiment inspection:** review saved CSV-based results through the dashboard.
- **Interactive exploration:** use a browser-based Streamlit interface without running the dashboard locally.

## High-Level Workflow

```text
Data Input
    |
    v
Data Validation and Preparation
    |
    v
Prediction
    |
    v
Drift Measurement
    |
    v
Impact Assessment
    |
    v
Cost-Sensitive Retraining Decision
    |
    v
Evaluation and Experiment Review
```

This is a conceptual workflow. The exact steps and available outputs depend on the selected application and experiment.

## Technology Stack

The project is built primarily with:

- **Python** — application and machine-learning logic
- **Streamlit** — interactive dashboard
- **pandas** and **NumPy** — data handling
- **scikit-learn** — machine-learning workflows
- **SciPy** and **statsmodels** — statistical analysis
- **Matplotlib**, **Seaborn**, and **Plotly** — visualisation
- **XGBoost** — supported gradient-boosting workflows
- **pytest** — automated tests

See [`requirements.txt`](requirements.txt) for the repository dependency list.

## Repository Structure

The main application entry point is:

```text
applications/
├── metropt/
│   └── dashboard/
│       └── app.py          # Main Streamlit dashboard entry point
└── impact_aware/
    └── outputs/            # Impact-Aware experiment result files
```

The repository also contains supporting source code, configuration, documentation, tests, and experiment outputs for the individual application tracks.

> This is a focused view of the key application locations, not a complete listing of every repository file.

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Algoryns07/-unified-predictive-analytics.git
cd -unified-predictive-analytics
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the dashboard

```bash
streamlit run applications/metropt/dashboard/app.py
```

Streamlit will print a local address in the terminal, usually `http://localhost:8501`.

## Deployment

The dashboard is published using **Streamlit Community Cloud**.

- **Live dashboard:** [impact-aware-ml.streamlit.app](https://impact-aware-ml.streamlit.app/)
- **GitHub repository:** [Algoryns07 / -unified-predictive-analytics](https://github.com/Algoryns07/-unified-predictive-analytics)
- **Deployment branch:** `cloud-preview`
- **Main file:** `applications/metropt/dashboard/app.py`

## Experimental Interpretation and Limitations

Interpret results in the context of the dataset, experimental setup, and assumptions used to generate them.

- A controlled or simulated drift scenario is useful for testing system behaviour, but it is **not evidence of naturally occurring real-world drift**.
- A drift alert does not automatically mean model performance has deteriorated or that retraining is necessary.
- False-positive alerts and no-shift control scenarios must be considered when evaluating a detector.
- Retraining decisions should be assessed against an explicit policy, evaluation protocol, and cost assumptions.
- Results from separate datasets and application tracks should not be compared as though they came from one shared training dataset.
- Dashboard outputs are intended for analysis and demonstration; they should not be treated as operational maintenance or deployment decisions without further validation.

## Reproducibility

For meaningful comparisons, record the dataset, preprocessing choices, model and detector settings, random seed, evaluation period, and retraining-policy assumptions associated with each experiment. Where multiple random seeds or datasets are available, report variability rather than relying on a single run.

## Usage

This repository is maintained as an academic machine-learning project. Please contact the repository owner before reusing, redistributing, or adapting the code.

## License

No open-source license has been provided with this repository. Unless the repository owner adds a license, the code is not granted open-source reuse permissions by default.

---

**Quick links:** [Live Dashboard](https://impact-aware-ml.streamlit.app/) · [GitHub Repository](https://github.com/Algoryns07/-unified-predictive-analytics)
