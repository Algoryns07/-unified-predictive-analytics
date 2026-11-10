# Impact-Aware Data Drift Assessment & Predictive Analytics Platform

An interactive machine-learning platform for monitoring data distribution changes, assessing their potential impact, and supporting cost-sensitive model-retraining decisions. The platform also includes a separate MetroPT industrial predictive-analytics application.

<p align="center">
  <a href="https://impact-aware-ml.streamlit.app/">
    <strong>🚀 OPEN THE LIVE DASHBOARD</strong>
  </a>
</p>

**Live application:** https://impact-aware-ml.streamlit.app/  
**Source code:** https://github.com/Algoryns07/-unified-predictive-analytics

---

## 1. Project Overview

Machine-learning models can become less reliable when incoming data changes over time. Detecting a statistical change is not, by itself, enough to decide whether a model should be retrained. Monitoring systems should also consider the possible impact of the change and the cost of taking action.

This project provides an interactive platform for exploring that workflow:

1. Prepare and validate incoming data.
2. Generate predictions using supported machine-learning workflows.
3. measure changes between reference and incoming data.
4. Assess detected changes in an impact-aware context.
5. Compare cost-sensitive retraining policies.
6. Inspect experiment outputs and evaluation results.

The repository brings together two application tracks in one dashboard. Their datasets, experiments, and results remain separate; unrelated datasets are not combined into a single training dataset.

## 2. Live Dashboard

### [🚀 Launch Impact-Aware Data Drift Assessment Dashboard](https://impact-aware-ml.streamlit.app/)

The deployed Streamlit application provides four main sections:

| Dashboard section | Purpose |
|---|---|
| **Upload & Predict** | Upload supported data and explore model predictions. |
| **Existing Experiments** | View available experiment outputs and comparison results. |
| **Method & Limitations** | Review the methods, assumptions, and limitations. |
| **Impact-Aware Monitoring** | Explore drift measurements, impact assessment, and cost-sensitive retraining results. |

If the application has been idle, allow a short time for the hosted service to start.

## 3. Application Components

### A. Impact-Aware Data Drift Monitoring

This component explores how incoming data differs from a reference dataset and how monitoring results can inform retraining decisions.

Key areas include:
- Statistical and distance-based feature-drift measurements.
- Impact-aware interpretation of detected changes.
- No-shift control and false-positive considerations.
- Comparison of cost-sensitive retraining policies.
- Experiment-result inspection.

### B. MetroPT Industrial Predictive Analytics

This component provides a separate industrial predictive-analytics workflow using the MetroPT application data and its associated experiments.

The two application tracks are presented through the same dashboard, but their datasets and experimental results are not merged.

## 4. Conceptual Workflow

```text
Reference Data + Incoming Data
              |
              v
      Data Preparation
              |
              v
       Model Prediction
              |
              v
       Drift Measurement
              |
              v
       Impact Assessment
              |
              v
   Cost-Sensitive Decision
              |
              v
 Retraining Policy Evaluation
              |
              v
       Results Review
```

This diagram represents the overall monitoring concept. The exact operations and outputs depend on the selected application and experiment.

## 5. Technology Stack

- **Python** — application and machine-learning implementation
- **Streamlit** — interactive web dashboard
- **pandas, NumPy** — data processing
- **scikit-learn** — machine-learning workflows
- **SciPy, statsmodels** — statistical analysis
- **Matplotlib, Seaborn, Plotly** — visualisation
- **XGBoost** — supported gradient-boosting workflows
- **pytest** — automated testing

The full dependency list is maintained in [`requirements.txt`](requirements.txt).

## 6. Repository Structure

```text
applications/
├── metropt/
│   └── dashboard/
│       └── app.py          # Main Streamlit dashboard entry point
└── impact_aware/
    └── outputs/            # Impact-Aware experiment result files
```

The repository also includes application source code, configuration, documentation, tests, and experiment outputs. The structure above highlights the principal application locations rather than listing every file.

## 7. Run the Project Locally

### Requirements

- Python 3.12
- Git
- A compatible environment for the packages listed in `requirements.txt`

### Step 1 — Clone the repository

```bash
git clone https://github.com/Algoryns07/-unified-predictive-analytics.git
cd -unified-predictive-analytics
```

### Step 2 — Create and activate a virtual environment

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

### Step 3 — Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4 — Launch the dashboard

```bash
streamlit run applications/metropt/dashboard/app.py
```

Open the local address printed by Streamlit, normally `http://localhost:8501`.

## 8. Deployment

The dashboard is hosted on **Streamlit Community Cloud**.

- **Dashboard URL:** [https://impact-aware-ml.streamlit.app/](https://impact-aware-ml.streamlit.app/)
- **GitHub repository:** [https://github.com/Algoryns07/-unified-predictive-analytics](https://github.com/Algoryns07/-unified-predictive-analytics)
- **Deployment branch:** `cloud-preview`
- **Application entry point:** `applications/metropt/dashboard/app.py`

## 9. Experimental Considerations and Limitations

Results must be interpreted in the context of the dataset, experimental setup, and assumptions used.

- Controlled or simulated drift is useful for testing system behaviour, but it is not evidence of naturally occurring real-world drift.
- A drift alert does not automatically establish that predictive performance has deteriorated or that retraining is required.
- False-positive alerts and no-shift control scenarios are important when evaluating drift detectors.
- Retraining policies should be assessed using explicit evaluation protocols and cost assumptions.
- Results from separate datasets and application tracks should not be treated as though they came from one shared training dataset.
- The dashboard is an academic and experimental platform; its outputs should not be used as operational decisions without further validation.

## 10. Reproducibility

For reproducible comparisons, record the dataset, preprocessing steps, model and detector settings, random seed, evaluation period, and retraining-policy assumptions. Where repeated runs are available, report variability rather than relying on a single result.

## 11. Usage and License

This repository is maintained as an academic machine-learning project. No open-source license has been provided. Unless a license is added by the repository owner, the code is not granted open-source reuse permissions by default. Please contact the repository owner before reuse or redistribution.

---

### Quick Access

- **[🚀 Open the Live Dashboard](https://impact-aware-ml.streamlit.app/)**
- **[View the GitHub Repository](https://github.com/Algoryns07/-unified-predictive-analytics)**
