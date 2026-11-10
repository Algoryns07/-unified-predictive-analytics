# Unified Predictive Analytics Platform

An integrated machine-learning platform for predictive analytics, prediction uncertainty, data quality, statistical drift monitoring, and impact-aware cost-sensitive model retraining.

## Applications

- MetroPT: Industrial predictive analytics and monitoring.
- Impact-Aware Monitoring: Drift detection and cost-sensitive retraining.
- Appliances Energy Prediction: Planned chronological regression application.
- NASA MERRA-2: Planned environmental-data application.

Each dataset is developed and evaluated independently.

## Core Capabilities

- Data loading and quality validation
- Feature engineering and predictive modelling
- Regression evaluation
- Chronological validation
- Prediction uncertainty and conformal calibration
- Statistical feature-drift monitoring
- Predictive performance monitoring
- Impact-aware retraining decisions
- Controlled multi-seed experiments
- Interactive Streamlit dashboard

## Installation

Python 3.12 is recommended.

Clone the repository:

    git clone https://github.com/Algoryns07/-unified-predictive-analytics.git

Install dependencies:

    python -m pip install -r requirements.txt

## Launch Dashboard

Run from the repository root:

    streamlit run applications/metropt/dashboard/app.py

## Tests

MetroPT and Impact-Aware applications maintain separate test suites. Run tests from their respective application directories.

## Methodological Principles

- Use time-aware validation for time-dependent datasets.
- Distinguish feature drift from predictive performance deterioration.
- Separate controlled synthetic experiments from real-world observations.
- Evaluate challenger models before approving model replacement.
- Report assumptions, limitations, and uncertainty transparently.

## Data

Raw datasets and large generated artifacts are not intended to be committed to Git. Refer to application data documentation for dataset placement.

## Project Status

The MetroPT dashboard and Impact-Aware monitoring functionality are integrated. Additional dataset applications and experimental validation remain ongoing.

## Limitations

Controlled synthetic drift experiments do not establish real-world drift performance. Results must be interpreted in the context of their datasets and experimental conditions.

## License

No open-source license is provided. All rights reserved.

## Live Application

**Access the deployed dashboard:**

https://impact-aware-ml.streamlit.app/

The application is hosted on Streamlit Community Cloud.
