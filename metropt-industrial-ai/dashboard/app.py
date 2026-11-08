import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"

RESULT_FILES = {
    "Baseline regression": "metropt_baseline_regression_results.csv",
    "Temporal robustness": "metropt_temporal_robustness_aggregate.csv",
    "Residual analysis": "metropt_residual_aggregate.csv",
    "Conformal prediction": "metropt_conformal_aggregate.csv",
    "Adaptive calibration": "metropt_adaptive_calibration_aggregate.csv",
    "Adaptive retraining": "adaptive_retraining_aggregate.csv",
    "Controlled drift": "controlled_drift_summary.csv",
    "Controlled adaptive retraining": "controlled_adaptive_retraining_aggregate.csv",
    "Performance-aware drift": "metropt_performance_drift_summary.csv",
}

st.set_page_config(page_title="Predictive Analytics Platform", layout="wide")
st.title("Intelligent Predictive Analytics Platform")
st.caption("MetroPT industrial predictive analytics and adaptive model monitoring")

available = {name: filename for name, filename in RESULT_FILES.items() if (OUTPUTS / filename).exists()}

if not available:
    st.warning("No experiment summary files found. Run the relevant experiment scripts first.")
else:
    choice = st.selectbox("Select experiment", list(available.keys()))
    path = OUTPUTS / available[choice]
    df = pd.read_csv(path)
    st.subheader(choice)
    st.caption(f"Source: {path.name} | Rows: {len(df)}")
    st.dataframe(df, width='stretch')

    numeric = df.select_dtypes(include="number").columns.tolist()
    if numeric:
        metric = st.selectbox("Select metric to visualize", numeric)
        category = st.selectbox("Group results by", ["None"] + df.columns.tolist())
        chart_data = df
        if category != "None" and category != metric:
            fig = px.bar(chart_data, x=category, y=metric, title=f"{metric} by {category}")
        else:
            fig = px.line(chart_data, y=metric, title=metric, markers=True)
        st.plotly_chart(fig, width='stretch')

st.info("Interpret results alongside the technical reports in the docs folder. Synthetic drift results are separate from real MetroPT experiments.")
