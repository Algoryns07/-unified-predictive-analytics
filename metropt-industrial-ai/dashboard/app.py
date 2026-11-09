from pathlib import Path
import sys
import hashlib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.interactive_pipeline import (
    profile_dataset,
    train_and_predict,
    compare_feature_drift,
    recommend_retraining,
    estimate_retraining_costs,
    analyze_dataset_shift,
)

st.set_page_config(page_title="Predictive Analytics | Data Drift Monitoring", page_icon="\U0001F4CA", layout="wide")
st.title("Intelligent Predictive Analytics with Data Drift Monitoring")
st.caption("Interactive dataset exploration, regression prediction and impact-aware retraining assessment | MetroPT application")

with st.sidebar:
    st.header("How to use")
    st.markdown("1. Upload a **reference** dataset (historical/training data).\n2. Upload an **incoming** dataset to assess.\n3. Select the target and model.\n4. Run prediction and drift analysis.")
    st.warning("Recommendations are decision-support estimates. They do not automatically replace a deployed model.")

outputs_dir = ROOT / "outputs"

tab_upload, tab_results, tab_about = st.tabs(["Upload & Predict", "Existing Experiments", "Method & Limitations"])

with tab_upload:
    st.subheader("1. Upload datasets")
    c1, c2 = st.columns(2)
    with c1:
        reference_file = st.file_uploader("Reference dataset (CSV)", type=["csv"], key="reference_csv", help="Historical data used to fit and validate the model.")
    with c2:
        incoming_file = st.file_uploader("Incoming dataset (CSV)", type=["csv"], key="incoming_csv", help="New data for predictions and drift comparison.")

    if reference_file is not None and incoming_file is not None:
        _reference_bytes = reference_file.getvalue()
        _incoming_bytes = incoming_file.getvalue()
        _upload_signature = hashlib.sha256(_reference_bytes + b"\\0" + _incoming_bytes).hexdigest()
        _previous_result = st.session_state.get("interactive_result")
        if _previous_result and _previous_result.get("upload_signature") != _upload_signature:
            st.session_state.pop("interactive_result", None)
    else:
        _upload_signature = None
        st.session_state.pop("interactive_result", None)

    if reference_file is not None:
        try:
            reference = pd.read_csv(reference_file)
            ref_profile = profile_dataset(reference)
            st.markdown("#### Reference dataset profile")
            a,b,c,d = st.columns(4)
            a.metric("Rows", f"{ref_profile['rows']:,}"); b.metric("Columns", ref_profile["columns"])
            c.metric("Missing cells", f"{ref_profile['missing_cells']:,}"); d.metric("Duplicate rows", f"{ref_profile['duplicate_rows']:,}")
            with st.expander("Preview and column quality"):
                st.dataframe(reference.head(100), use_container_width=True)
                st.dataframe(ref_profile["missing_by_column"].rename("missing_values").to_frame(), use_container_width=True)
        except Exception as exc:
            st.error(f"Could not read reference CSV: {exc}")
            reference = None
    else:
        reference = None

    if incoming_file is not None:
        try:
            incoming = pd.read_csv(incoming_file)
            inc_profile = profile_dataset(incoming)
            st.markdown("#### Incoming dataset profile")
            a,b,c,d = st.columns(4)
            a.metric("Rows", f"{inc_profile['rows']:,}"); b.metric("Columns", inc_profile["columns"])
            c.metric("Missing cells", f"{inc_profile['missing_cells']:,}"); d.metric("Duplicate rows", f"{inc_profile['duplicate_rows']:,}")
            with st.expander("Incoming data preview"):
                st.dataframe(incoming.head(100), use_container_width=True)
        except Exception as exc:
            st.error(f"Could not read incoming CSV: {exc}")
            incoming = None
    else:
        incoming = None

    if reference is not None and incoming is not None:
        dataset_report = analyze_dataset_shift(reference, incoming)
        all_features = dataset_report["features"]
        st.divider()
        st.subheader("Whole-dataset intelligence")
        st.caption("Independent of the selected prediction target: every column in both uploaded files is assessed.")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Shared columns", len(dataset_report["schema"]["shared_columns"]))
        q2.metric("Columns flagged", int(all_features["drift_flag"].sum()) if not all_features.empty else 0)
        q3.metric("Reference duplicates", f"{dataset_report['reference_duplicates']:,}")
        q4.metric("Incoming duplicates", f"{dataset_report['incoming_duplicates']:,}")
        if dataset_report["schema"]["reference_only_columns"]:
            st.warning("Missing from incoming: " + ", ".join(dataset_report["schema"]["reference_only_columns"]))
        if dataset_report["schema"]["incoming_only_columns"]:
            st.info("New in incoming: " + ", ".join(dataset_report["schema"]["incoming_only_columns"]))
        if dataset_report["schema"]["type_changed_columns"]:
            st.warning("Changed data types: " + ", ".join(dataset_report["schema"]["type_changed_columns"]))
        if not all_features.empty:
            columns = [
                "feature", "analysis_type", "schema_status", "severity",
                "reference_missing_pct", "incoming_missing_pct",
                "PSI", "KS_p_value", "Wasserstein", "TV_distance",
                "drift_flag", "recommended_action",
            ]
            st.dataframe(all_features[columns], use_container_width=True, hide_index=True)
            with st.expander("Detailed all-column statistics"):
                st.dataframe(all_features, use_container_width=True, hide_index=True)
        st.caption("Screening thresholds: numeric PSI â‰¥ 0.20 or BH-adjusted KS q-value < 0.05; categorical total-variation distance â‰¥ 0.20 or BH-adjusted chi-square q-value < 0.05. Distribution changes do not prove model failure.")
        shared = [c for c in reference.columns if c in incoming.columns]
        targets = [c for c in shared if c in reference.select_dtypes(include="number").columns]
        st.divider()
        st.subheader("2. Configure prediction")
        if not targets:
            st.error("No shared numeric target column found. Both files must contain the numeric target in the reference data. Incoming target is optional.")
        else:
            left, middle, right = st.columns(3)
            with left:
                target = st.selectbox("Prediction target", targets, index=targets.index("Oil_temperature") if "Oil_temperature" in targets else 0)
            with middle:
                model_name = st.selectbox("Regression model", ["HistGradientBoosting", "Linear Regression", "Decision Tree"])
            with right:
                test_fraction = st.slider("Reference validation fraction", 0.1, 0.5, 0.2, 0.05)

            st.caption("The reference rows are split chronologically only when a recognisable timestamp column is present; otherwise file row order is preserved.")

            if st.button("Run prediction and drift analysis", type="primary", use_container_width=True):
                try:
                    with st.spinner("Training the selected model, generating predictions and measuring feature drift..."):
                        prediction_df, metrics, features = train_and_predict(reference, incoming, target, model_name, test_fraction)
                        numeric_features = [f for f in features if f in reference.select_dtypes(include="number").columns and f in incoming.select_dtypes(include="number").columns]
                        drift_df = compare_feature_drift(reference, incoming, numeric_features)
                        decision = recommend_retraining(dataset_report["features"], metrics)

                    st.session_state["interactive_result"] = {
                        "upload_signature": _upload_signature,
                        "predictions": prediction_df,
                        "metrics": metrics,
                        "drift": drift_df,
                        "dataset_report": dataset_report,
                        "decision": decision,
                        "target": target,
                        "model": model_name,
                    }
                except Exception as exc:
                    st.error(f"Analysis could not be completed: {exc}")

    result = st.session_state.get("interactive_result")

    if result:
        st.divider()
        st.subheader("3. Prediction results")
        metrics = result["metrics"]

        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Validation MAE", f"{metrics['MAE']:.4f}")
        m2.metric("Validation RMSE", f"{metrics['RMSE']:.4f}")
        m3.metric("Validation R-squared", f"{metrics['R2']:.4f}")
        m4.metric("Incoming rows predicted", f"{len(result['predictions']):,}")

        if "incoming_MAE" in metrics:
            st.info(f"Incoming labelled data: MAE {metrics['incoming_MAE']:.4f} | RMSE {metrics['incoming_RMSE']:.4f} across {metrics['incoming_labeled_rows']:,} rows.")

        st.caption(f"Model: {result['model']} | Target: {result['target']} | Training rows: {metrics['training_rows']:,} | Holdout rows: {metrics['validation_rows']:,}")
        st.dataframe(result["predictions"].head(500), use_container_width=True)

        csv = result["predictions"].to_csv(index=False).encode("utf-8")
        st.download_button("Download predictions as CSV", csv, "predictions.csv", "text/csv")

        if result["target"] in result["predictions"].columns:
            chart = result["predictions"][[result["target"], "prediction"]].head(500).reset_index(drop=True)
            st.plotly_chart(px.line(chart, title="Actual vs predicted values (first 500 rows)"), use_container_width=True)

        st.divider()
        st.subheader("4. Feature drift assessment")
        drift = result["drift"]

        if drift.empty:
            st.warning("No numeric shared input features had enough valid observations for drift testing.")
        else:
            drifted = int(drift["drift_flag"].sum())
            d1,d2,d3 = st.columns(3)
            d1.metric("Features tested", len(drift))
            d2.metric("Features flagged", drifted)
            d3.metric("Flagged feature share", f"{100*drifted/len(drift):.1f}%")
            st.dataframe(drift, use_container_width=True)
            st.plotly_chart(px.bar(drift, x="feature", y="PSI", color="drift_flag", title="Population Stability Index by feature"), use_container_width=True)
            st.caption("Exploratory thresholds: PSI >= 0.20 or total-variation distance >= 0.20, or Benjamini-Hochberg adjusted q-value < 0.05. Drift alone does not prove accuracy loss.")

        st.divider()
        st.subheader("5. Impact-aware retraining recommendation")

        st.caption("Configure the minimum evidence required before treating incoming performance degradation as measurable.")

        col_a, col_b = st.columns(2)
        with col_a:
            degradation_threshold = st.slider(
                "Required MAE degradation",
                min_value=5,
                max_value=100,
                value=25,
                step=5,
                format="%d%%",
                key="decision_threshold",
            )
        with col_b:
            minimum_labels = st.number_input(
                "Minimum labelled incoming rows",
                min_value=5,
                max_value=1000,
                value=30,
                step=5,
                key="minimum_labels",
            )

        full_dataset_features = result.get("dataset_report", {}).get("features", drift)
        decision = recommend_retraining(
            full_dataset_features,
            metrics,
            performance_threshold=degradation_threshold / 100.0,
            minimum_labeled_rows=int(minimum_labels),
        )

        if decision["status"] == "RETRAIN_CANDIDATE":
            st.error("## RETRAINING RECOMMENDED")
            st.markdown("**Action:** Train and evaluate a challenger model. Do not replace the current model without independent validation.")
        elif decision["status"] == "MONITOR":
            st.warning("## MONITOR - DO NOT RETRAIN YET")
            st.markdown("**Action:** Investigate flagged columns and collect labelled observations.")
        elif decision["status"] == "INSUFFICIENT_EVIDENCE":
            st.info("## INSUFFICIENT EVIDENCE")
            st.markdown("**Action:** Collect enough labelled incoming observations to verify model performance.")
        else:
            st.success("## DO NOT RETRAIN")
            st.markdown("**Action:** Keep the current model and continue routine monitoring.")

        st.write("**Recommendation:**", decision["recommendation"])
        st.write("**Reason:**", decision["reason"])

        st.markdown("#### Feature investigation priorities")
        ranked = result.get("dataset_report", {}).get("features", pd.DataFrame()).copy()
        if not ranked.empty:
            importance = metrics.get("feature_importance", {})
            ranked["model_importance"] = ranked["feature"].map(importance).fillna(0.0)
            ranked["attention_score"] = (
                ranked["drift_flag"].astype(float)
                + ranked["model_importance"].rank(pct=True).fillna(0.0) * 0.5
                + (ranked["missingness_change_pp"].abs().fillna(0.0) / 100.0).clip(upper=1.0)
            )
            ranked = ranked.sort_values(["attention_score", "drift_flag"], ascending=[False, False])
            st.dataframe(
                ranked[[
                    "feature", "severity", "drift_flag", "model_importance",
                    "missingness_change_pp", "attention_score", "recommended_action",
                ]].head(30),
                use_container_width=True,
                hide_index=True,
            )
            st.caption("Permutation importance is predictive association, not causality. The attention score prioritises investigation; it does not independently trigger retraining.")
        else:
            st.info("No comparable columns were available for feature ranking.")

        evidence_a, evidence_b, evidence_c = st.columns(3)
        evidence_a.metric("Drifted features", decision["drifted_features"])
        evidence_b.metric("Labelled observations", decision["labeled_rows"])
        evidence_c.metric(
            "Performance verified",
            "Yes" if decision["performance_verified"] else "No",
        )

        if decision["MAE_degradation_ratio"] is not None:
            st.metric(
                "Incoming MAE degradation",
                f"{decision['MAE_degradation_ratio'] * 100:.2f}%",
            )

        st.info(
            "A retraining candidate is not an automatic approval. Train a challenger "
            "on appropriate recent data and compare it against the existing model "
            "on an independent, representative validation window."
        )

        st.divider()
        st.subheader("6. Estimated retraining cost comparison")

        st.caption("All amounts are assumptions supplied by the user. They are not measured invoices or guaranteed savings.")

        cost_a, cost_b, cost_c, cost_d = st.columns(4)

        with cost_a:
            horizon = st.number_input(
                "Evaluation horizon (windows)",
                min_value=1,
                max_value=365,
                value=30,
            )

        with cost_b:
            interval = st.number_input(
                "Periodic retraining interval",
                min_value=1,
                max_value=365,
                value=7,
            )

        with cost_c:
            retraining_cost = st.number_input(
                "Cost per retraining (INR)",
                min_value=0.0,
                value=2500.0,
                step=500.0,
            )

        with cost_d:
            monitoring_cost = st.number_input(
                "Monitoring cost per window (INR)",
                min_value=0.0,
                value=50.0,
                step=10.0,
            )

        costs = estimate_retraining_costs(
            decision,
            horizon_windows=int(horizon),
            periodic_interval=int(interval),
            retraining_cost=float(retraining_cost),
            monitoring_cost_per_window=float(monitoring_cost),
        )

        cost_m1, cost_m2, cost_m3 = st.columns(3)
        cost_m1.metric("Periodic policy cost", f"INR {costs['periodic_total_cost']:,.2f}")
        cost_m2.metric("Impact-aware policy cost", f"INR {costs['adaptive_total_cost']:,.2f}")
        cost_m3.metric(
            "Estimated operational savings",
            f"INR {costs['estimated_savings']:,.2f}",
            f"{costs['estimated_savings_percentage']:.1f}%",
        )

        st.write(
            f"Periodic retraining events: **{costs['periodic_retraining_events']}** "
            f"| Impact-aware retraining events: **{costs['adaptive_retraining_events']}**"
        )

        st.caption(
            "Cost model limitation: this first version compares assumed retraining "
            "and monitoring expenditure only. It excludes prediction-error impact, "
            "downtime, labour not included in the inputs and infrastructure overhead. "
            "Savings are therefore scenario estimates, not validated financial outcomes."
        )

with tab_results:
    st.subheader("Previously generated experiment results")

    result_files = {
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

    available = {
        name: filename
        for name, filename in result_files.items()
        if (outputs_dir / filename).exists()
    }

    if not available:
        st.info("No experiment output CSVs were found in the outputs folder.")
    else:
        choice = st.selectbox("Select experiment", list(available), key="old_experiment")
        frame = pd.read_csv(outputs_dir / available[choice])
        st.caption(f"Source: {available[choice]} | {len(frame):,} rows")
        st.dataframe(frame, use_container_width=True)

        numeric = frame.select_dtypes(include="number").columns.tolist()
        if numeric:
            metric = st.selectbox("Metric", numeric, key="old_metric")
            st.plotly_chart(px.line(frame, y=metric, markers=True, title=metric), use_container_width=True)

with tab_about:
    st.subheader("Methodology and limitations")

    st.markdown("""
    - **Reference data:** used to fit a model and create a holdout validation estimate.
    - **Incoming data:** receives predictions and is compared with reference features.
    - **Feature drift:** PSI, KS and Wasserstein quantify distribution differences.
    - **Performance monitoring:** incoming MAE is assessed only when actual target labels are supplied.
    - **Retraining recommendation:** requires measured performance degradation, not feature drift alone.
    - **Challenger validation:** a recommendation does not automatically approve model replacement.
    - **Cost comparison:** uses explicit assumptions and excludes unmodelled operational and prediction-error costs.
    - **Specialised MetroPT experiments:** are separate from the generic uploaded-dataset workflow.
    """)

    st.info("Use representative time-aware reference and incoming windows. Avoid random shuffling for time-dependent industrial data.")
