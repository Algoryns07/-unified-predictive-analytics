import numpy as np
import pandas as pd
import pytest
from src.interactive_pipeline import profile_dataset, train_and_predict, compare_feature_drift, analyze_dataset_shift, _bh_adjust


def sample_frames():
    rng = np.random.default_rng(12)
    n = 160
    x = np.linspace(0, 10, n) + rng.normal(0, .1, n)
    ref = pd.DataFrame({"sensor": x, "other": rng.normal(5, 1, n)})
    ref["target"] = 2.5 * ref["sensor"] - .4 * ref["other"] + rng.normal(0, .2, n)
    inc = pd.DataFrame({"sensor": np.linspace(10, 15, 40), "other": rng.normal(5, 1, 40)})
    return ref, inc


def test_profile_dataset_reports_shape_and_missing():
    result = profile_dataset(pd.DataFrame({"x": [1, None], "cat": ["a", "b"]}))
    assert result["rows"] == 2
    assert result["missing_cells"] == 1
    assert "x" in result["numeric_columns"]


def test_train_predict_and_drift():
    ref, inc = sample_frames()
    predictions, metrics, features = train_and_predict(ref, inc, "target", "Linear Regression")
    assert len(predictions) == len(inc)
    assert "prediction" in predictions
    assert metrics["RMSE"] >= 0
    drift = compare_feature_drift(ref, inc, features)
    assert set(["PSI", "KS_p_value", "adjusted_q_value", "drift_flag"]).issubset(drift.columns)
    assert len(drift) == 2


def test_incoming_labels_produce_incoming_metrics():
    ref, inc = sample_frames()
    inc["target"] = 2.5 * inc["sensor"] - .4 * inc["other"]
    _, metrics, _ = train_and_predict(ref, inc, "target", "Linear Regression")
    assert "incoming_MAE" in metrics


def test_rejects_small_reference_dataset():
    ref, inc = sample_frames()
    with pytest.raises(ValueError, match="20 reference rows"):
        train_and_predict(ref.head(10), inc, "target")


def test_decision_tree_model():
    ref, inc = sample_frames()
    predictions, metrics, features = train_and_predict(
        ref, inc, "target", "Decision Tree"
    )
    assert len(predictions) == len(inc)
    assert "prediction" in predictions.columns
    assert metrics["RMSE"] >= 0
    assert len(features) > 0

from src.interactive_pipeline import (
    recommend_retraining,
    estimate_retraining_costs,
)


def test_no_drift_recommends_continue():
    drift = pd.DataFrame({"drift_flag": [False, False]})
    metrics = {"MAE": 1.0, "incoming_MAE": 1.05, "incoming_labeled_rows": 100}

    result = recommend_retraining(drift, metrics)

    assert result["status"] == "CONTINUE"
    assert result["performance_verified"] is True


def test_drift_without_labels_recommends_monitor():
    drift = pd.DataFrame({"drift_flag": [True, False]})
    metrics = {"MAE": 1.0}

    result = recommend_retraining(drift, metrics)

    assert result["status"] == "MONITOR"
    assert result["performance_verified"] is False


def test_insufficient_labels_do_not_trigger_retraining():
    drift = pd.DataFrame({"drift_flag": [True]})
    metrics = {
        "MAE": 1.0,
        "incoming_MAE": 2.0,
        "incoming_labeled_rows": 10,
    }

    result = recommend_retraining(
        drift, metrics, minimum_labeled_rows=30
    )

    assert result["status"] == "MONITOR"
    assert result["performance_degraded"] is False


def test_significant_performance_degradation_recommends_challenger():
    drift = pd.DataFrame({"drift_flag": [False, True]})
    metrics = {
        "MAE": 1.0,
        "incoming_MAE": 1.50,
        "incoming_labeled_rows": 100,
    }

    result = recommend_retraining(drift, metrics)

    assert result["status"] == "RETRAIN_CANDIDATE"
    assert result["performance_degraded"] is True
    assert result["MAE_degradation_ratio"] == pytest.approx(0.50)


def test_drift_without_performance_degradation_recommends_monitor():
    drift = pd.DataFrame({"drift_flag": [True]})
    metrics = {
        "MAE": 1.0,
        "incoming_MAE": 1.10,
        "incoming_labeled_rows": 100,
    }

    result = recommend_retraining(drift, metrics)

    assert result["status"] == "MONITOR"
    assert result["performance_verified"] is True
    assert result["performance_degraded"] is False


def test_periodic_cost_calculation():
    decision = {"status": "RETRAIN_CANDIDATE"}

    result = estimate_retraining_costs(
        decision,
        horizon_windows=30,
        periodic_interval=7,
        retraining_cost=1000,
        monitoring_cost_per_window=10,
    )

    assert result["periodic_retraining_events"] == 5
    assert result["adaptive_retraining_events"] == 1
    assert result["periodic_total_cost"] == pytest.approx(5300)
    assert result["adaptive_total_cost"] == pytest.approx(1300)
    assert result["estimated_savings"] == pytest.approx(4000)


def test_no_retraining_cost_when_continue_recommended():
    decision = {"status": "CONTINUE"}

    result = estimate_retraining_costs(
        decision,
        horizon_windows=30,
        periodic_interval=7,
        retraining_cost=1000,
        monitoring_cost_per_window=10,
    )

    assert result["adaptive_retraining_events"] == 0
    assert result["adaptive_total_cost"] == pytest.approx(300)
    assert result["estimated_savings"] == pytest.approx(5000)


def test_negative_cost_is_rejected():
    with pytest.raises(ValueError, match="negative"):
        estimate_retraining_costs(
            {"status": "CONTINUE"},
            retraining_cost=-1,
        )


def test_whole_dataset_scan_covers_numeric_categorical_and_schema():
    reference = pd.DataFrame({
        "sensor": np.arange(100, dtype=float),
        "mode": ["A"] * 80 + ["B"] * 20,
        "target": np.arange(100, dtype=float),
        "removed_column": np.arange(100),
    })
    incoming = pd.DataFrame({
        "sensor": np.arange(100, 200, dtype=float),
        "mode": ["B"] * 80 + ["A"] * 20,
        "target": np.arange(100, dtype=float),
        "new_column": ["new"] * 100,
    })
    result = analyze_dataset_shift(reference, incoming)
    report = result["features"].set_index("feature")
    assert {"sensor", "mode", "target", "removed_column", "new_column"}.issubset(report.index)
    assert report.loc["sensor", "analysis_type"] == "Numeric"
    assert report.loc["mode", "analysis_type"] == "Categorical"
    assert result["schema"]["reference_only_columns"] == ["removed_column"]
    assert result["schema"]["incoming_only_columns"] == ["new_column"]
    assert bool(report.loc["removed_column", "drift_flag"])
    assert bool(report.loc["new_column", "drift_flag"])


def test_whole_dataset_scan_flags_type_and_missingness_changes():
    reference = pd.DataFrame({"value": np.arange(50, dtype=float), "category": ["x"] * 50})
    incoming = pd.DataFrame({"value": ["1"] * 45 + [None] * 5, "category": ["x"] * 25 + ["y"] * 25})
    result = analyze_dataset_shift(reference, incoming)
    report = result["features"].set_index("feature")
    assert "value" in result["schema"]["type_changed_columns"]
    assert report.loc["value", "schema_status"] == "Type changed"
    assert bool(report.loc["value", "drift_flag"])
    assert report.loc["category", "analysis_type"] == "Categorical"


def test_whole_dataset_scan_rejects_empty_or_duplicate_columns():
    with pytest.raises(ValueError, match="at least one row"):
        analyze_dataset_shift(pd.DataFrame(), pd.DataFrame({"x": [1]}))
    duplicate = pd.DataFrame([[1, 2]], columns=["x", "x"])
    with pytest.raises(ValueError, match="unique"):
        analyze_dataset_shift(duplicate, pd.DataFrame({"x": [1]}))


def test_no_drift_without_labels_is_insufficient_evidence():
    result = recommend_retraining(pd.DataFrame({"drift_flag": [False, False]}), {"MAE": 1.0})
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["performance_verified"] is False


def test_feature_importance_is_returned_for_prediction():
    reference, incoming = sample_frames()
    _, metrics, features = train_and_predict(reference, incoming, "target", "Linear Regression")
    assert "feature_importance" in metrics
    assert set(metrics["feature_importance"]).issubset(set(features))

def test_bh_adjustment_matches_known_values():
    p_values = np.array([0.01, 0.04, 0.03, 0.002])
    adjusted = _bh_adjust(p_values)

    expected = np.array([0.02, 0.04, 0.04, 0.008])

    assert np.allclose(adjusted, expected)
    assert np.all(adjusted >= p_values)
    assert np.all(adjusted <= 1.0)


def test_whole_dataset_reports_nonfinite_values_and_numeric_ranges():
    reference = pd.DataFrame({
        "sensor": [1.0, 1.0, 2.0, 3.0, np.inf, np.nan]
    })

    incoming = pd.DataFrame({
        "sensor": [1.0, 2.0, 3.0, 4.0, -np.inf, np.nan]
    })

    result = analyze_dataset_shift(reference, incoming)
    row = result["features"].set_index("feature").loc["sensor"]

    assert row["reference_nonfinite_count"] == 1
    assert row["incoming_nonfinite_count"] == 1
    assert row["reference_min"] == 1.0
    assert row["reference_max"] == 3.0
    assert row["incoming_min"] == 1.0
    assert row["incoming_max"] == 4.0


def test_whole_dataset_reports_duplicate_rates():
    reference = pd.DataFrame({"sensor": [1, 1, 2, 3]})
    incoming = pd.DataFrame({"sensor": [1, 2, 3, 4]})

    result = analyze_dataset_shift(reference, incoming)

    assert result["reference_duplicates"] == 1
    assert result["incoming_duplicates"] == 0
    assert result["reference_duplicate_rate_pct"] == 25.0
    assert result["incoming_duplicate_rate_pct"] == 0.0
