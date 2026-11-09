import numpy as np
import pandas as pd
import pytest
from src.interactive_pipeline import profile_dataset, train_and_predict, compare_feature_drift


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
