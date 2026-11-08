import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = (
    Path(__file__).resolve().parents[1]
    / "src"
)

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from controlled_adaptive_retraining import (
    POLICIES,
    calculate_metrics,
    calculate_calibration_baseline,
    evaluate_policy,
    fit_model,
    calculate_post_injection_retraining_delay,
    count_false_policy_trigger_events,
)


def test_policy_list_contains_four_policies():

    assert len(POLICIES) == 4
    assert len(set(POLICIES)) == 4


def test_calculate_metrics():

    result = calculate_metrics(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 2.0, 4.0]),
    )

    assert result["mae"] == pytest.approx(1 / 3)
    assert result["rmse"] == pytest.approx(
        np.sqrt(1 / 3)
    )


def test_calibration_baseline_is_positive():

    rng = np.random.default_rng(42)

    X_train = pd.DataFrame(
        rng.normal(size=(500, 5)),
        columns=[f"feature_{i}" for i in range(5)],
    )

    y_train = (
        2 * X_train["feature_0"]
        - X_train["feature_1"]
        + rng.normal(0, 0.2, 500)
    )

    model, _ = fit_model(
        X_train,
        y_train,
        42,
    )

    X_cal = pd.DataFrame(
        rng.normal(size=(400, 5)),
        columns=X_train.columns,
    )

    y_cal = (
        2 * X_cal["feature_0"]
        - X_cal["feature_1"]
        + rng.normal(0, 0.2, 400)
    )

    baseline = calculate_calibration_baseline(
        model,
        X_cal,
        y_cal,
    )

    assert np.isfinite(baseline)
    assert baseline > 0


def test_invalid_policy_is_rejected():

    rng = np.random.default_rng(42)

    X = pd.DataFrame(
        rng.normal(size=(100, 5)),
        columns=[f"feature_{i}" for i in range(5)],
    )

    y = rng.normal(size=100)

    model, fit_seconds = fit_model(
        X,
        y,
        42,
    )

    with pytest.raises(ValueError):

        evaluate_policy(
            policy="INVALID",
            seed=42,
            scenario="no_drift",
            X_train=X,
            y_train=y,
            X_calibration=X,
            y_calibration=y,
            X_evaluation=X,
            y_evaluation=y,
            baseline_mae=1.0,
            initial_model=model,
            initial_fit_seconds=fit_seconds,
            injection_index=50,
        )


def test_no_retraining_policy_has_no_events():

    rng = np.random.default_rng(42)

    X_train = pd.DataFrame(
        rng.normal(size=(500, 5)),
        columns=[f"feature_{i}" for i in range(5)],
    )

    y_train = (
        2 * X_train["feature_0"]
        - 1.5 * X_train["feature_1"]
        + rng.normal(0, 0.2, 500)
    )

    X_cal = pd.DataFrame(
        rng.normal(size=(400, 5)),
        columns=X_train.columns,
    )

    y_cal = (
        2 * X_cal["feature_0"]
        - 1.5 * X_cal["feature_1"]
        + rng.normal(0, 0.2, 400)
    )

    X_eval = pd.DataFrame(
        rng.normal(size=(400, 5)),
        columns=X_train.columns,
    )

    y_eval = (
        2 * X_eval["feature_0"]
        - 1.5 * X_eval["feature_1"]
        + rng.normal(0, 0.2, 400)
    )

    model, fit_seconds = fit_model(
        X_train,
        y_train,
        42,
    )

    baseline = calculate_calibration_baseline(
        model,
        X_cal,
        y_cal,
    )

    summary, windows, events = evaluate_policy(
        policy="P1_No_Retraining",
        seed=42,
        scenario="no_drift",
        X_train=X_train,
        y_train=y_train,
        X_calibration=X_cal,
        y_calibration=y_cal,
        X_evaluation=X_eval,
        y_evaluation=y_eval,
        baseline_mae=baseline,
        initial_model=model,
        initial_fit_seconds=fit_seconds,
        injection_index=200,
    )

    assert summary["total_retraining_events"] == 0
    assert summary["candidate_attempts"] == 0
    assert summary["total_fit_seconds"] == 0
    assert len(events) == 0
    assert len(windows) == 2



def test_post_injection_retraining_delay_ignores_earlier_events():

    events = pd.DataFrame([
        {
            "event_type": "accepted_retraining",
            "accepted": True,
            "window_number": 4,
        },
        {
            "event_type": "accepted_retraining",
            "accepted": True,
            "window_number": 9,
        },
        {
            "event_type": "accepted_retraining",
            "accepted": True,
            "window_number": 14,
        },
    ])

    delay = calculate_post_injection_retraining_delay(
        events=events,
        injection_index=2000,
        window_size=200,
        scenario="concept_drift",
    )

    assert delay == 1000


def test_post_injection_retraining_delay_is_nan_without_post_event():

    events = pd.DataFrame([
        {
            "event_type": "accepted_retraining",
            "accepted": True,
            "window_number": 4,
        },
    ])

    delay = calculate_post_injection_retraining_delay(
        events=events,
        injection_index=2000,
        window_size=200,
        scenario="covariate_drift",
    )

    assert np.isnan(delay)


def test_no_drift_retraining_delay_is_nan():

    events = pd.DataFrame([
        {
            "event_type": "accepted_retraining",
            "accepted": True,
            "window_number": 4,
        },
    ])

    delay = calculate_post_injection_retraining_delay(
        events=events,
        injection_index=2000,
        window_size=200,
        scenario="no_drift",
    )

    assert np.isnan(delay)


def test_periodic_retraining_is_not_a_false_policy_trigger():

    events = pd.DataFrame([
        {
            "event_type": "accepted_retraining",
            "accepted": True,
            "window_number": 4,
        },
        {
            "event_type": "accepted_retraining",
            "accepted": True,
            "window_number": 9,
        },
    ])

    count = count_false_policy_trigger_events(
        events=events,
        policy="P2_Periodic",
        scenario="no_drift",
        injection_window=10,
    )

    assert count == 0


def test_no_drift_policy_triggers_are_counted_as_false():

    events = pd.DataFrame([
        {
            "event_type": "accepted_retraining",
            "accepted": True,
            "window_number": 4,
        },
        {
            "event_type": "candidate_evaluation",
            "accepted": True,
            "window_number": 5,
        },
    ])

    count = count_false_policy_trigger_events(
        events=events,
        policy="P3_Performance_Triggered",
        scenario="no_drift",
        injection_window=10,
    )

    assert count == 1


def test_pre_injection_policy_triggers_are_counted():

    events = pd.DataFrame([
        {
            "event_type": "accepted_retraining",
            "accepted": True,
            "window_number": 4,
        },
        {
            "event_type": "accepted_retraining",
            "accepted": True,
            "window_number": 12,
        },
    ])

    count = count_false_policy_trigger_events(
        events=events,
        policy="P4_Impact_Aware",
        scenario="concept_drift",
        injection_window=10,
    )

    assert count == 1
