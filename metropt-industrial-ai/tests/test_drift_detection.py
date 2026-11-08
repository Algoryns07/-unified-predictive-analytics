import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from drift_detection import (
    benjamini_hochberg,
    detect_feature_drift,
    population_stability_index
)


def test_psi_identical_distributions():
    values = np.linspace(0, 1, 500)

    assert population_stability_index(
        values, values
    ) == pytest.approx(0.0)


def test_psi_detects_distribution_shift():
    reference = np.linspace(0, 1, 1000)
    current = np.linspace(2, 3, 1000)

    assert population_stability_index(
        reference, current
    ) > 0.2


def test_psi_constant_feature():
    assert population_stability_index(
        np.ones(100), np.ones(100)
    ) == 0.0


def test_bh_adjusted_values_are_monotonic():
    p_values = np.array([0.01, 0.04, 0.03, 0.002])

    adjusted = benjamini_hochberg(p_values)

    assert np.all((adjusted >= 0) & (adjusted <= 1))

    order = np.argsort(p_values)

    assert np.all(
        np.diff(adjusted[order]) >= -1e-12
    )


def test_bh_preserves_original_order():
    p_values = np.array([0.04, 0.01, 0.03])

    adjusted = benjamini_hochberg(p_values)

    assert adjusted[1] <= adjusted[2] <= adjusted[0]


def test_multiple_shifted_features_are_detected():
    rng = np.random.default_rng(42)

    reference = pd.DataFrame({
        "sensor_a": rng.normal(0, 1, 1000),
        "sensor_b": rng.normal(0, 1, 1000),
        "sensor_c": rng.normal(0, 1, 1000)
    })

    current = pd.DataFrame({
        "sensor_a": rng.normal(2, 1, 1000),
        "sensor_b": rng.normal(2, 1, 1000),
        "sensor_c": rng.normal(0, 1, 1000)
    })

    result = detect_feature_drift(reference, current)

    assert result["drift_detected"] is True
    assert result["drifted_feature_count"] >= 2
    assert "sensor_a" in result["drifted_features"]
    assert "sensor_b" in result["drifted_features"]


def test_identical_data_is_not_flagged():
    rng = np.random.default_rng(7)

    frame = pd.DataFrame({
        "sensor_a": rng.normal(size=500),
        "sensor_b": rng.normal(size=500),
        "sensor_c": rng.normal(size=500)
    })

    result = detect_feature_drift(frame, frame.copy())

    assert result["drift_detected"] is False
    assert result["drifted_feature_count"] == 0


def test_feature_order_can_differ():
    reference = pd.DataFrame({
        "a": np.arange(100),
        "b": np.arange(100) * 2
    })

    current = reference[["b", "a"]].copy()

    result = detect_feature_drift(reference, current)

    assert list(
        result["feature_results"]["feature"]
    ) == ["a", "b"]


def test_mismatched_columns_are_rejected():
    reference = pd.DataFrame({"a": [1.0, 2.0]})
    current = pd.DataFrame({"b": [1.0, 2.0]})

    with pytest.raises(ValueError):
        detect_feature_drift(reference, current)


def test_non_numeric_data_are_rejected():
    frame = pd.DataFrame({"a": ["x", "y"]})

    with pytest.raises(TypeError):
        detect_feature_drift(frame, frame)


def test_non_finite_data_are_rejected():
    reference = pd.DataFrame({"a": [1.0, np.nan]})
    current = pd.DataFrame({"a": [1.0, 2.0]})

    with pytest.raises(ValueError):
        detect_feature_drift(reference, current)


@pytest.mark.parametrize("kwargs", [
    {"psi_threshold": -0.1},
    {"psi_threshold": np.nan},
    {"q_value_threshold": 1.1},
    {"q_value_threshold": True},
    {"bins": 1},
    {"bins": 2.5},
    {"min_drifted_features": 0}
])
def test_invalid_configuration_is_rejected(kwargs):
    frame = pd.DataFrame({
        "a": [1.0, 2.0],
        "b": [2.0, 3.0]
    })

    with pytest.raises((TypeError, ValueError)):
        detect_feature_drift(frame, frame, **kwargs)


def test_bh_rejects_invalid_p_values():
    with pytest.raises(ValueError):
        benjamini_hochberg([0.1, np.nan])

    with pytest.raises(ValueError):
        benjamini_hochberg([0.1, 1.2])

    with pytest.raises(ValueError):
        benjamini_hochberg([])


def test_psi_identical_constant_distributions():
    reference = np.zeros(500)
    current = np.zeros(500)

    assert population_stability_index(
        reference, current
    ) == pytest.approx(0.0)


def test_psi_detects_shift_from_constant_reference():
    reference = np.zeros(500)
    current = np.full(500, 5.0)

    assert population_stability_index(
        reference, current
    ) > 0.2


def test_psi_detects_partial_shift_from_constant_reference():
    reference = np.zeros(500)
    current = np.concatenate([
        np.zeros(250),
        np.full(250, 5.0)
    ])

    assert population_stability_index(
        reference, current
    ) > 0.2


def test_psi_detects_variable_current_against_constant_reference():
    reference = np.zeros(500)
    current = np.linspace(-2, 2, 500)

    assert population_stability_index(
        reference, current
    ) > 0.2


def test_psi_detects_shift_between_two_constant_distributions():
    reference = np.full(500, 2.0)
    current = np.full(500, 8.0)

    assert population_stability_index(
        reference, current
    ) > 0.2


def test_psi_handles_two_valued_reference():
    reference = np.tile([0.0, 10.0], 250)
    current = np.full(500, 10.0)

    assert population_stability_index(
        reference, current
    ) > 0.2
