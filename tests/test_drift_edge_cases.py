import numpy as np
import pandas as pd

from src.drift_detection import psi, compare_feature, compare_frames


def test_psi_ignores_non_finite_values():
    reference = np.array([1, 2, 3, 4, np.nan, np.inf])
    current = np.array([1, 2, 3, 4, np.nan, -np.inf])

    assert np.isclose(psi(reference, current), 0.0)


def test_empty_arrays_return_zero_psi():
    assert psi([], []) == 0.0


def test_all_invalid_values_return_safe_statistics():
    result = compare_feature(
        [np.nan, np.inf, -np.inf],
        [np.nan, np.inf, -np.inf]
    )

    assert result["psi"] == 0.0
    assert result["ks_p_value"] == 1.0
    assert result["wasserstein"] == 0.0


def test_constant_reference_feature_handles_zero_variance():
    result = compare_feature([5, 5, 5], [7, 7, 7])

    assert result["wasserstein"] == 2.0
    assert result["normalized_wasserstein"] == 2.0


def test_no_common_columns_returns_no_drift():
    reference = pd.DataFrame({"feature_a": [1, 2, 3]})
    current = pd.DataFrame({"feature_b": [4, 5, 6]})

    result = compare_frames(reference, current)

    assert result["any_drift"] is False
    assert result["drifted_features"] == []
import pytest


@pytest.mark.parametrize("bins", [0, 1, -5, 2.5])
def test_psi_rejects_invalid_bin_counts(bins):
    with pytest.raises(ValueError):
        psi([1, 2, 3], [2, 3, 4], bins=bins)


@pytest.mark.parametrize("epsilon", [0, -0.1, 0.5, 1])
def test_psi_rejects_invalid_epsilon(epsilon):
    with pytest.raises(ValueError):
        psi([1, 2, 3], [2, 3, 4], epsilon=epsilon)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"psi_threshold": -0.1},
        {"psi_threshold": np.nan},
        {"ks_alpha": 0},
        {"ks_alpha": 1},
        {"ks_alpha": -0.1},
        {"wasserstein_threshold": -0.1},
        {"wasserstein_threshold": np.inf},
    ],
)
def test_compare_frames_rejects_invalid_thresholds(kwargs):
    reference = pd.DataFrame({"feature": [1, 2, 3]})
    current = pd.DataFrame({"feature": [2, 3, 4]})

    with pytest.raises(ValueError):
        compare_frames(reference, current, **kwargs)


def test_compare_frames_rejects_non_dataframe_inputs():
    with pytest.raises(TypeError):
        compare_frames([1, 2, 3], [2, 3, 4])

def test_multiple_feature_comparison_reports_adjusted_p_values():
    reference = pd.DataFrame({
        "stable_a": np.arange(1000, dtype=float),
        "stable_b": np.arange(1000, dtype=float),
        "shifted": np.arange(1000, dtype=float),
    })

    current = reference.copy()
    current["shifted"] = current["shifted"] + 500.0

    result = compare_frames(reference, current)

    assert len(result["features"]) == 3

    for statistics in result["features"].values():
        assert "ks_adjusted_p_value" in statistics
        assert 0.0 <= statistics["ks_adjusted_p_value"] <= 1.0
        assert (
            statistics["ks_adjusted_p_value"]
            >= statistics["ks_p_value"] - 1e-12
        )

    assert result["features"]["shifted"]["drift"] is True
    assert result["any_drift"] is True
