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
