import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from performance_monitoring import (
    calculate_error_metrics,
    evaluate_performance_deterioration,
    identify_sustained_deterioration,
    calculate_signal_association,
)


def test_error_metrics():

    result = calculate_error_metrics(
        [1, 2, 3],
        [1, 3, 3],
    )

    assert result["mae"] == pytest.approx(1 / 3)
    assert result["rmse"] == pytest.approx(np.sqrt(1 / 3))
    assert result["bias"] == pytest.approx(-1 / 3)


def test_perfect_predictions():

    result = calculate_error_metrics(
        [1, 2, 3],
        [1, 2, 3],
    )

    assert result["mae"] == 0
    assert result["rmse"] == 0


def test_non_finite_actual_rejected():

    with pytest.raises(ValueError):
        calculate_error_metrics([1, np.nan], [1, 2])


def test_unequal_lengths_rejected():

    with pytest.raises(ValueError):
        calculate_error_metrics([1, 2], [1])


def test_deterioration_detected():

    result = evaluate_performance_deterioration(
        1.6,
        1.0,
        1.5,
    )

    assert result["deteriorated"] is True
    assert result["error_ratio"] == pytest.approx(1.6)


def test_deterioration_not_detected():

    result = evaluate_performance_deterioration(
        1.4,
        1.0,
        1.5,
    )

    assert result["deteriorated"] is False


def test_invalid_baseline_rejected():

    with pytest.raises(ValueError):
        evaluate_performance_deterioration(1.0, 0.0)


def test_invalid_multiplier_rejected():

    with pytest.raises(ValueError):
        evaluate_performance_deterioration(1.0, 1.0, 1.0)


def test_sustained_deterioration():

    result = identify_sustained_deterioration(
        [False, True, True, True, False],
        [0, 0, 0, 0, 0],
        required_windows=2,
    )

    assert result == [False, False, True, True, False]


def test_sustained_detection_resets_at_block_boundary():

    result = identify_sustained_deterioration(
        [True, True, True],
        [0, 0, 1],
        required_windows=2,
    )

    assert result == [False, True, False]


def test_signal_association():

    result = calculate_signal_association(
        [True, True, False, False],
        [True, False, True, False],
    )

    assert result["both"] == 1
    assert result["drift_only"] == 1
    assert result["performance_only"] == 1
    assert result["neither"] == 1
    assert result["cooccurrence_rate_pct"] == 25


def test_signal_length_validation():

    with pytest.raises(ValueError):
        calculate_signal_association([True], [True, False])
