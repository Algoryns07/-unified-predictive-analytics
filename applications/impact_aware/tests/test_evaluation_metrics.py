import pytest

from src.evaluation_metrics import evaluate_detection


def test_detection_metrics():
    result = evaluate_detection(
        [0, 0, 1, 1],
        [0, 1, 1, 1]
    )

    assert result["true_negative"] == 1
    assert result["false_positive"] == 1
    assert result["false_negative"] == 0
    assert result["true_positive"] == 2

    assert result["accuracy"] == 0.75
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == 1.0
    assert result["specificity"] == 0.5
    assert result["false_positive_rate"] == 0.5
    assert result["f1"] == pytest.approx(0.8)


def test_perfect_detection():
    result = evaluate_detection([0, 1, 0, 1], [0, 1, 0, 1])

    assert result["accuracy"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0


def test_no_false_alarms():
    result = evaluate_detection([0, 0, 0], [0, 0, 0])

    assert result["false_positive"] == 0
    assert result["false_positive_rate"] == 0.0
    assert result["specificity"] == 1.0


def test_mismatched_lengths():
    with pytest.raises(ValueError):
        evaluate_detection([0, 1], [0])


def test_empty_inputs():
    with pytest.raises(ValueError):
        evaluate_detection([], [])


def test_non_binary_values():
    with pytest.raises(ValueError):
        evaluate_detection([0, 2], [0, 1])

def test_balanced_accuracy_and_mcc():
    result = evaluate_detection(
        [0, 0, 0, 1],
        [0, 1, 0, 1]
    )

    assert result["balanced_accuracy"] == pytest.approx(5 / 6)
    assert result["mcc"] > 0


def test_all_negative_predictions_have_safe_mcc():
    result = evaluate_detection([0, 0, 1], [0, 0, 0])

    assert result["mcc"] == 0.0
    assert result["balanced_accuracy"] == pytest.approx(0.5)
