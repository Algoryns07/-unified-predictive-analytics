from src.run_project import classification_metrics


def test_classification_metrics_and_confusion_matrix():
    result = classification_metrics(
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        [0.1, 0.7, 0.8, 0.9],
    )

    assert result["true_negative"] == 1
    assert result["false_positive"] == 1
    assert result["false_negative"] == 0
    assert result["true_positive"] == 2

    assert result["accuracy"] == 0.75
    assert result["recall"] == 1.0
    assert 0 <= result["f1"] <= 1
    assert 0 <= result["roc_auc"] <= 1
