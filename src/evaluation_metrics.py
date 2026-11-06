def evaluate_detection(true_drift, predicted_drift):
    """Evaluate binary drift detection using confusion-matrix metrics."""

    if len(true_drift) != len(predicted_drift):
        raise ValueError("Input sequences must have equal lengths.")

    if len(true_drift) == 0:
        raise ValueError("Input sequences cannot be empty.")

    valid_values = {0, 1, False, True}

    if any(value not in valid_values for value in true_drift):
        raise ValueError("true_drift must contain only binary values.")

    if any(value not in valid_values for value in predicted_drift):
        raise ValueError("predicted_drift must contain only binary values.")

    tn = fp = fn = tp = 0

    for actual, predicted in zip(true_drift, predicted_drift):
        if actual == 0 and predicted == 0:
            tn += 1
        elif actual == 0 and predicted == 1:
            fp += 1
        elif actual == 1 and predicted == 0:
            fn += 1
        else:
            tp += 1

    def safe_divide(numerator, denominator):
        return numerator / denominator if denominator else 0.0

    accuracy = safe_divide(tp + tn, tp + tn + fp + fn)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    false_positive_rate = safe_divide(fp, fp + tn)

    f1 = safe_divide(
        2 * precision * recall,
        precision + recall
    )

    return {
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate": false_positive_rate,
        "f1": f1,
    }
