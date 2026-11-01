"""Run controlled drift experiments and compare retraining strategies."""

from pathlib import Path
import json

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.drift_detection import compare_frames
from src.impact_policy import decide


RANDOM_STATE = 42
RETRAINING_COST_F1_EQUIVALENT = 0.08


def make_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    )


def classification_metrics(y_true, predictions, probabilities):
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def evaluate_model(model, X, y):
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    return classification_metrics(y, predictions, probabilities)


def apply_shift(X, columns, amount):
    shifted = X.copy()

    for column in columns:
        shifted[column] = shifted[column] + amount

    return shifted


def run():
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target

    # Training data and independent experimental data.
    X_train, X_hold, y_train, y_hold = train_test_split(
        X,
        y,
        test_size=0.35,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Reference data and remaining experimental data.
    X_reference, X_remaining, y_reference, y_remaining = train_test_split(
        X_hold,
        y_hold,
        test_size=0.50,
        stratify=y_hold,
        random_state=RANDOM_STATE,
    )

    # Monitoring/retraining data and separate future evaluation data.
    X_adapt, X_evaluation, y_adapt, y_evaluation = train_test_split(
        X_remaining,
        y_remaining,
        test_size=0.50,
        stratify=y_remaining,
        random_state=RANDOM_STATE,
    )

    baseline_model = make_model()
    baseline_model.fit(X_train, y_train)

    reference_metrics = evaluate_model(
        baseline_model, X_reference, y_reference
    )

    scenarios = [
        ("control_no_shift", [], 0.0),
        ("single_feature_shift", [X.columns[0]], 1.5),
        ("multi_feature_shift", list(X.columns[:5]), 1.0),
    ]

    policies = [
        "no_retraining",
        "always_retrain",
        "drift_based",
        "impact_aware",
    ]

    scenario_rows = []
    comparison_rows = []

    for scenario_name, shifted_columns, shift_amount in scenarios:

        # Apply the same controlled shift to monitoring and evaluation data.
        X_current = apply_shift(
            X_adapt, shifted_columns, shift_amount
        )
        X_future = apply_shift(
            X_evaluation, shifted_columns, shift_amount
        )

        drift = compare_frames(X_reference, X_current)

        # Measure current performance using monitoring labels.
        monitoring_metrics = evaluate_model(
            baseline_model, X_current, y_adapt
        )

        impact_decision = decide(
            drift["any_drift"],
            reference_metrics["f1"],
            monitoring_metrics["f1"],
        )

        # Keep the original scenario-level experiment report.
        scenario_rows.append({
            "scenario": scenario_name,
            "drift_detected": drift["any_drift"],
            "drifted_features": ",".join(drift["drifted_features"]),
            "reference_f1": reference_metrics["f1"],
            "monitoring_f1": monitoring_metrics["f1"],
            "monitoring_recall": monitoring_metrics["recall"],
            **impact_decision,
        })

        scenario_evaluations = {}

        for policy in policies:

            if policy == "no_retraining":
                should_retrain = False

            elif policy == "always_retrain":
                should_retrain = True

            elif policy == "drift_based":
                should_retrain = drift["any_drift"]

            else:
                should_retrain = (
                    impact_decision["recommended_action"] == "RETRAIN"
                )

            if should_retrain:
                # Retraining uses monitoring data, not evaluation data.
                X_retrain = pd.concat([X_train, X_current])
                y_retrain = pd.concat([y_train, y_adapt])

                model = make_model()
                model.fit(X_retrain, y_retrain)
            else:
                model = baseline_model

            evaluation = evaluate_model(
                model, X_future, y_evaluation
            )

            scenario_evaluations[policy] = {
                "evaluation": evaluation,
                "retrained": should_retrain,
            }

        baseline_f1 = scenario_evaluations[
            "no_retraining"
        ]["evaluation"]["f1"]

        for policy in policies:
            result = scenario_evaluations[policy]
            evaluation = result["evaluation"]
            retrained = result["retrained"]

            f1_gain = evaluation["f1"] - baseline_f1

            # Cost is expressed as an assumed F1-equivalent penalty.
            cost = (
                RETRAINING_COST_F1_EQUIVALENT
                if retrained
                else 0.0
            )

            comparison_rows.append({
                "scenario": scenario_name,
                "policy": policy,
                "drift_detected": drift["any_drift"],
                "impact_class": impact_decision["impact_class"],
                "impact_recommendation": impact_decision[
                    "recommended_action"
                ],
                "retraining_performed": retrained,
                **evaluation,
                "f1_gain_vs_no_retraining": float(f1_gain),
                "retraining_cost_units": cost,
                "net_utility": float(f1_gain - cost),
            })

    scenario_frame = pd.DataFrame(scenario_rows)
    comparison_frame = pd.DataFrame(comparison_rows)

    scenario_frame.to_csv(
        output_dir / "drift_experiment_results.csv",
        index=False,
    )

    comparison_frame.to_csv(
        output_dir / "policy_comparison.csv",
        index=False,
    )

    summary = {
        "dataset": "scikit-learn breast cancer",
        "model": "Logistic Regression",
        "random_state": RANDOM_STATE,
        "data_protocol": (
            "Separate training, reference, monitoring/retraining, "
            "and future evaluation datasets."
        ),
        "strategies": policies,
        "retraining_cost_assumption": {
            "unit": "F1-equivalent",
            "value": RETRAINING_COST_F1_EQUIVALENT,
            "note": (
                "Illustrative cost penalty, not a measured monetary cost."
            ),
        },
        "scenarios": scenario_rows,
        "evaluation_note": (
            "Controlled simulated feature shifts; results are not "
            "real-world evidence."
        ),
    }

    (output_dir / "drift_experiment_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n========== DRIFT SCENARIO RESULTS ==========")
    print(
        scenario_frame[
            [
                "scenario",
                "drift_detected",
                "impact_class",
                "recommended_action",
            ]
        ].to_string(index=False)
    )

    print("\n========== RETRAINING POLICY COMPARISON ==========")
    print(
        comparison_frame[
            [
                "scenario",
                "policy",
                "retraining_performed",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "f1_gain_vs_no_retraining",
                "net_utility",
            ]
        ].to_string(index=False)
    )

    print("\nGenerated:")
    print("outputs/drift_experiment_results.csv")
    print("outputs/policy_comparison.csv")
    print("outputs/drift_experiment_summary.json")


if __name__ == "__main__":
    run()
