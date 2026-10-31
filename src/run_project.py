"""Run the central impact-aware data drift experiment."""
from pathlib import Path
import json
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.drift_detection import compare_frames
from src.impact_policy import decide

def metrics(model, X, y):
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    return {"accuracy": float(accuracy_score(y,pred)), "precision": float(precision_score(y,pred)),
            "recall": float(recall_score(y,pred)), "f1": float(f1_score(y,pred)),
            "roc_auc": float(roc_auc_score(y,prob))}

def run():
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    bunch = load_breast_cancer(as_frame=True)
    X, y = bunch.data, bunch.target
    X_train, X_hold, y_train, y_hold = train_test_split(X, y, test_size=.35, stratify=y, random_state=42)
    X_ref, X_unused, y_ref, y_unused = train_test_split(X_hold, y_hold, test_size=.5, stratify=y_hold, random_state=42)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))
    model.fit(X_train, y_train)
    ref_metrics = metrics(model, X_ref, y_ref)
    scenarios = []
    for name, cols, shift in [
        ("control_no_shift", [], 0.0),
        ("single_feature_shift", [X.columns[0]], 1.5),
        ("multi_feature_shift", list(X.columns[:5]), 1.0),
    ]:
        X_current = X_unused.copy()
        for col in cols:
            X_current[col] = X_current[col] + shift
        drift = compare_frames(X_ref, X_current)
        current_metrics = metrics(model, X_current, y_unused)
        decision = decide(drift["any_drift"], ref_metrics["f1"], current_metrics["f1"])
        scenarios.append({"scenario": name, "drifted_features": ",".join(drift["drifted_features"]),
                          "drift_detected": drift["any_drift"], **{f"reference_{k}":v for k,v in ref_metrics.items()},
                          **{f"current_{k}":v for k,v in current_metrics.items()}, **decision})
    frame = pd.DataFrame(scenarios)
    frame.to_csv(out / "drift_experiment_results.csv", index=False)
    (out / "drift_experiment_summary.json").write_text(json.dumps(
        {"dataset": "scikit-learn breast cancer", "model": "Logistic Regression",
         "note": "Controlled simulated feature shifts; not real-world evidence."}, indent=2))
    print(frame[["scenario","drift_detected","impact_class","recommended_action","f1_drop"]].to_string(index=False))
    print("Saved outputs/drift_experiment_results.csv")

if __name__ == "__main__":
    run()
