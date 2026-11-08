from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
DOCS = ROOT / "docs"

DATA = OUTPUT / "metropt_minute_level_data.csv"

OUTPUT.mkdir(parents=True, exist_ok=True)
DOCS.mkdir(parents=True, exist_ok=True)

SENSORS = [
    "TP2", "TP3", "H1", "DV_pressure",
    "Reservoirs", "Oil_temperature",
    "Flowmeter", "Motor_current"
]

TARGETS = [
    "Oil_temperature",
    "TP3",
    "Motor_current"
]

LAGS = [1, 2, 5, 10, 15]


def calculate_metrics(actual, predicted):
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted))
    }


def prepare_target_data(data, target):
    frame = data.copy()

    # Each continuous recording period gets a separate segment.
    # No lag or rolling window is allowed to cross a recording gap.
    gap = (
        frame["timestamp"].diff()
        .ne(pd.Timedelta(minutes=1))
    )

    frame["_segment"] = gap.cumsum()

    groups = frame.groupby("_segment", sort=False)

    X = frame[SENSORS].copy()

    # Historical values of the prediction target.
    for lag in LAGS:
        X[f"{target}_lag_{lag}"] = groups[target].shift(lag)

    # Trailing rolling averages, including the current observation.
    X[f"{target}_rolling_mean_5"] = groups[target].transform(
        lambda s: s.rolling(5, min_periods=5).mean()
    )

    X[f"{target}_rolling_mean_15"] = groups[target].transform(
        lambda s: s.rolling(15, min_periods=15).mean()
    )

    # Next-minute target and its actual timestamp.
    y = groups[target].shift(-1)
    target_time = groups["timestamp"].shift(-1)

    valid = (
        X.notna().all(axis=1)
        & y.notna()
        & target_time.notna()
    )

    X = X.loc[valid].astype(float)
    y = y.loc[valid].astype(float)
    target_time = target_time.loc[valid]

    return X, y, target_time


def create_models():
    return {
        "Linear Regression": LinearRegression(),

        "Decision Tree": DecisionTreeRegressor(
            max_depth=10,
            min_samples_leaf=50,
            random_state=42
        ),

        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=100,
            max_leaf_nodes=15,
            learning_rate=0.08,
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=42
        )
    }


def evaluate_target(data, target):
    print(f"\n========== TARGET: {target} ==========")

    X, y, target_time = prepare_target_data(data, target)

    # Chronological boundaries based on target observation timestamps.
    ordered_times = target_time.sort_values().reset_index(drop=True)

    train_boundary = ordered_times.iloc[int(len(ordered_times) * 0.70)]
    test_boundary = ordered_times.iloc[int(len(ordered_times) * 0.85)]

    train_mask = target_time < train_boundary

    validation_mask = (
        (target_time >= train_boundary)
        & (target_time < test_boundary)
    )

    test_mask = target_time >= test_boundary

    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_val, y_val = X.loc[validation_mask], y.loc[validation_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]

    print("Features:", X.shape[1])
    print("Training observations:", len(X_train))
    print("Validation observations:", len(X_val))
    print("Testing observations:", len(X_test))

    if min(len(X_train), len(X_val), len(X_test)) == 0:
        raise RuntimeError(f"Empty chronological split for {target}")

    results = []

    # Baseline 1: persistence.
    for split_name, X_part, y_part in [
        ("Validation", X_val, y_val),
        ("Test", X_test, y_test)
    ]:
        metrics = calculate_metrics(
            y_part,
            X_part[target].to_numpy()
        )

        results.append({
            "target": target,
            "model": "Persistence",
            "split": split_name,
            "observations": len(y_part),
            **metrics
        })

    # Baseline 2: training-period mean.
    mean_prediction = float(y_train.mean())

    for split_name, y_part in [
        ("Validation", y_val),
        ("Test", y_test)
    ]:
        metrics = calculate_metrics(
            y_part,
            np.full(len(y_part), mean_prediction)
        )

        results.append({
            "target": target,
            "model": "Training Mean",
            "split": split_name,
            "observations": len(y_part),
            **metrics
        })

    # Train models using training data only.
    validation_scores = {}

    for model_name, model in create_models().items():
        print("Training:", model_name)

        model.fit(X_train, y_train)

        val_prediction = model.predict(X_val)

        val_metrics = calculate_metrics(
            y_val,
            val_prediction
        )

        validation_scores[model_name] = val_metrics["RMSE"]

        results.append({
            "target": target,
            "model": model_name,
            "split": "Validation",
            "observations": len(y_val),
            **val_metrics
        })

    # Select model using validation RMSE only.
    selected_model = min(
        validation_scores,
        key=validation_scores.get
    )

    print("Selected using validation:", selected_model)

    # Refit each model on training + validation.
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])

    test_predictions = {}

    for model_name, model in create_models().items():
        model.fit(X_train_val, y_train_val)

        prediction = model.predict(X_test)
        test_predictions[model_name] = prediction

        metrics = calculate_metrics(y_test, prediction)

        results.append({
            "target": target,
            "model": model_name,
            "split": "Test",
            "observations": len(y_test),
            **metrics
        })

    print("\nTest results:")

    for model_name, prediction in test_predictions.items():
        metrics = calculate_metrics(y_test, prediction)
        marker = " <-- VALIDATION WINNER" if model_name == selected_model else ""

        print(
            f"{model_name}: "
            f"MAE={metrics['MAE']:.5f}, "
            f"RMSE={metrics['RMSE']:.5f}, "
            f"R2={metrics['R2']:.5f}"
            f"{marker}"
        )

    return results, {
        "target": target,
        "feature_count": X.shape[1],
        "training_observations": len(X_train),
        "validation_observations": len(X_val),
        "testing_observations": len(X_test),
        "train_boundary": str(train_boundary),
        "test_boundary": str(test_boundary),
        "selected_model_by_validation_RMSE": selected_model,
        "validation_RMSE_by_model": validation_scores
    }


def main():
    if not DATA.exists():
        raise FileNotFoundError(DATA)

    print("Loading cached minute-level data...")
    data = pd.read_csv(DATA, parse_dates=["timestamp"])

    data = data.sort_values("timestamp").reset_index(drop=True)

    if data["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps found.")

    if not data["timestamp"].is_monotonic_increasing:
        raise ValueError("Timestamps are not chronological.")

    print("Observations:", f"{len(data):,}")
    print("Time range:", data["timestamp"].min(), "to", data["timestamp"].max())

    all_results = []
    target_summaries = []

    for target in TARGETS:
        results, summary = evaluate_target(data, target)
        all_results.extend(results)
        target_summaries.append(summary)

    result_df = pd.DataFrame(all_results)

    result_path = OUTPUT / "metropt_baseline_regression_results.csv"
    json_path = OUTPUT / "metropt_baseline_regression_summary.json"
    report_path = DOCS / "METROPT_BASELINE_REGRESSION_REPORT.md"

    result_df.to_csv(result_path, index=False)

    summary = {
        "dataset": "MetroPT 2022",
        "prediction_task": "Next consecutive one-minute sensor prediction",
        "targets": target_summaries,
        "models": [
            "Linear Regression",
            "Decision Tree",
            "HistGradientBoosting"
        ],
        "baselines": [
            "Persistence",
            "Training Mean"
        ],
        "validation_strategy": "Chronological 70/15/15 split",
        "feature_engineering": {
            "current_sensor_features": SENSORS,
            "target_lags": LAGS,
            "rolling_windows": [5, 15],
            "recording_gap_handling": "Separate segments; no lag windows across gaps"
        },
        "limitations": [
            "One-minute aggregation may hide short-duration events.",
            "Only one chronological holdout period is evaluated.",
            "Model selection uses validation RMSE, not test performance.",
            "Further repeated temporal evaluation is required."
        ]
    }

    json_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    report = [
        "# MetroPT Baseline Regression Experiment",
        "",
        "## Experimental design",
        "",
        "- Forecast horizon: next consecutive one-minute observation.",
        "- Split: chronological 70% training, 15% validation, 15% testing.",
        "- Models: Linear Regression, Decision Tree, HistGradientBoosting.",
        "- Baselines: persistence and training-period mean.",
        "- Features: current sensor values, target lags and rolling averages.",
        "- Recording gaps: lag and rolling windows do not cross gaps.",
        "- Model selection: validation RMSE.",
        "",
        "## Results",
        "",
        "| Target | Model | Split | N | MAE | RMSE | R² |",
        "|---|---|---|---:|---:|---:|---:|"
    ]

    for row in result_df.itertuples(index=False):
        report.append(
            f"| {row.target} | {row.model} | {row.split} | "
            f"{row.observations} | {row.MAE:.6f} | "
            f"{row.RMSE:.6f} | {row.R2:.6f} |"
        )

    report.extend([
        "",
        "## Validation-based model selections",
        ""
    ])

    for item in target_summaries:
        report.append(
            f"- **{item['target']}**: "
            f"{item['selected_model_by_validation_RMSE']}"
        )

    report.extend([
        "",
        "## Limitations",
        "",
        "- A single chronological test period does not establish generalization.",
        "- The selected model must be assessed across additional time periods.",
        "- Improvements over persistence must be practically meaningful.",
        "- Prediction uncertainty and drift effects are not yet evaluated.",
        ""
    ])

    report_path.write_text("\n".join(report), encoding="utf-8")

    print("\n========== CONSOLIDATED TEST RESULTS ==========")

    print(
        result_df[
            ["target", "model", "split", "MAE", "RMSE", "R2"]
        ].to_string(index=False)
    )

    print("\nSelected models:")

    for item in target_summaries:
        print(
            item["target"],
            "->",
            item["selected_model_by_validation_RMSE"]
        )

    print("\nGenerated:")
    print(result_path)
    print(json_path)
    print(report_path)

    print("\nBaseline regression experiment completed.")


if __name__ == "__main__":
    main()
