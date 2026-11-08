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

SENSORS = [
    "TP2", "TP3", "H1", "DV_pressure",
    "Reservoirs", "Oil_temperature",
    "Flowmeter", "Motor_current"
]

TARGETS = ["Oil_temperature", "TP3", "Motor_current"]
LAGS = [1, 2, 5, 10, 15]
SEEDS = [17, 42, 73]

# Each test window covers approximately 15% of the observations.
FOLDS = [
    {"name": "Window 1", "train_end": 0.55, "test_end": 0.70},
    {"name": "Window 2", "train_end": 0.70, "test_end": 0.85},
    {"name": "Window 3", "train_end": 0.85, "test_end": 1.00},
]


def calculate_metrics(actual, predicted):
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted))
    }


def prepare_target_data(data, target):
    frame = data.copy()

    # Prevent lag features from crossing recording gaps.
    gap = frame["timestamp"].diff().ne(pd.Timedelta(minutes=1))
    frame["_segment"] = gap.cumsum()

    groups = frame.groupby("_segment", sort=False)

    X = frame[SENSORS].copy()

    for lag in LAGS:
        X[f"{target}_lag_{lag}"] = groups[target].shift(lag)

    X[f"{target}_rolling_mean_5"] = groups[target].transform(
        lambda s: s.rolling(5, min_periods=5).mean()
    )

    X[f"{target}_rolling_mean_15"] = groups[target].transform(
        lambda s: s.rolling(15, min_periods=15).mean()
    )

    y = groups[target].shift(-1)
    target_time = groups["timestamp"].shift(-1)

    valid = X.notna().all(axis=1) & y.notna() & target_time.notna()

    X = X.loc[valid].astype(float)
    y = y.loc[valid].astype(float)
    target_time = target_time.loc[valid]

    return X, y, target_time


def create_models(seed):
    return {
        "Linear Regression": LinearRegression(),

        "Decision Tree": DecisionTreeRegressor(
            max_depth=10,
            min_samples_leaf=50,
            random_state=seed
        ),

        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=100,
            max_leaf_nodes=15,
            learning_rate=0.08,
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=seed
        )
    }


def evaluate_target(data, target):
    print(f"\n{'=' * 65}")
    print(f"TARGET: {target}")
    print("=" * 65)

    X, y, target_time = prepare_target_data(data, target)

    ordered_times = target_time.sort_values().reset_index(drop=True)

    print("Usable observations:", f"{len(X):,}")
    print("Features:", X.shape[1])

    results = []
    fold_summaries = []

    for fold in FOLDS:
        start_position = int(len(ordered_times) * fold["train_end"])
        end_position = min(
            int(len(ordered_times) * fold["test_end"]),
            len(ordered_times) - 1
        )

        start_time = ordered_times.iloc[start_position]
        end_time = (
            ordered_times.iloc[end_position]
            if fold["test_end"] < 1.0
            else ordered_times.iloc[-1] + pd.Timedelta(minutes=1)
        )

        train_mask = target_time < start_time
        test_mask = (
            (target_time >= start_time)
            & (target_time < end_time)
        )

        X_train = X.loc[train_mask]
        y_train = y.loc[train_mask]
        X_test = X.loc[test_mask]
        y_test = y.loc[test_mask]

        if len(X_train) == 0 or len(X_test) == 0:
            raise RuntimeError(
                f"Empty training or test set: {target}, {fold['name']}"
            )

        print(
            f"\n{fold['name']}: "
            f"Train={len(X_train):,}, Test={len(X_test):,}"
        )
        print("Test period:", start_time, "to", end_time)

        fold_summaries.append({
            "target": target,
            "fold": fold["name"],
            "training_observations": len(X_train),
            "testing_observations": len(X_test),
            "test_start": str(start_time),
            "test_end_exclusive": str(end_time)
        })

        # Persistence baseline: predict the next value using the current value.
        persistence_prediction = X_test[target].to_numpy()

        persistence_metrics = calculate_metrics(
            y_test,
            persistence_prediction
        )

        results.append({
            "target": target,
            "fold": fold["name"],
            "seed": None,
            "model": "Persistence",
            "training_observations": len(X_train),
            "testing_observations": len(X_test),
            **persistence_metrics
        })

        for seed in SEEDS:
            models = create_models(seed)

            for model_name, model in models.items():
                print(f"  Seed {seed}: {model_name}", flush=True)

                model.fit(X_train, y_train)
                prediction = model.predict(X_test)

                metrics = calculate_metrics(y_test, prediction)

                results.append({
                    "target": target,
                    "fold": fold["name"],
                    "seed": seed,
                    "model": model_name,
                    "training_observations": len(X_train),
                    "testing_observations": len(X_test),
                    **metrics
                })

                print(
                    f"    MAE={metrics['MAE']:.5f} | "
                    f"RMSE={metrics['RMSE']:.5f} | "
                    f"R2={metrics['R2']:.5f}",
                    flush=True
                )

    return results, fold_summaries


def main():
    if not DATA.exists():
        raise FileNotFoundError(
            f"Cached dataset not found: {DATA}"
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    print("Loading cached minute-level dataset...")
    data = pd.read_csv(DATA, parse_dates=["timestamp"])
    data = data.sort_values("timestamp").reset_index(drop=True)

    if data["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps detected.")

    if not data["timestamp"].is_monotonic_increasing:
        raise ValueError("Timestamps are not chronological.")

    if data[SENSORS].isna().any().any():
        raise ValueError("Missing sensor values detected.")

    print("Observations:", f"{len(data):,}")
    print("Random seeds:", SEEDS)
    print("Temporal windows:", len(FOLDS))

    all_results = []
    all_fold_summaries = []

    for target in TARGETS:
        results, fold_summaries = evaluate_target(data, target)
        all_results.extend(results)
        all_fold_summaries.extend(fold_summaries)

    result_df = pd.DataFrame(all_results)
    fold_df = pd.DataFrame(all_fold_summaries)

    result_path = OUTPUT / "metropt_temporal_robustness_results.csv"
    summary_path = OUTPUT / "metropt_temporal_robustness_summary.json"
    report_path = DOCS / "METROPT_TEMPORAL_ROBUSTNESS_REPORT.md"

    result_df.to_csv(result_path, index=False)

    # Aggregate model performance across folds and random seeds.
    model_df = result_df[result_df["model"] != "Persistence"]

    aggregate = (
        model_df.groupby(["target", "model"])
        .agg(
            experiments=("RMSE", "count"),
            mean_MAE=("MAE", "mean"),
            std_MAE=("MAE", "std"),
            mean_RMSE=("RMSE", "mean"),
            std_RMSE=("RMSE", "std"),
            mean_R2=("R2", "mean"),
            std_R2=("R2", "std")
        )
        .reset_index()
    )

    persistence = (
        result_df[result_df["model"] == "Persistence"]
        .groupby("target")
        .agg(
            mean_persistence_RMSE=("RMSE", "mean"),
            std_persistence_RMSE=("RMSE", "std")
        )
        .reset_index()
    )

    comparison = aggregate.merge(persistence, on="target")

    comparison["RMSE_reduction_percent"] = (
        (
            comparison["mean_persistence_RMSE"]
            - comparison["mean_RMSE"]
        )
        / comparison["mean_persistence_RMSE"]
    ) * 100

    comparison_path = OUTPUT / "metropt_temporal_robustness_aggregate.csv"
    comparison.to_csv(comparison_path, index=False)

    summary = {
        "dataset": "MetroPT 2022",
        "task": "Next consecutive one-minute sensor prediction",
        "targets": TARGETS,
        "seeds": SEEDS,
        "folds": FOLDS,
        "models": [
            "Linear Regression",
            "Decision Tree",
            "HistGradientBoosting"
        ],
        "baseline": "Persistence",
        "validation_design": (
            "Expanding chronological training windows with "
            "three successive non-overlapping test periods"
        ),
        "feature_design": {
            "current_sensor_features": SENSORS,
            "target_lags": LAGS,
            "rolling_windows": [5, 15],
            "recording_gap_handling": (
                "Separate segments; lag and rolling features "
                "do not cross recording gaps"
            )
        },
        "limitations": [
            "The three test windows come from one dataset and one machine context.",
            "Random seeds assess stochastic model variation, not independent datasets.",
            "The experiment does not establish performance under all operating conditions.",
            "Persistence and model comparisons are predictive, not causal.",
            "The test windows are chronological and non-overlapping, but training windows expand."
        ],
        "fold_details": all_fold_summaries,
        "aggregate_results": comparison.to_dict(orient="records")
    }

    summary_path.write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8"
    )

    report = [
        "# MetroPT Temporal Robustness Assessment",
        "",
        "## Experimental design",
        "",
        "- Three expanding chronological training windows.",
        "- Three successive, non-overlapping test periods.",
        "- Three random seeds: 17, 42 and 73.",
        "- Three regression models and a persistence baseline.",
        "- Targets: Oil_temperature, TP3 and Motor_current.",
        "- Metrics: MAE, RMSE and R².",
        "- Lag and rolling features do not cross recording gaps.",
        "- No model selection is performed using test performance.",
        "",
        "## Aggregate results",
        "",
        "| Target | Model | Experiments | Mean RMSE | RMSE Std | Mean R² | Persistence RMSE | RMSE reduction (%) |",
        "|---|---|---:|---:|---:|---:|---:|---:|"
    ]

    for row in comparison.itertuples(index=False):
        report.append(
            f"| {row.target} | {row.model} | {row.experiments} | "
            f"{row.mean_RMSE:.6f} | {row.std_RMSE:.6f} | "
            f"{row.mean_R2:.6f} | {row.mean_persistence_RMSE:.6f} | "
            f"{row.RMSE_reduction_percent:.2f} |"
        )

    report.extend([
        "",
        "## Per-window results",
        "",
        "| Target | Window | Model | Seed | MAE | RMSE | R² |",
        "|---|---|---|---:|---:|---:|---:|"
    ])

    for row in result_df.itertuples(index=False):
        seed = "Baseline" if pd.isna(row.seed) else str(int(row.seed))
        report.append(
            f"| {row.target} | {row.fold} | {row.model} | {seed} | "
            f"{row.MAE:.6f} | {row.RMSE:.6f} | {row.R2:.6f} |"
        )

    report.extend([
        "",
        "## Interpretation",
        "",
        "Mean performance summarizes results across temporal windows and seeds. "
        "Standard deviation indicates variation across these experiments.",
        "",
        "Positive RMSE reduction indicates lower average RMSE than persistence. "
        "Negative reduction indicates worse performance than persistence.",
        "",
        "## Limitations",
        "",
        "- Repeated seeds do not replace validation on independent datasets.",
        "- Temporal windows belong to the same MetroPT recording.",
        "- Model stability should be assessed alongside changes in operating conditions.",
        "- Results do not establish causal relationships or failure prediction capability.",
        ""
    ])

    report_path.write_text("\n".join(report), encoding="utf-8")

    print("\n========== AGGREGATE RESULTS ==========")
    print(
        comparison[
            [
                "target", "model", "experiments",
                "mean_RMSE", "std_RMSE", "mean_R2",
                "mean_persistence_RMSE",
                "RMSE_reduction_percent"
            ]
        ].to_string(index=False)
    )

    print("\n========== GENERATED FILES ==========")
    print(result_path)
    print(comparison_path)
    print(summary_path)
    print(report_path)

    print("\nTemporal robustness experiment completed.")


if __name__ == "__main__":
    main()
