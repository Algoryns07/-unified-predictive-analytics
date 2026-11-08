from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
DOCS = ROOT / "docs"

DATA = OUTPUT / "metropt_minute_level_data.csv"

TARGETS = [
    "Oil_temperature",
    "TP3",
    "Motor_current"
]

SENSORS = [
    "TP2", "TP3", "H1", "DV_pressure",
    "Reservoirs", "Oil_temperature",
    "Flowmeter", "Motor_current"
]

LAGS = [1, 2, 5, 10, 15]
SEEDS = [17, 42, 73]

# Each experiment reserves 10% of observations for calibration
# and 15% for testing.
FOLDS = [
    {
        "name": "Window 1",
        "train_end": 0.45,
        "calibration_end": 0.55,
        "test_end": 0.70
    },
    {
        "name": "Window 2",
        "train_end": 0.60,
        "calibration_end": 0.70,
        "test_end": 0.85
    },
    {
        "name": "Window 3",
        "train_end": 0.75,
        "calibration_end": 0.85,
        "test_end": 1.00
    }
]

MODEL_PARAMETERS = {
    "max_iter": 100,
    "max_leaf_nodes": 15,
    "learning_rate": 0.08,
    "min_samples_leaf": 30,
    "l2_regularization": 1.0
}

COVERAGE_LEVELS = {
    "90": 0.10,
    "95": 0.05
}


def prepare_target_data(data, target):
    frame = data.copy()

    # Prevent lag and rolling features from crossing recording gaps.
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

    # Predict the next consecutive minute.
    y = groups[target].shift(-1)
    target_time = groups["timestamp"].shift(-1)

    valid = X.notna().all(axis=1) & y.notna() & target_time.notna()

    X = X.loc[valid].astype(float)
    y = y.loc[valid].astype(float)
    target_time = target_time.loc[valid]

    return X, y, target_time


def make_model(seed):
    return HistGradientBoostingRegressor(
        **MODEL_PARAMETERS,
        random_state=seed
    )


def calculate_conformal_quantile(calibration_residuals, alpha):
    """
    Finite-sample split-conformal order statistic.

    k = ceil((n + 1) * (1 - alpha))

    If k > n, return infinity, as required by the
    conservative finite-sample construction.
    """
    residuals = np.asarray(calibration_residuals, dtype=float)

    if residuals.ndim != 1 or len(residuals) == 0:
        raise ValueError("Calibration residuals must be a non-empty vector.")

    if not np.isfinite(residuals).all():
        raise ValueError("Calibration residuals contain non-finite values.")

    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between zero and one.")

    n = len(residuals)
    k = math.ceil((n + 1) * (1 - alpha))

    if k > n:
        return float("inf")

    # kth order statistic using zero-based indexing.
    return float(np.partition(residuals, k - 1)[k - 1])


def make_split_masks(target_time, fold):
    ordered = target_time.sort_values().reset_index(drop=True)

    train_position = int(len(ordered) * fold["train_end"])
    calibration_position = int(len(ordered) * fold["calibration_end"])
    test_position = int(len(ordered) * fold["test_end"])

    train_start = ordered.iloc[train_position]
    calibration_start = ordered.iloc[calibration_position]

    if fold["test_end"] < 1.0:
        test_end = ordered.iloc[test_position]
    else:
        test_end = ordered.iloc[-1] + pd.Timedelta(minutes=1)

    return {
        "train": target_time < train_start,
        "calibration": (
            (target_time >= train_start)
            & (target_time < calibration_start)
        ),
        "test": (
            (target_time >= calibration_start)
            & (target_time < test_end)
        )
    }


def evaluate_interval(actual, predicted, quantile):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    lower = predicted - quantile
    upper = predicted + quantile

    covered = (actual >= lower) & (actual <= upper)

    return {
        "coverage": float(np.mean(covered)),
        "mean_width": float(np.mean(upper - lower)),
        "median_width": float(np.median(upper - lower)),
        "lower_miss_rate": float(np.mean(actual < lower)),
        "upper_miss_rate": float(np.mean(actual > upper)),
        "mean_lower_bound": float(np.mean(lower)),
        "mean_upper_bound": float(np.mean(upper))
    }


def run_experiment(data, target, fold):
    print(f"\n{'=' * 70}")
    print(f"TARGET: {target} | {fold['name']}")
    print("=" * 70)

    X, y, target_time = prepare_target_data(data, target)

    masks = make_split_masks(target_time, fold)

    X_train = X.loc[masks["train"]]
    y_train = y.loc[masks["train"]]

    X_cal = X.loc[masks["calibration"]]
    y_cal = y.loc[masks["calibration"]]

    X_test = X.loc[masks["test"]]
    y_test = y.loc[masks["test"]]
    time_test = target_time.loc[masks["test"]]

    if min(len(X_train), len(X_cal), len(X_test)) == 0:
        raise RuntimeError("An experiment split is empty.")

    print(
        f"Training: {len(X_train):,} | "
        f"Calibration: {len(X_cal):,} | "
        f"Testing: {len(X_test):,}"
    )

    all_predictions = []
    all_metrics = []

    for seed in SEEDS:
        print(f"\nTraining model with seed {seed}...")

        model = make_model(seed)
        model.fit(X_train, y_train)

        calibration_prediction = model.predict(X_cal)
        calibration_residuals = np.abs(
            y_cal.to_numpy() - calibration_prediction
        )

        test_prediction = model.predict(X_test)

        actual = y_test.to_numpy()
        residual = actual - test_prediction

        base_metrics = {
            "MAE": float(mean_absolute_error(actual, test_prediction)),
            "RMSE": float(np.sqrt(mean_squared_error(actual, test_prediction))),
            "R2": float(r2_score(actual, test_prediction)),
            "mean_residual": float(np.mean(residual))
        }

        print(
            f"Point prediction | "
            f"MAE={base_metrics['MAE']:.5f} | "
            f"RMSE={base_metrics['RMSE']:.5f} | "
            f"R2={base_metrics['R2']:.5f}"
        )

        for level, alpha in COVERAGE_LEVELS.items():

            quantile = calculate_conformal_quantile(
                calibration_residuals,
                alpha
            )

            interval_metrics = evaluate_interval(
                actual,
                test_prediction,
                quantile
            )

            print(
                f"{level}% interval | "
                f"q={quantile:.5f} | "
                f"coverage={interval_metrics['coverage']:.4f} | "
                f"width={interval_metrics['mean_width']:.5f}"
            )

            all_metrics.append({
                "target": target,
                "fold": fold["name"],
                "seed": seed,
                "interval_level": int(level),
                "alpha": alpha,
                "training_observations": len(X_train),
                "calibration_observations": len(X_cal),
                "testing_observations": len(X_test),
                "calibration_quantile": quantile,
                **base_metrics,
                **interval_metrics
            })

            lower = test_prediction - quantile
            upper = test_prediction + quantile

            all_predictions.append(
                pd.DataFrame({
                    "timestamp": time_test.to_numpy(),
                    "target": target,
                    "fold": fold["name"],
                    "seed": seed,
                    "interval_level": int(level),
                    "actual": actual,
                    "predicted": test_prediction,
                    "residual": residual,
                    "absolute_error": np.abs(residual),
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "interval_width": upper - lower,
                    "covered": (
                        (actual >= lower) & (actual <= upper)
                    )
                })
            )

    return all_predictions, all_metrics


def create_report(metrics, aggregate, path):
    lines = [
        "# MetroPT Split Conformal Prediction Assessment",
        "",
        "## Experimental design",
        "",
        "- Base estimator: HistGradientBoostingRegressor.",
        "- Three targets: Oil_temperature, TP3 and Motor_current.",
        "- Three chronological windows.",
        "- Three random seeds: 17, 42 and 73.",
        "- Training, calibration and test periods are kept separate.",
        "- Calibration uses absolute residuals.",
        "- Finite-sample conformal order statistic is used.",
        "- Prediction intervals are symmetric around point predictions.",
        "- Nominal interval levels: 90% and 95%.",
        "",
        "## Aggregate results",
        "",
        "| Target | Level | Experiments | Mean Coverage | Mean Width | Mean Median Width | Mean Lower Miss | Mean Upper Miss | Mean RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    ]

    for row in aggregate.itertuples(index=False):
        lines.append(
            f"| {row.target} | {row.interval_level}% | "
            f"{row.experiments} | {row.mean_coverage:.4f} | "
            f"{row.mean_width:.6f} | {row.mean_median_width:.6f} | "
            f"{row.mean_lower_miss_rate:.4f} | "
            f"{row.mean_upper_miss_rate:.4f} | "
            f"{row.mean_RMSE:.6f} |"
        )

    lines.extend([
        "",
        "## Per-window results",
        "",
        "| Target | Window | Seed | Level | Coverage | Mean Width | Calibration Quantile | Lower Miss | Upper Miss | RMSE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    ])

    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.target} | {row.fold} | {row.seed} | "
            f"{row.interval_level}% | {row.coverage:.4f} | "
            f"{row.mean_width:.6f} | {row.calibration_quantile:.6f} | "
            f"{row.lower_miss_rate:.4f} | "
            f"{row.upper_miss_rate:.4f} | {row.RMSE:.6f} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Coverage is the observed fraction of test targets inside the prediction interval.",
        "- Mean width measures interval size in the target's original units.",
        "- Lower miss rate measures observations below the lower bound.",
        "- Upper miss rate measures observations above the upper bound.",
        "- Wider intervals are not automatically better; coverage and width must be considered together.",
        "- Nominal 90% and 95% levels are targets, not guaranteed empirical outcomes on chronological data.",
        "",
        "## Important methodological limitations",
        "",
        "- Standard split conformal finite-sample marginal coverage relies on exchangeability.",
        "- Chronological sensor observations may be dependent and non-stationary.",
        "- Distribution shifts between calibration and testing can reduce empirical coverage.",
        "- Results are specific to this dataset and its chronological windows.",
        "- Symmetric intervals may not be optimal for asymmetric residual distributions.",
        "- Coverage is marginal and does not guarantee coverage for every operating condition.",
        "- Repeated seeds measure estimator variability, not independent machine generalization.",
        "",
        "## Next research direction",
        "",
        "Compare fixed calibration intervals with rolling or adaptive calibration under controlled drift.",
        "Evaluate whether coverage deterioration can serve as an impact signal for model monitoring and retraining."
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    if not DATA.exists():
        raise FileNotFoundError(
            f"Cached minute-level dataset not found: {DATA}"
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    print("Loading cached MetroPT minute-level dataset...")

    data = pd.read_csv(
        DATA,
        parse_dates=["timestamp"]
    )

    data = data.sort_values("timestamp").reset_index(drop=True)

    if data["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps detected.")

    if not data["timestamp"].is_monotonic_increasing:
        raise ValueError("Timestamp order validation failed.")

    if data[SENSORS].isna().any().any():
        raise ValueError("Missing sensor values detected.")

    prediction_frames = []
    metric_rows = []

    for target in TARGETS:
        for fold in FOLDS:
            predictions, metrics = run_experiment(
                data,
                target,
                fold
            )

            prediction_frames.extend(predictions)
            metric_rows.extend(metrics)

    prediction_df = pd.concat(
        prediction_frames,
        ignore_index=True
    )

    metric_df = pd.DataFrame(metric_rows)

    aggregate = (
        metric_df.groupby(
            ["target", "interval_level"],
            as_index=False
        )
        .agg(
            experiments=("coverage", "count"),
            mean_coverage=("coverage", "mean"),
            std_coverage=("coverage", "std"),
            mean_width=("mean_width", "mean"),
            mean_median_width=("median_width", "mean"),
            mean_lower_miss_rate=("lower_miss_rate", "mean"),
            mean_upper_miss_rate=("upper_miss_rate", "mean"),
            mean_RMSE=("RMSE", "mean"),
            mean_quantile=("calibration_quantile", "mean")
        )
    )

    prediction_path = OUTPUT / "metropt_conformal_predictions.csv"
    metrics_path = OUTPUT / "metropt_conformal_metrics.csv"
    aggregate_path = OUTPUT / "metropt_conformal_aggregate.csv"
    report_path = DOCS / "METROPT_CONFORMAL_PREDICTION_REPORT.md"

    prediction_df.to_csv(prediction_path, index=False)
    metric_df.to_csv(metrics_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)

    create_report(metric_df, aggregate, report_path)

    print("\n========== AGGREGATE CONFORMAL RESULTS ==========")
    print(aggregate.to_string(index=False))

    print("\n========== GENERATED FILES ==========")

    for path in [
        prediction_path,
        metrics_path,
        aggregate_path,
        report_path
    ]:
        print(path)

    print("\n========== EXPERIMENT SUMMARY ==========")
    print("Targets:", len(TARGETS))
    print("Chronological windows:", len(FOLDS))
    print("Random seeds:", len(SEEDS))
    print("Nominal interval levels:", list(COVERAGE_LEVELS.keys()))
    print("Total model fits:", len(TARGETS) * len(FOLDS) * len(SEEDS))
    print("Total interval evaluations:", len(metric_df))

    print("\nConformal prediction experiment completed.")


if __name__ == "__main__":
    main()
