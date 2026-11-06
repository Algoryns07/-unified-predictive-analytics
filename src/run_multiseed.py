"""Run repeated drift experiments and aggregate results across random seeds."""

from pathlib import Path
import json

import pandas as pd

from src.run_project import run
from src.evaluation_metrics import evaluate_detection


DEFAULT_SEEDS = (11, 42, 73, 101, 202)

POLICY_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "f1_gain_vs_no_retraining",
    "retraining_cost_units",
    "net_utility",
    "retraining_performed",
]

SCENARIO_METRICS = [
    "drift_detected",
    "monitoring_f1",
    "monitoring_recall",
]


def run_multiseed(seeds=DEFAULT_SEEDS, output_dir=Path("outputs/multi_seed")):
    """Execute experiments for multiple seeds and summarize variability."""

    seeds = tuple(seeds)

    if not seeds:
        raise ValueError("At least one random seed is required.")

    if len(set(seeds)) != len(seeds):
        raise ValueError("Random seeds must be unique.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy_runs = []
    scenario_runs = []

    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed}"

        scenario_frame, comparison_frame = run(
            random_state=seed,
            output_dir=seed_dir,
        )

        scenario_frame = scenario_frame.copy()
        comparison_frame = comparison_frame.copy()

        scenario_frame["random_state"] = seed
        comparison_frame["random_state"] = seed

        scenario_runs.append(scenario_frame)
        policy_runs.append(comparison_frame)

    all_scenarios = pd.concat(scenario_runs, ignore_index=True)
    all_policies = pd.concat(policy_runs, ignore_index=True)

    available_policy_metrics = [
        metric for metric in POLICY_METRICS
        if metric in all_policies.columns
    ]

    policy_summary = (
        all_policies
        .groupby(["scenario", "policy"])[available_policy_metrics]
        .agg(["mean", "std"])
        .reset_index()
    )

    policy_summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple) else column
        for column in policy_summary.columns
    ]

    available_scenario_metrics = [
        metric for metric in SCENARIO_METRICS
        if metric in all_scenarios.columns
    ]

    scenario_summary = (
        all_scenarios
        .groupby("scenario")[available_scenario_metrics]
        .agg(["mean", "std"])
        .reset_index()
    )

    scenario_summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple) else column
        for column in scenario_summary.columns
    ]

    true_labels = all_scenarios["scenario"].map({
        "control_no_shift": 0,
        "single_feature_shift": 1,
        "multi_feature_shift": 1,
    })

    if true_labels.isna().any():
        raise ValueError("An experiment scenario has no ground-truth drift label.")

    detection_metrics = evaluate_detection(
        true_labels.astype(int).tolist(),
        all_scenarios["drift_detected"].astype(int).tolist(),
    )

    pd.DataFrame([detection_metrics]).to_csv(
        output_dir / "drift_detection_metrics.csv", index=False
    )

    scenario_summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple) else column
        for column in scenario_summary.columns
    ]

    all_policies.to_csv(
        output_dir / "policy_results_by_seed.csv",
        index=False,
    )

    policy_summary.to_csv(
        output_dir / "policy_metrics_summary.csv",
        index=False,
    )

    all_scenarios.to_csv(
        output_dir / "scenario_results_by_seed.csv",
        index=False,
    )

    scenario_summary.to_csv(
        output_dir / "scenario_detection_summary.csv",
        index=False,
    )

    metadata = {
        "random_seeds": list(seeds),
        "number_of_seeds": len(seeds),
        "number_of_scenarios": int(all_scenarios["scenario"].nunique()),
        "number_of_policies": int(all_policies["policy"].nunique()),
        "aggregation": "Mean and sample standard deviation across seeds",
        "detection_metrics": detection_metrics,
        "note": (
            "Repeated controlled experiments. Results are based on "
            "synthetic feature shifts in the breast cancer dataset "
            "and are not real-world evidence."
        ),
    }

    (output_dir / "multiseed_summary.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\n========== MULTI-SEED POLICY SUMMARY ==========")
    print(policy_summary.to_string(index=False))

    print("\n========== MULTI-SEED SCENARIO SUMMARY ==========")
    print(scenario_summary.to_string(index=False))

    print("\n========== DRIFT DETECTION QUALITY ==========")
    for metric, value in detection_metrics.items():
        print(f"{metric}: {value}")

    print(f"\nResults saved in: {output_dir.resolve()}")

    return all_scenarios, all_policies, policy_summary, scenario_summary


if __name__ == "__main__":
    run_multiseed()
