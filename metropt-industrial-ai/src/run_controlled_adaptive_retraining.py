import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from controlled_adaptive_retraining import (
    POLICIES,
    run_experiment,
    calculate_post_injection_retraining_delay,
    count_false_policy_trigger_events,
)


OUTPUT_DIR = ROOT / "outputs"
REPORT_PATH = (
    ROOT
    / "docs"
    / "CONTROLLED_ADAPTIVE_RETRAINING_REPORT.md"
)

SEEDS = [17, 42, 73, 101, 2026]

SCENARIOS = [
    "no_drift",
    "covariate_drift",
    "concept_drift",
]


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nPHASE 7.6: CONTROLLED-DRIFT ADAPTIVE RETRAINING")
    print("=" * 70)

    parser = argparse.ArgumentParser(
        description="Controlled-drift adaptive retraining evaluation."
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Regenerate summaries and report from existing experiment CSVs.",
    )
    args = parser.parse_args()

    if args.reuse_existing:
        print("Reusing existing experiment results. Models will not be rerun.")

        summaries = pd.read_csv(
            OUTPUT_DIR / "controlled_adaptive_retraining_seed_results.csv"
        )
        windows = pd.read_csv(
            OUTPUT_DIR / "controlled_adaptive_retraining_windows.csv"
        )
        events = pd.read_csv(
            OUTPUT_DIR / "controlled_adaptive_retraining_events.csv"
        )

        if len(summaries) != 60:
            raise RuntimeError(
                f"Expected 60 seed results; found {len(summaries)}."
            )

    else:
        summaries, windows, events = run_experiment(
            seeds=SEEDS,
            scenarios=SCENARIOS,
        )

    # Recalculate reporting-only metrics consistently, whether
    # results were freshly generated or loaded from existing CSVs.
    injection_index = 2000
    window_size = 200
    injection_window = injection_index // window_size

    # Initialize reporting columns before indexed assignment.
    summaries["first_retraining_delay"] = float("nan")
    summaries["false_policy_trigger_events"] = 0

    for index, row in summaries.iterrows():
        matching_events = events[
            (events["scenario"] == row["scenario"])
            & (events["seed"] == row["seed"])
            & (events["policy"] == row["policy"])
        ]

        summaries.at[
            index, "first_retraining_delay"
        ] = calculate_post_injection_retraining_delay(
            events=matching_events,
            injection_index=injection_index,
            window_size=window_size,
            scenario=row["scenario"],
        )

        summaries.at[
            index, "false_policy_trigger_events"
        ] = count_false_policy_trigger_events(
            events=matching_events,
            policy=row["policy"],
            scenario=row["scenario"],
            injection_window=injection_window,
        )

    # Remove the obsolete metric if present in previously saved results.
    if "false_retraining_events" in summaries.columns:
        summaries = summaries.drop(
            columns=["false_retraining_events"]
        )

    summary_path = (
        OUTPUT_DIR
        / "controlled_adaptive_retraining_seed_results.csv"
    )

    windows_path = (
        OUTPUT_DIR
        / "controlled_adaptive_retraining_windows.csv"
    )

    events_path = (
        OUTPUT_DIR
        / "controlled_adaptive_retraining_events.csv"
    )

    aggregate_path = (
        OUTPUT_DIR
        / "controlled_adaptive_retraining_aggregate.csv"
    )

    summaries.to_csv(
        summary_path,
        index=False,
    )

    windows.to_csv(
        windows_path,
        index=False,
    )

    events.to_csv(
        events_path,
        index=False,
    )

    aggregate = (
        summaries
        .groupby(
            ["scenario", "policy"],
            as_index=False,
        )
        .agg(
            experiments=("seed", "count"),
            mean_mae=("overall_mae", "mean"),
            std_mae=("overall_mae", "std"),
            mean_rmse=("overall_rmse", "mean"),
            mean_r2=("overall_r2", "mean"),
            mean_pre_drift_mae=("pre_drift_mae", "mean"),
            mean_post_drift_mae=("post_drift_mae", "mean"),
            mean_post_drift_error_ratio=(
                "post_drift_error_ratio",
                "mean",
            ),
            mean_detection_delay=(
                "performance_detection_delay",
                "mean",
            ),
            mean_retraining_delay=(
                "first_retraining_delay",
                "mean",
            ),
            mean_retraining_events=(
                "total_retraining_events",
                "mean",
            ),
            mean_false_policy_trigger_events=(
                "false_policy_trigger_events",
                "mean",
            ),
            mean_candidate_attempts=(
                "candidate_attempts",
                "mean",
            ),
            mean_candidate_acceptances=(
                "candidate_acceptances",
                "mean",
            ),
            mean_candidate_acceptance_rate_pct=(
                "candidate_acceptance_rate_pct",
                "mean",
            ),
            mean_fit_seconds=(
                "total_fit_seconds",
                "mean",
            ),
            mean_training_rows=(
                "total_training_rows",
                "mean",
            ),
        )
    )

    aggregate.to_csv(
        aggregate_path,
        index=False,
    )

    report = []

    report.append(
        "# Controlled-Drift Adaptive Retraining Evaluation"
    )

    report.append("")
    report.append("## Experimental design")
    report.append("")

    report.extend([
        "- Training observations: 5,000.",
        "- Calibration observations: 1,000.",
        "- Evaluation observations: 4,000.",
        "- Drift injection point: observation 2,000.",
        "- Monitoring window: 200 observations.",
        "- Periodic retraining interval: 1,000 observations.",
        "- Performance threshold: 1.5 times calibration MAE.",
        "- Sustained performance trigger: two consecutive windows.",
        "- Retraining cooldown: 400 observations.",
        "- Random seeds: " + str(SEEDS) + ".",
        "- All policies use the same initial model and evaluation stream.",
        "- Predictions are generated before current targets are revealed.",
        "- Drift injection positions are withheld from policy decisions.",
        "- Feature drift is diagnostic and is not a mandatory trigger.",
    ])

    report.append("")
    report.append("## Policy definitions")
    report.append("")

    report.extend([
        "1. P1: No retraining.",
        "2. P2: Periodic retraining.",
        "3. P3: Sustained performance-triggered retraining.",
        "4. P4: Impact-aware challenger evaluation and cost approval.",
    ])

    report.append("")
    report.append("## Aggregate results")
    report.append("")

    report.append(
        aggregate.to_string(index=False)
    )

    report.append("")
    report.append("## Per-seed results")
    report.append("")

    report.append(
        summaries.to_string(index=False)
    )

    report.append("")
    report.append("## Methodological safeguards")
    report.append("")

    report.extend([
        "- No policy receives the known drift injection position.",
        "- The injection position is used only for retrospective evaluation.",
        "- The same training, calibration and evaluation streams are used across policies for each seed.",
        "- The initial model is fitted only on the initial training stream.",
        "- Calibration observations establish the performance baseline.",
        "- Challenger fitting costs are counted even when candidates are rejected.",
        "- Accepted models are fitted using only observations available at the decision time.",
        "- Retraining affects future predictions, never predictions already evaluated.",
        "- Synthetic scenarios are kept separate from MetroPT real-data evidence.",
    ])

    report.append("")
    report.append("## Metric definitions")
    report.append("")

    report.extend([
        "- Retraining delay is measured from the known injection observation to the end of the first monitoring window containing an accepted post-injection retraining event.",
        "- No-drift experiments have no retraining delay because no drift injection occurs.",
        "- False policy-trigger events count accepted P3/P4 retraining actions before injection, or all accepted P3/P4 actions in a no-drift experiment.",
        "- P2 periodic updates are scheduled maintenance and are excluded from false policy-trigger counts.",
        "- Performance detection delay and retraining delay are separate measures: detecting deterioration does not guarantee that a policy accepts a retraining action.",
    ])

    report.append("")
    report.append("## Interpretation limitations")
    report.append("")

    report.extend([
        "These experiments use controlled synthetic data and do not establish real-world drift behaviour.",
        "Covariate drift and concept drift are generated using predefined mathematical changes.",
        "The challenger is evaluated on the current observed window before that window is added to its training history.",
        "The impact thresholds are operational choices and have not been established as universal optima.",
        "The computational cost depends on the hardware and software environment.",
        "Five random seeds provide a reproducibility check, not a comprehensive statistical guarantee.",
        "Feature-distribution change is not necessarily harmful predictive drift.",
        "Performance deterioration does not, by itself, prove concept drift.",
    ])

    REPORT_PATH.write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    print(
        aggregate.to_string(index=False)
    )

    print("\nOutput files:")
    print(summary_path)
    print(windows_path)
    print(events_path)
    print(aggregate_path)
    print(REPORT_PATH)

    print("\nPhase 7.6 completed.")


if __name__ == "__main__":
    main()
