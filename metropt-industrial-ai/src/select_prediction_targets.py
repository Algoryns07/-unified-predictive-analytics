from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "dataset_train.csv"
OUTPUT = ROOT / "outputs"
DOCS = ROOT / "docs"

OUTPUT.mkdir(parents=True, exist_ok=True)
DOCS.mkdir(parents=True, exist_ok=True)

TARGETS = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Flowmeter",
    "Motor_current",
]

COLUMNS = ["timestamp"] + TARGETS
CHUNK_SIZE = 100_000

def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")

    print("Dataset:", DATA)
    print("Reading full dataset in chunks...")

    aggregated_parts = []
    carry = None
    total_rows = 0
    processed_chunks = 0

    for chunk in pd.read_csv(
        DATA,
        usecols=COLUMNS,
        chunksize=CHUNK_SIZE
    ):
        processed_chunks += 1
        total_rows += len(chunk)

        chunk["timestamp"] = pd.to_datetime(
            chunk["timestamp"], errors="coerce"
        )

        chunk = chunk.dropna(subset=["timestamp"])

        for col in TARGETS:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)

        chunk["_minute"] = chunk["timestamp"].dt.floor("min")

        if chunk.empty:
            continue

        last_minute = chunk["_minute"].max()

        complete = chunk[chunk["_minute"] < last_minute]
        carry = chunk[chunk["_minute"] == last_minute].copy()

        if not complete.empty:
            grouped = (
                complete.groupby("_minute", sort=True)[TARGETS]
                .mean()
                .reset_index()
                .rename(columns={"_minute": "timestamp"})
            )
            aggregated_parts.append(grouped)

        if processed_chunks % 10 == 0:
            print(
                f"Processed {processed_chunks} chunks | "
                f"{total_rows:,} raw rows"
            )

    if carry is not None and not carry.empty:
        grouped = (
            carry.groupby("_minute", sort=True)[TARGETS]
            .mean()
            .reset_index()
            .rename(columns={"_minute": "timestamp"})
        )
        aggregated_parts.append(grouped)

    if not aggregated_parts:
        raise RuntimeError("No valid minute-level observations generated.")

    minute_data = pd.concat(aggregated_parts, ignore_index=True)
    minute_data = (
        minute_data.sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    print("\nRaw rows processed:", f"{total_rows:,}")
    print("Minute-level observations:", f"{len(minute_data):,}")

    # Save the reusable intermediate dataset.
    minute_path = OUTPUT / "metropt_minute_level_data.csv"
    minute_data.to_csv(minute_path, index=False)

    results = []

    for target in TARGETS:
        current = minute_data[target]
        future = current.shift(-1)

        # Only evaluate genuine consecutive one-minute observations.
        consecutive = (
            minute_data["timestamp"].shift(-1)
            - minute_data["timestamp"]
        ).eq(pd.Timedelta(minutes=1))

        valid = consecutive & current.notna() & future.notna()

        actual = future[valid].to_numpy(dtype=float)
        predicted = current[valid].to_numpy(dtype=float)

        if len(actual) < 2:
            continue

        errors = actual - predicted

        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))

        denominator = float(np.sum((actual - np.mean(actual)) ** 2))
        r2 = (
            float(1 - np.sum(errors ** 2) / denominator)
            if denominator > 0 else None
        )

        correlation = (
            float(np.corrcoef(predicted, actual)[0, 1])
            if np.std(predicted) > 0 and np.std(actual) > 0
            else None
        )

        std = float(current.std())
        normalized_mae = mae / std if std > 0 else None

        results.append({
            "target": target,
            "valid_consecutive_pairs": int(len(actual)),
            "observed_mean": float(current.mean()),
            "observed_std": std,
            "observed_min": float(current.min()),
            "observed_max": float(current.max()),
            "unique_minute_values": int(current.nunique()),
            "persistence_MAE": mae,
            "persistence_RMSE": rmse,
            "persistence_R2": r2,
            "temporal_correlation": correlation,
            "normalized_MAE": normalized_mae,
            "mean_absolute_change": float(np.mean(np.abs(errors))),
        })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="persistence_R2",
        ascending=False,
        na_position="last"
    ).reset_index(drop=True)

    csv_path = OUTPUT / "metropt_target_screening.csv"
    json_path = OUTPUT / "metropt_target_screening_summary.json"
    report_path = DOCS / "METROPT_PREDICTION_TARGET_ASSESSMENT.md"

    results_df.to_csv(csv_path, index=False)

    summary = {
        "dataset": "MetroPT 2022 Zenodo dataset",
        "raw_rows_processed": total_rows,
        "minute_observations": len(minute_data),
        "aggregation": "One-minute sensor means",
        "forecast_horizon": "Next consecutive one-minute interval",
        "baseline": "Persistence: predict next minute using current minute",
        "targets_evaluated": results,
        "limitations": [
            "Persistence is a baseline, not a trained ML model.",
            "High persistence performance alone does not establish useful predictive value.",
            "Results are observational and do not establish causality.",
            "Minute aggregation can conceal short-lived events.",
            "Target selection requires subsequent chronological model validation."
        ]
    }

    json_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    report = [
        "# MetroPT Prediction Target Assessment",
        "",
        "## Experimental setup",
        "",
        f"- Raw rows processed: {total_rows:,}",
        f"- Minute-level observations: {len(minute_data):,}",
        "- Aggregation: one-minute sensor means.",
        "- Forecast horizon: next consecutive one-minute interval.",
        "- Baseline: persistence prediction.",
        "",
        "## Results",
        "",
        results_df.to_markdown(index=False),
        "",
        "## Interpretation and limitations",
        "",
        "- Persistence is a simple baseline, not a trained predictive model.",
        "- High persistence R² does not independently establish a useful target.",
        "- Short-lived events may be hidden by minute aggregation.",
        "- Long recording gaps are excluded from consecutive-minute comparisons.",
        "- Final target selection requires chronological ML validation.",
        "",
    ]

    report_path.write_text("\n".join(report), encoding="utf-8")

    print("\n========== TARGET SCREENING RESULTS ==========")
    print(
        results_df[
            [
                "target",
                "valid_consecutive_pairs",
                "persistence_MAE",
                "persistence_RMSE",
                "persistence_R2",
                "temporal_correlation"
            ]
        ].to_string(index=False)
    )

    print("\nGenerated files:")
    print(csv_path)
    print(json_path)
    print(report_path)
    print(minute_path)

    print("\nTarget screening completed successfully.")

if __name__ == "__main__":
    main()
