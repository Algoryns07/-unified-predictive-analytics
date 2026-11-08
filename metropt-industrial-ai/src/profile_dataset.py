import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "metropt-industrial-ai" / "data" / "raw" / "dataset_train.csv"
OUTPUT_DIR = ROOT / "metropt-industrial-ai" / "outputs"

TIMESTAMP_COLUMN = "timestamp"

DISCRETE_COLUMNS = [
    "COMP",
    "DV_eletric",
    "Pressure_switch",
    "LPS",
]


def profile_dataset(chunksize=100_000):
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("METROPT FULL DATASET PROFILING")
    print("=" * 75)
    print(f"Dataset: {DATA_PATH}")
    print(f"File size: {DATA_PATH.stat().st_size / (1024 ** 3):.3f} GB")
    print(f"Chunk size: {chunksize:,}")

    total_rows = 0
    missing_counts = Counter()
    invalid_numeric_counts = Counter()

    numeric_stats = {}
    discrete_counts = {
        column: Counter() for column in DISCRETE_COLUMNS
    }

    timestamp_min = None
    timestamp_max = None
    first_timestamp = None
    last_timestamp = None

    invalid_timestamps = 0
    duplicate_adjacent_timestamps = 0
    out_of_order_timestamps = 0

    previous_timestamp = None
    interval_counts = Counter()

    columns = None
    numeric_columns = None

    reader = pd.read_csv(
        DATA_PATH,
        chunksize=chunksize,
        encoding="utf-8-sig",
        low_memory=False,
    )

    for chunk_number, chunk in enumerate(reader, start=1):

        if columns is None:
            columns = list(chunk.columns)

            if TIMESTAMP_COLUMN not in columns:
                raise ValueError(
                    f"Missing required timestamp column: {TIMESTAMP_COLUMN}"
                )

            numeric_columns = [
                column for column in columns
                if column != TIMESTAMP_COLUMN
            ]

            numeric_stats = {
                column: {
                    "count": 0,
                    "sum": 0.0,
                    "sum_squares": 0.0,
                    "minimum": None,
                    "maximum": None,
                }
                for column in numeric_columns
            }

            print("\nColumns detected:", len(columns))
            print("Numeric sensor columns:", len(numeric_columns))

        rows = len(chunk)
        total_rows += rows

        # Missing-value counts
        for column in columns:
            missing_counts[column] += int(chunk[column].isna().sum())

        # Numeric sensor statistics
        for column in numeric_columns:
            original = chunk[column]
            values = pd.to_numeric(original, errors="coerce")
            values = values.replace([np.inf, -np.inf], np.nan)

            invalid_numeric_counts[column] += int(
                (original.notna() & values.isna()).sum()
            )

            valid = values.dropna()

            if len(valid) == 0:
                continue

            stats = numeric_stats[column]

            stats["count"] += int(len(valid))
            stats["sum"] += float(valid.sum())
            stats["sum_squares"] += float(
                np.square(valid.to_numpy(dtype=np.float64)).sum()
            )

            current_min = float(valid.min())
            current_max = float(valid.max())

            stats["minimum"] = (
                current_min if stats["minimum"] is None
                else min(stats["minimum"], current_min)
            )

            stats["maximum"] = (
                current_max if stats["maximum"] is None
                else max(stats["maximum"], current_max)
            )

        # Discrete operating-state frequencies
        for column in DISCRETE_COLUMNS:
            if column in chunk.columns:
                counts = chunk[column].value_counts(dropna=False)

                for value, count in counts.items():
                    discrete_counts[column][str(value)] += int(count)

        # Timestamp quality and coverage
        timestamps = pd.to_datetime(
            chunk[TIMESTAMP_COLUMN],
            errors="coerce",
        )

        invalid_timestamps += int(timestamps.isna().sum())
        valid_timestamps = timestamps.dropna()

        if not valid_timestamps.empty:

            if first_timestamp is None:
                first_timestamp = valid_timestamps.iloc[0]

            timestamp_min = (
                valid_timestamps.min()
                if timestamp_min is None
                else min(timestamp_min, valid_timestamps.min())
            )

            timestamp_max = (
                valid_timestamps.max()
                if timestamp_max is None
                else max(timestamp_max, valid_timestamps.max())
            )

            # Check ordering and adjacent duplicates across chunk boundaries.
            if previous_timestamp is not None:
                difference = (
                    valid_timestamps.iloc[0] - previous_timestamp
                ).total_seconds()

                if difference == 0:
                    duplicate_adjacent_timestamps += 1
                elif difference < 0:
                    out_of_order_timestamps += 1
                else:
                    interval_counts[round(difference, 6)] += 1

            differences = valid_timestamps.diff().dt.total_seconds()

            duplicate_adjacent_timestamps += int(
                (differences == 0).sum()
            )

            out_of_order_timestamps += int(
                (differences < 0).sum()
            )

            positive_differences = differences[differences > 0]

            for interval, count in positive_differences.value_counts().items():
                interval_counts[round(float(interval), 6)] += int(count)

            previous_timestamp = valid_timestamps.iloc[-1]
            last_timestamp = valid_timestamps.iloc[-1]

        print(
            f"Processed chunk {chunk_number:,} | "
            f"Rows processed: {total_rows:,}"
        )

    if total_rows == 0:
        raise ValueError("The dataset contains no records.")

    # Final numeric statistics
    final_numeric_stats = {}

    for column, stats in numeric_stats.items():
        count = stats["count"]

        if count == 0:
            final_numeric_stats[column] = {
                "count": 0,
                "mean": None,
                "std": None,
                "minimum": None,
                "maximum": None,
            }
            continue

        mean = stats["sum"] / count

        variance = (
            stats["sum_squares"] / count
        ) - (mean ** 2)

        variance = max(variance, 0.0)

        final_numeric_stats[column] = {
            "count": count,
            "mean": mean,
            "std": float(np.sqrt(variance)),
            "minimum": stats["minimum"],
            "maximum": stats["maximum"],
        }

    missing_report = []

    for column in columns:
        missing = missing_counts[column]

        missing_report.append({
            "column": column,
            "missing_count": missing,
            "missing_percentage": round(
                100 * missing / total_rows, 6
            ),
            "invalid_numeric_count": invalid_numeric_counts[column],
        })

    missing_df = pd.DataFrame(missing_report)
    numeric_df = pd.DataFrame.from_dict(
        final_numeric_stats,
        orient="index",
    ).reset_index(names="column")

    discrete_report = {
        column: dict(counts.most_common(30))
        for column, counts in discrete_counts.items()
    }

    summary = {
        "dataset": "MetroPT dataset_train.csv",
        "file_size_bytes": DATA_PATH.stat().st_size,
        "total_rows": total_rows,
        "total_columns": len(columns),
        "columns": columns,
        "timestamp": {
            "first_valid_timestamp_in_file": (
                str(first_timestamp) if first_timestamp is not None else None
            ),
            "last_valid_timestamp_in_file": (
                str(last_timestamp) if last_timestamp is not None else None
            ),
            "minimum_timestamp": (
                str(timestamp_min) if timestamp_min is not None else None
            ),
            "maximum_timestamp": (
                str(timestamp_max) if timestamp_max is not None else None
            ),
            "invalid_timestamps": invalid_timestamps,
            "adjacent_duplicate_timestamps": duplicate_adjacent_timestamps,
            "out_of_order_adjacent_timestamps": out_of_order_timestamps,
            "most_common_positive_intervals_seconds": [
                {
                    "interval_seconds": interval,
                    "count": count,
                }
                for interval, count in interval_counts.most_common(15)
            ],
        },
        "missing_values": {
            column: int(missing_counts[column])
            for column in columns
        },
        "numeric_statistics": final_numeric_stats,
        "discrete_value_frequencies": discrete_report,
        "profiling_notes": [
            "Processed in chunks; the complete CSV was not loaded into memory.",
            "Timestamp duplicate and ordering checks are adjacent-row checks.",
            "Full-row duplicate detection has not been performed.",
            "Numeric standard deviations are population estimates from accumulated sums.",
            "No predictive target or failure label has been assumed.",
        ],
    }

    json_path = OUTPUT_DIR / "metropt_profile_summary.json"
    missing_path = OUTPUT_DIR / "metropt_missing_values.csv"
    numeric_path = OUTPUT_DIR / "metropt_numeric_statistics.csv"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, allow_nan=False)

    missing_df.to_csv(missing_path, index=False)
    numeric_df.to_csv(numeric_path, index=False)

    print("\n" + "=" * 75)
    print("PROFILING COMPLETED")
    print("=" * 75)

    print(f"Total records: {total_rows:,}")
    print(f"Total columns: {len(columns)}")
    print(f"Invalid timestamps: {invalid_timestamps:,}")
    print(
        "Adjacent duplicate timestamps:",
        f"{duplicate_adjacent_timestamps:,}",
    )
    print(
        "Out-of-order adjacent timestamps:",
        f"{out_of_order_timestamps:,}",
    )
    print("Minimum timestamp:", timestamp_min)
    print("Maximum timestamp:", timestamp_max)

    print("\nREPORTS CREATED:")
    print(json_path)
    print(missing_path)
    print(numeric_path)

    print("\nMISSING VALUE SUMMARY:")
    print(
        missing_df.sort_values(
            "missing_percentage",
            ascending=False,
        ).to_string(index=False)
    )

    print("\nNUMERIC SENSOR STATISTICS:")
    print(numeric_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Memory-efficient MetroPT dataset profiling."
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
        help="Number of rows processed per chunk.",
    )

    args = parser.parse_args()

    if args.chunksize < 1:
        parser.error("--chunksize must be greater than zero.")

    profile_dataset(args.chunksize)
