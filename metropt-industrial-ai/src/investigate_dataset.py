import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT / "metropt-industrial-ai"
    / "data" / "raw" / "dataset_train.csv"
)

OUTPUT_DIR = (
    ROOT / "metropt-industrial-ai"
    / "outputs"
)


def investigate(chunksize=100_000):

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_rows = 0

    state_pairs = {}
    lps_by_comp = {}
    lps_by_dv = {}

    gps_counts = {
        "gps_lat_zero": 0,
        "gps_long_zero": 0,
        "gps_both_coordinates_zero": 0,
        "gps_quality_zero": 0,
        "gps_quality_one": 0,
        "gps_speed_above_50": 0,
        "gps_speed_above_100": 0,
    }

    gps_speed_max = None

    previous_timestamp = None
    timestamp_gaps = []

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            DATA_PATH,
            chunksize=chunksize,
            encoding="utf-8-sig",
            low_memory=False,
        ),
        start=1,
    ):

        rows = len(chunk)
        offset = total_rows
        total_rows += rows

        # Exact relationship between COMP and DV_eletric.
        pairs = chunk.groupby(
            ["COMP", "DV_eletric"],
            dropna=False,
        ).size()

        for pair, count in pairs.items():
            key = f"{pair[0]}_{pair[1]}"
            state_pairs[key] = (
                state_pairs.get(key, 0) + int(count)
            )

        # LPS frequency by operating state.
        for column, destination in [
            ("COMP", lps_by_comp),
            ("DV_eletric", lps_by_dv),
        ]:

            grouped = chunk.groupby(
                [column, "LPS"],
                dropna=False,
            ).size()

            for pair, count in grouped.items():
                key = f"{pair[0]}_{pair[1]}"
                destination[key] = (
                    destination.get(key, 0) + int(count)
                )

        # GPS quality and coordinate checks.
        lat = pd.to_numeric(
            chunk["gpsLat"], errors="coerce"
        )

        lon = pd.to_numeric(
            chunk["gpsLong"], errors="coerce"
        )

        speed = pd.to_numeric(
            chunk["gpsSpeed"], errors="coerce"
        )

        quality = pd.to_numeric(
            chunk["gpsQuality"], errors="coerce"
        )

        gps_counts["gps_lat_zero"] += int((lat == 0).sum())
        gps_counts["gps_long_zero"] += int((lon == 0).sum())

        gps_counts["gps_both_coordinates_zero"] += int(
            ((lat == 0) & (lon == 0)).sum()
        )

        gps_counts["gps_quality_zero"] += int(
            (quality == 0).sum()
        )

        gps_counts["gps_quality_one"] += int(
            (quality == 1).sum()
        )

        gps_counts["gps_speed_above_50"] += int(
            (speed > 50).sum()
        )

        gps_counts["gps_speed_above_100"] += int(
            (speed > 100).sum()
        )

        current_max = speed.max()

        if pd.notna(current_max):
            gps_speed_max = (
                float(current_max)
                if gps_speed_max is None
                else max(gps_speed_max, float(current_max))
            )

        # Locate timestamp interruptions.
        timestamps = pd.to_datetime(
            chunk["timestamp"],
            errors="coerce",
        )

        if timestamps.isna().any():
            raise ValueError(
                "Unexpected invalid timestamp found."
            )

        if previous_timestamp is not None:
            first = timestamps.iloc[0]

            difference = (
                first - previous_timestamp
            ).total_seconds()

            if difference > 1:
                timestamp_gaps.append({
                    "previous_row_number": offset,
                    "next_row_number": offset + 1,
                    "gap_seconds": difference,
                    "previous_timestamp": str(previous_timestamp),
                    "next_timestamp": str(first),
                })

        differences = timestamps.diff().dt.total_seconds()

        gap_positions = differences[differences > 1]

        for local_position, difference in gap_positions.items():

            timestamp_gaps.append({
                "previous_row_number": offset + int(local_position),
                "next_row_number": offset + int(local_position) + 1,
                "gap_seconds": float(difference),
                "previous_timestamp": str(
                    timestamps.loc[local_position - 1]
                ),
                "next_timestamp": str(
                    timestamps.loc[local_position]
                ),
            })

        previous_timestamp = timestamps.iloc[-1]

        print(
            f"Processed chunk {chunk_number:,} | "
            f"Rows: {total_rows:,}"
        )

    # Exact complementarity calculation.
    total_mismatches = (
        state_pairs.get("0_0", 0)
        + state_pairs.get("1_1", 0)
    )

    total_matches = (
        state_pairs.get("0_1", 0)
        + state_pairs.get("1_0", 0)
    )

    summary = {
        "total_rows": total_rows,

        "COMP_vs_DV_eletric": {
            "pair_counts": state_pairs,
            "complementary_records": total_matches,
            "non_complementary_records": total_mismatches,
            "complementary_percentage": (
                100 * total_matches / total_rows
            ),
        },

        "LPS_by_COMP": lps_by_comp,
        "LPS_by_DV_eletric": lps_by_dv,

        "GPS_investigation": {
            **gps_counts,
            "maximum_gps_speed": gps_speed_max,
        },

        "timestamp_investigation": {
            "gap_count": len(timestamp_gaps),
            "gaps": timestamp_gaps,
        },
    }

    output_path = (
        OUTPUT_DIR / "metropt_data_investigation.json"
    )

    with output_path.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=4)

    print("\n" + "=" * 70)
    print("DATASET INVESTIGATION COMPLETED")
    print("=" * 70)

    print("\nCOMP vs DV_eletric:")
    print(json.dumps(summary["COMP_vs_DV_eletric"], indent=4))

    print("\nLPS by COMP:")
    print(json.dumps(lps_by_comp, indent=4))

    print("\nLPS by DV_eletric:")
    print(json.dumps(lps_by_dv, indent=4))

    print("\nGPS investigation:")
    print(json.dumps(summary["GPS_investigation"], indent=4))

    print("\nTimestamp gaps:")
    print(json.dumps(timestamp_gaps, indent=4))

    print("\nReport saved to:")
    print(output_path)


if __name__ == "__main__":
    investigate()
