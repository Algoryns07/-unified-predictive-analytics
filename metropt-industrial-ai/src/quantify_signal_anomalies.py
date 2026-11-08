import pandas as pd
from pathlib import Path

path = Path("metropt-industrial-ai/data/raw/dataset_train.csv")

columns = [
    "timestamp",
    "TP2",
    "TP3",
    "Motor_current",
    "COMP",
    "DV_eletric",
    "gpsSpeed",
    "gpsQuality",
]

exception_parts = []

speed_counts = {
    "above_100": 0,
    "above_150": 0,
    "above_200": 0,
    "above_250": 0,
}

speed_run_count = 0
previous_was_high = False
longest_run = 0
current_run = 0
total_rows = 0

for chunk in pd.read_csv(
    path,
    usecols=columns,
    chunksize=100_000
):
    total_rows += len(chunk)

    # Digital signal exceptions
    exceptions = chunk[
        (chunk["COMP"] == 1) &
        (chunk["DV_eletric"] == 1)
    ]

    if not exceptions.empty:
        exception_parts.append(exceptions)

    # GPS speed distribution
    valid = chunk[
        (chunk["gpsQuality"] == 1) &
        (chunk["gpsSpeed"] > 100)
    ]

    speed_counts["above_100"] += int((valid["gpsSpeed"] > 100).sum())
    speed_counts["above_150"] += int((valid["gpsSpeed"] > 150).sum())
    speed_counts["above_200"] += int((valid["gpsSpeed"] > 200).sum())
    speed_counts["above_250"] += int((valid["gpsSpeed"] > 250).sum())

    # Consecutive high-speed observations in original row order
    flags = (
        (chunk["gpsQuality"] == 1) &
        (chunk["gpsSpeed"] > 100)
    ).tolist()

    for flag in flags:
        if flag:
            if not previous_was_high:
                speed_run_count += 1
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0

        previous_was_high = flag

print("\n========== DIGITAL SIGNAL EXCEPTIONS ==========")

if exception_parts:
    exceptions = pd.concat(exception_parts, ignore_index=True)

    print(f"Total exceptions: {len(exceptions)}")

    exceptions["motor_current_group"] = pd.cut(
        exceptions["Motor_current"],
        bins=[-float("inf"), 1, 4, float("inf")],
        labels=["Below 1 A", "1 to 4 A", "Above 4 A"]
    )

    print("\nMotor current groups:")
    print(
        exceptions["motor_current_group"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nAnalogue sensor statistics:")
    print(
        exceptions[
            ["TP2", "TP3", "Motor_current"]
        ].describe().round(3).to_string()
    )

print("\n========== GPS SPEED INVESTIGATION ==========")
print(f"Rows scanned: {total_rows:,}")

for key, value in speed_counts.items():
    print(f"{key}: {value}")

print(f"\nSeparate consecutive high-speed episodes: {speed_run_count}")
print(f"Longest consecutive episode: {longest_run} records")

print("\nInvestigation completed.")
