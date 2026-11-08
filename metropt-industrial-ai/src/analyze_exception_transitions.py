import pandas as pd
from pathlib import Path

source = Path(
    "metropt-industrial-ai/outputs/metropt_signal_exception_context.csv"
)

df = pd.read_csv(source, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Identify contradictory observations.
mask = (df["COMP"] == 1) & (df["DV_eletric"] == 1)
exceptions = df.loc[mask].copy()

# Find the observations immediately before and after each exception.
records = []

for _, row in exceptions.iterrows():
    t = row["timestamp"]

    before = df.loc[df["timestamp"] == t - pd.Timedelta(seconds=1)]
    after = df.loc[df["timestamp"] == t + pd.Timedelta(seconds=1)]

    if before.empty or after.empty:
        continue

    b = before.iloc[0]
    a = after.iloc[0]

    records.append({
        "timestamp": t,
        "before_COMP": int(b["COMP"]),
        "before_DV": int(b["DV_eletric"]),
        "exception_COMP": int(row["COMP"]),
        "exception_DV": int(row["DV_eletric"]),
        "after_COMP": int(a["COMP"]),
        "after_DV": int(a["DV_eletric"]),
        "motor_before": b["Motor_current"],
        "motor_exception": row["Motor_current"],
        "motor_after": a["Motor_current"],
        "TP2_before": b["TP2"],
        "TP2_exception": row["TP2"],
        "TP2_after": a["TP2"],
    })

result = pd.DataFrame(records)

if result.empty:
    raise RuntimeError("No complete before/after observations found.")

# Classify the surrounding operating-state transitions.
result["transition"] = (
    result["before_COMP"].astype(str)
    + "/" + result["before_DV"].astype(str)
    + " -> "
    + result["after_COMP"].astype(str)
    + "/" + result["after_DV"].astype(str)
)

output = Path("metropt-industrial-ai/outputs")
destination = output / "metropt_exception_transition_analysis.csv"
result.to_csv(destination, index=False)

print("\n========== TRANSITION ANALYSIS ==========")
print(f"Total contradictory observations: {len(exceptions)}")
print(f"Observations with both neighbours available: {len(result)}")

print("\nTransition patterns:")
print(result["transition"].value_counts().to_string())

print("\nMotor-current change:")
print(
    (result["motor_after"] - result["motor_before"])
    .describe()
    .to_string()
)

print("\nTP2 change:")
print(
    (result["TP2_after"] - result["TP2_before"])
    .describe()
    .to_string()
)

print("\nFirst 20 analysed observations:")
print(
    result[
        [
            "timestamp",
            "transition",
            "motor_before",
            "motor_exception",
            "motor_after",
            "TP2_before",
            "TP2_exception",
            "TP2_after",
        ]
    ].head(20).to_string(index=False)
)

print(f"\nSaved analysis: {destination}")
