from pathlib import Path

import numpy as np
import pandas as pd

from src.drift_detection import compare_feature


def run_comparison():
    rng = np.random.default_rng(42)
    reference = rng.normal(0, 1, 2000)

    scenarios = {
        "no_shift": 0.0,
        "small_shift": 0.2,
        "moderate_shift": 0.5,
        "large_shift": 1.0,
        "extreme_shift": 2.0,
    }

    rows = []

    for scenario, shift in scenarios.items():
        current = rng.normal(shift, 1, 2000)
        result = compare_feature(reference, current)

        rows.append({
            "scenario": scenario,
            "shift_magnitude": shift,
            "psi": result["psi"],
            "ks_statistic": result["ks_statistic"],
            "ks_p_value": result["ks_p_value"],
            "wasserstein": result["wasserstein"],
            "normalized_wasserstein": result["normalized_wasserstein"],
        })

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    results = pd.DataFrame(rows)
    results.to_csv(
        output_dir / "drift_metric_comparison.csv",
        index=False
    )

    print("\nDRIFT METRIC COMPARISON")
    print(results.to_string(index=False))
    print("\nSaved: outputs/drift_metric_comparison.csv")


if __name__ == "__main__":
    run_comparison()
