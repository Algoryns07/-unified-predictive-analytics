import numpy as np
import pandas as pd


FEATURES = [
    "feature_0",
    "feature_1",
    "feature_2",
    "feature_3",
    "feature_4",
]


def generate_synthetic_stream(
    n_observations,
    scenario,
    injection_index,
    random_state,
):
    if (
        isinstance(n_observations, bool)
        or not isinstance(n_observations, int)
        or n_observations < 1
    ):
        raise ValueError("n_observations must be a positive integer.")

    if scenario not in {
        "no_drift",
        "covariate_drift",
        "concept_drift",
    }:
        raise ValueError("Unsupported scenario.")

    if (
        isinstance(injection_index, bool)
        or not isinstance(injection_index, int)
        or not 0 < injection_index < n_observations
    ):
        raise ValueError(
            "injection_index must be between 0 and n_observations."
        )

    rng = np.random.default_rng(random_state)

    X = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(n_observations, len(FEATURES)),
    )

    injected = np.zeros(n_observations, dtype=bool)

    if scenario != "no_drift":
        injected[injection_index:] = True

    if scenario == "covariate_drift":

        # Change input distributions only.
        # The target-generating relationship remains unchanged.

        X[injection_index:, 0] += 1.5
        X[injection_index:, 1] -= 1.0

    base_target = (
        2.0 * X[:, 0]
        - 1.5 * X[:, 1]
        + 0.8 * (X[:, 2] ** 2)
        + 0.5 * X[:, 3] * X[:, 4]
    )

    y = base_target.copy()

    if scenario == "concept_drift":

        # Change the target-generating relationship.
        # Input feature distributions remain unchanged.

        changed_target = (
            -1.5 * X[:, 0]
            + 2.0 * X[:, 1]
            - 0.8 * (X[:, 2] ** 2)
            + 0.5 * X[:, 3] * X[:, 4]
        )

        y[injection_index:] = changed_target[injection_index:]

    noise = rng.normal(
        loc=0.0,
        scale=0.2,
        size=n_observations,
    )

    y = y + noise

    frame = pd.DataFrame(
        X,
        columns=FEATURES,
    )

    return {
        "X": frame,
        "y": y,
        "injected": injected,
        "injection_index": injection_index,
        "scenario": scenario,
    }
