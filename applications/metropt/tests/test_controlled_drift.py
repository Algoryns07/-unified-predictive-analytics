import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from controlled_drift import generate_synthetic_stream


def test_stream_shapes_and_finite_values():

    result = generate_synthetic_stream(
        1000,
        "no_drift",
        500,
        42,
    )

    assert result["X"].shape == (1000, 5)
    assert len(result["y"]) == 1000
    assert np.isfinite(result["X"].to_numpy()).all()
    assert np.isfinite(result["y"]).all()


def test_no_drift_has_no_injected_observations():

    result = generate_synthetic_stream(
        1000,
        "no_drift",
        500,
        42,
    )

    assert not result["injected"].any()


def test_covariate_drift_has_known_injection_point():

    result = generate_synthetic_stream(
        1000,
        "covariate_drift",
        500,
        42,
    )

    assert not result["injected"][:500].any()
    assert result["injected"][500:].all()


def test_covariate_drift_changes_feature_distribution():

    result = generate_synthetic_stream(
        4000,
        "covariate_drift",
        2000,
        42,
    )

    X = result["X"]

    before = X.iloc[:2000]
    after = X.iloc[2000:]

    assert after["feature_0"].mean() > before["feature_0"].mean() + 1.0
    assert after["feature_1"].mean() < before["feature_1"].mean() - 0.5


def test_concept_drift_preserves_input_distribution():

    normal = generate_synthetic_stream(
        4000,
        "no_drift",
        2000,
        42,
    )

    concept = generate_synthetic_stream(
        4000,
        "concept_drift",
        2000,
        42,
    )

    assert np.array_equal(
        normal["X"].to_numpy(),
        concept["X"].to_numpy(),
    )


def test_concept_drift_changes_target_relationship():

    normal = generate_synthetic_stream(
        4000,
        "no_drift",
        2000,
        42,
    )

    concept = generate_synthetic_stream(
        4000,
        "concept_drift",
        2000,
        42,
    )

    assert np.allclose(
        normal["y"][:2000],
        concept["y"][:2000],
    )

    assert not np.allclose(
        normal["y"][2000:],
        concept["y"][2000:],
    )


def test_same_seed_reproduces_same_stream():

    first = generate_synthetic_stream(
        1000,
        "concept_drift",
        500,
        17,
    )

    second = generate_synthetic_stream(
        1000,
        "concept_drift",
        500,
        17,
    )

    assert np.array_equal(
        first["X"].to_numpy(),
        second["X"].to_numpy(),
    )

    assert np.array_equal(
        first["y"],
        second["y"],
    )


def test_invalid_scenario_rejected():

    with pytest.raises(ValueError):
        generate_synthetic_stream(
            1000,
            "unknown",
            500,
            42,
        )


def test_invalid_injection_index_rejected():

    with pytest.raises(ValueError):
        generate_synthetic_stream(
            1000,
            "concept_drift",
            1000,
            42,
        )
