import importlib.util
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "run_adaptive_calibration.py"
)

SPEC = importlib.util.spec_from_file_location(
    "adaptive_calibration",
    MODULE_PATH
)

MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_conformal_quantile_uses_finite_sample_order_statistic():
    residuals = np.arange(1, 20, dtype=float)

    # n=19, alpha=.10: ceil(20*.90) = 18.
    assert MODULE.conformal_quantile(residuals, 0.10) == 18.0

    # n=19, alpha=.05: ceil(20*.95) = 19.
    assert MODULE.conformal_quantile(residuals, 0.05) == 19.0


def test_conformal_quantile_returns_infinity_when_order_statistic_exceeds_n():
    residuals = np.arange(1, 10, dtype=float)

    assert np.isinf(
        MODULE.conformal_quantile(residuals, 0.05)
    )


@pytest.mark.parametrize(
    "residuals, alpha",
    [
        ([], 0.10),
        ([1.0, np.nan], 0.10),
        ([1.0, np.inf], 0.10),
        ([1.0, -1.0], 0.10),
        ([1.0, 2.0], 0.0),
        ([1.0, 2.0], 1.0),
        ([1.0, 2.0], -0.1),
    ]
)
def test_conformal_quantile_rejects_invalid_inputs(residuals, alpha):
    if np.asarray(residuals).ndim == 1 and len(residuals) > 0:
        if np.isfinite(np.asarray(residuals, dtype=float)).all():
            if (np.asarray(residuals, dtype=float) < 0).any():
                with pytest.raises(ValueError):
                    MODULE.conformal_quantile(residuals, alpha)
                return

    with pytest.raises(ValueError):
        MODULE.conformal_quantile(residuals, alpha)


def test_rolling_buffers_initialize_with_latest_calibration_residuals():
    residuals = np.arange(1, 21, dtype=float)

    buffers = MODULE.initialize_rolling_buffers(
        residuals,
        [5, 10]
    )

    assert list(buffers[5]) == [16, 17, 18, 19, 20]
    assert list(buffers[10]) == list(range(11, 21))


def test_rolling_buffers_never_exceed_configured_size():
    buffers = MODULE.initialize_rolling_buffers(
        np.arange(1, 21, dtype=float),
        [5]
    )

    for residual in [21, 22, 23]:
        buffers[5].append(residual)

    assert len(buffers[5]) == 5
    assert list(buffers[5]) == [19, 20, 21, 22, 23]


def test_current_observation_cannot_influence_its_own_interval():
    initial = np.arange(1, 101, dtype=float)

    buffers = MODULE.initialize_rolling_buffers(
        initial,
        [100]
    )

    alphas = {90: 0.10, 95: 0.05}

    quantiles = {
        ("Fixed", level): MODULE.conformal_quantile(initial, alpha)
        for level, alpha in alphas.items()
    }

    quantiles.update(
        MODULE.refresh_rolling_quantiles(buffers, alphas)
    )

    previous_q = quantiles[("Rolling_100", 90)]

    intervals = MODULE.prequential_interval_step(
        prediction=10.0,
        actual=10000.0,
        quantiles=quantiles,
        buffers=buffers,
        alphas=alphas
    )

    # The current, extreme residual is not used in its own interval.
    assert intervals[("Rolling_100", 90)] == (
        10.0 - previous_q,
        10.0 + previous_q
    )

    # It becomes available for future predictions.
    assert buffers[100][-1] == 9990.0


def test_fixed_calibration_quantile_remains_unchanged_after_updates():
    initial = np.arange(1, 101, dtype=float)

    buffers = MODULE.initialize_rolling_buffers(
        initial,
        [100]
    )

    alphas = {90: 0.10, 95: 0.05}

    fixed_q = MODULE.conformal_quantile(initial, 0.10)

    quantiles = {
        ("Fixed", 90): fixed_q,
        ("Fixed", 95): MODULE.conformal_quantile(initial, 0.05)
    }

    quantiles.update(
        MODULE.refresh_rolling_quantiles(buffers, alphas)
    )

    for actual in [1000.0, 2000.0, 3000.0]:
        MODULE.prequential_interval_step(
            prediction=10.0,
            actual=actual,
            quantiles=quantiles,
            buffers=buffers,
            alphas=alphas
        )

    assert quantiles[("Fixed", 90)] == fixed_q


def test_rolling_quantile_refresh_uses_updated_buffer():
    initial = np.arange(1, 101, dtype=float)

    buffers = MODULE.initialize_rolling_buffers(
        initial,
        [100]
    )

    alphas = {90: 0.10}

    before = MODULE.refresh_rolling_quantiles(
        buffers,
        alphas
    )[("Rolling_100", 90)]

    for actual in [10000.0, 12000.0, 15000.0]:
        MODULE.prequential_interval_step(
            prediction=0.0,
            actual=actual,
            quantiles={
                ("Fixed", 90): before,
                ("Rolling_100", 90): before
            },
            buffers=buffers,
            alphas=alphas
        )

    after = MODULE.refresh_rolling_quantiles(
        buffers,
        alphas
    )[("Rolling_100", 90)]

    assert after > before


def test_prequential_step_rejects_non_finite_observations():
    buffers = MODULE.initialize_rolling_buffers(
        np.arange(1, 101, dtype=float),
        [100]
    )

    alphas = {90: 0.10}
    quantiles = {
        ("Fixed", 90): 10.0,
        ("Rolling_100", 90): 10.0
    }

    with pytest.raises(ValueError):
        MODULE.prequential_interval_step(
            prediction=10.0,
            actual=np.nan,
            quantiles=quantiles,
            buffers=buffers,
            alphas=alphas
        )
