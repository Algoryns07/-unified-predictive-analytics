from __future__ import annotations

from collections import deque
from math import ceil

import numpy as np


def conformal_quantile(residuals, alpha):
    """
    Calculate a finite-sample split-conformal residual quantile.

    Returns infinity when the requested order statistic
    exceeds the available calibration sample size.
    """
    values = np.asarray(residuals, dtype=float)

    if values.ndim != 1 or values.size == 0:
        raise ValueError("Residuals must be a non-empty one-dimensional array.")

    if not np.isfinite(values).all():
        raise ValueError("Residuals must contain only finite values.")

    if (values < 0).any():
        raise ValueError("Residuals must be non-negative.")

    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("Alpha must be strictly between 0 and 1.")

    n = len(values)
    order = ceil((n + 1) * (1 - alpha))

    if order > n:
        return float("inf")

    return float(np.sort(values)[order - 1])


def initialize_rolling_buffers(residuals, window_sizes):
    """Initialize bounded rolling residual buffers."""
    values = np.asarray(residuals, dtype=float)

    if values.ndim != 1 or values.size == 0:
        raise ValueError("Residuals must be a non-empty one-dimensional array.")

    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Residuals must be finite and non-negative.")

    sizes = list(window_sizes)

    if not sizes or any(
        isinstance(size, bool) or not isinstance(size, int) or size <= 0
        for size in sizes
    ):
        raise ValueError("Window sizes must be positive integers.")

    if len(set(sizes)) != len(sizes):
        raise ValueError("Window sizes must be unique.")

    return {
        size: deque(values[-size:].tolist(), maxlen=size)
        for size in sizes
    }


def refresh_rolling_quantiles(buffers, alphas):
    """Calculate quantiles from the current rolling buffers."""
    if not buffers or not alphas:
        raise ValueError("Buffers and alpha levels cannot be empty.")

    result = {}

    for size, buffer in buffers.items():
        if len(buffer) == 0:
            raise ValueError("Rolling buffers cannot be empty.")

        for level, alpha in alphas.items():
            result[(f"Rolling_{size}", level)] = conformal_quantile(
                list(buffer), alpha
            )

    return result


def prequential_interval_step(
    prediction,
    actual,
    quantiles,
    buffers,
    alphas,
):
    """
    Generate prediction intervals before incorporating the
    current observation's residual into future calibration.
    """
    prediction = float(prediction)
    actual = float(actual)

    if not np.isfinite(prediction) or not np.isfinite(actual):
        raise ValueError("Prediction and actual value must be finite.")

    if not quantiles or not buffers or not alphas:
        raise ValueError("Quantiles, buffers and alpha levels are required.")

    residual = abs(actual - prediction)

    intervals = {}

    # Generate intervals using only previously available
    # calibration residuals.
    for key, quantile in quantiles.items():
        intervals[key] = (
            prediction - quantile,
            prediction + quantile,
        )

    # Make the current residual available only for future steps.
    for buffer in buffers.values():
        buffer.append(residual)

    return intervals
