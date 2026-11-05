import time
import tracemalloc
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


def measure_callable(
    function: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> tuple[T, dict[str, float]]:
    """Measure execution time, CPU time and peak traced Python memory."""

    if tracemalloc.is_tracing():
        raise RuntimeError("tracemalloc is already active.")

    tracemalloc.start()

    try:
        wall_start = time.perf_counter()
        cpu_start = time.process_time()

        result = function(*args, **kwargs)

        wall_time = time.perf_counter() - wall_start
        cpu_time = time.process_time() - cpu_start
        _, peak_memory = tracemalloc.get_traced_memory()

    finally:
        tracemalloc.stop()

    measurements = {
        "wall_time_seconds": float(wall_time),
        "process_cpu_time_seconds": float(cpu_time),
        "peak_traced_memory_mb": float(peak_memory / (1024 * 1024)),
    }

    return result, measurements
import statistics


def measure_repeated(
    function,
    repetitions=5,
    *args,
    **kwargs,
):
    """Measure repeated executions and summarize resource consumption."""

    if not isinstance(repetitions, int) or isinstance(repetitions, bool):
        raise TypeError("repetitions must be an integer.")

    if repetitions < 1:
        raise ValueError("repetitions must be at least 1.")

    results = []
    measurements = []

    for _ in range(repetitions):
        result, metrics = measure_callable(function, *args, **kwargs)
        results.append(result)
        measurements.append(metrics)

    summary = {}

    for metric in measurements[0]:
        values = [item[metric] for item in measurements]

        summary[metric] = {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std": statistics.pstdev(values),
        }

    return results, summary


def export_measurement_summary(summary, output_path):
    """Export repeated resource measurements to a CSV file."""
    import csv
    from pathlib import Path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["metric", "mean", "median", "std"]

    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for metric, values in summary.items():
            writer.writerow({
                "metric": metric,
                "mean": values["mean"],
                "median": values["median"],
                "std": values["std"],
            })

    return output_path
