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
