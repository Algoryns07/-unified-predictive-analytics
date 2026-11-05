import tracemalloc

import pytest

from src.retraining_cost import measure_callable


def test_measure_callable_returns_function_result():
    result, measurements = measure_callable(lambda x: x * 2, 5)

    assert result == 10
    assert measurements["wall_time_seconds"] >= 0
    assert measurements["process_cpu_time_seconds"] >= 0
    assert measurements["peak_traced_memory_mb"] >= 0


def test_measure_callable_supports_keyword_arguments():
    def add(a, b):
        return a + b

    result, measurements = measure_callable(add, a=10, b=20)

    assert result == 30
    assert measurements["wall_time_seconds"] >= 0


def test_measure_callable_rejects_existing_memory_trace():
    tracemalloc.start()

    try:
        with pytest.raises(RuntimeError):
            measure_callable(lambda: 42)
    finally:
        tracemalloc.stop()


def test_measure_callable_stops_memory_tracing_after_execution():
    assert not tracemalloc.is_tracing()

    measure_callable(lambda: sum(range(100)))

    assert not tracemalloc.is_tracing()

def test_measure_repeated_collects_all_runs():
    from src.retraining_cost import measure_repeated

    results, summary = measure_repeated(lambda: sum(range(100)), 3)

    assert len(results) == 3
    assert len(summary) == 3
    assert summary["wall_time_seconds"]["mean"] >= 0
    assert summary["process_cpu_time_seconds"]["mean"] >= 0
    assert summary["peak_traced_memory_mb"]["mean"] >= 0


def test_measure_repeated_rejects_zero_repetitions():
    from src.retraining_cost import measure_repeated

    with pytest.raises(ValueError):
        measure_repeated(lambda: 42, 0)
