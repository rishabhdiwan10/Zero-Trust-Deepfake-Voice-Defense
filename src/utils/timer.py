"""
src.utils.timer
================
Latency measurement and profiling utilities for the Zero-Trust Deepfake
Voice Defense System.

Provides:
  - ``Timer`` context manager for measuring elapsed time
  - ``timeit`` decorator for annotating functions with latency logging
  - ``LatencyTracker`` for accumulating per-stage measurements across requests
"""

from __future__ import annotations

import functools
import logging
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class Timer:
    """
    Context manager for measuring elapsed wall-clock time.

    Usage::

        with Timer("cnn_inference") as t:
            result = model.predict(features)
        print(f"Elapsed: {t.elapsed_ms:.2f} ms")
    """

    def __init__(self, label: str = "", log: bool = False) -> None:
        self.label = label
        self.log = log
        self._start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
        if self.log:
            logger.debug(
                "Timer [%s]: %.2f ms",
                self.label or "unnamed",
                self.elapsed_ms,
            )


def timeit(
    label: Optional[str] = None,
    log_level: int = logging.DEBUG,
) -> Callable:
    """
    Decorator that measures and logs function execution time.

    Parameters
    ----------
    label : str | None
        Custom label for the log message. Defaults to the function name.
    log_level : int
        Logging level for the timing message.

    Usage::

        @timeit("cnn_forward_pass")
        def predict(features):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            _label = label or func.__qualname__
            logger.log(log_level, "timeit [%s]: %.2f ms", _label, elapsed_ms)
            return result

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = await func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            _label = label or func.__qualname__
            logger.log(log_level, "timeit [%s]: %.2f ms", _label, elapsed_ms)
            return result

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


class LatencyTracker:
    """
    Accumulate per-stage latency measurements across multiple requests.

    Useful for generating aggregate statistics during benchmarking.

    Usage::

        tracker = LatencyTracker()
        tracker.record("cnn_inference", 45.2)
        tracker.record("cnn_inference", 48.1)
        print(tracker.summary())
    """

    def __init__(self) -> None:
        self._data: Dict[str, List[float]] = defaultdict(list)

    def record(self, stage: str, elapsed_ms: float) -> None:
        """Record a latency sample for a pipeline stage."""
        self._data[stage].append(elapsed_ms)

    def record_from_state(self, stage_latencies: Dict[str, float]) -> None:
        """Bulk-record latencies from a pipeline state dict."""
        for stage, ms in stage_latencies.items():
            self.record(stage, ms)

    def summary(self) -> Dict[str, Dict[str, float]]:
        """
        Return per-stage summary statistics (mean, p50, p95, p99, max).
        """
        import statistics

        result = {}
        for stage, samples in self._data.items():
            if not samples:
                continue
            sorted_s = sorted(samples)
            n = len(sorted_s)
            result[stage] = {
                "count": n,
                "mean_ms": round(statistics.mean(sorted_s), 2),
                "p50_ms": round(sorted_s[int(n * 0.50)], 2),
                "p95_ms": round(sorted_s[int(n * 0.95)], 2),
                "p99_ms": round(sorted_s[min(int(n * 0.99), n - 1)], 2),
                "max_ms": round(sorted_s[-1], 2),
            }
        return result

    def reset(self) -> None:
        """Clear all recorded measurements."""
        self._data.clear()
