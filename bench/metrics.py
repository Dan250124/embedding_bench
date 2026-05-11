"""Metrics collection and computation."""

import statistics
import time
from dataclasses import dataclass, field


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    model_name: str
    framework: str
    test_type: str  # "embedding" or "reranker"
    text_length: str  # "short", "medium", "long"
    batch_size: int
    concurrency: int = 1
    # Latency (ms)
    latencies_ms: list[float] = field(default_factory=list)
    # Throughput
    total_requests: int = 0
    duration_sec: float = 0.0
    # GPU
    gpu_memory_used_mb: float = 0.0
    gpu_utilization_pct: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p50_latency_ms(self) -> float:
        return _percentile(self.latencies_ms, 50)

    @property
    def p95_latency_ms(self) -> float:
        return _percentile(self.latencies_ms, 95)

    @property
    def p99_latency_ms(self) -> float:
        return _percentile(self.latencies_ms, 99)

    @property
    def requests_per_sec(self) -> float:
        if self.duration_sec > 0:
            return self.total_requests / self.duration_sec
        return 0.0


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100.0
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


class LatencyTimer:
    """Context manager to time a single operation."""

    def __init__(self, recorder: list[float]):
        self.recorder = recorder
        self._start = 0.0
        self._end = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self._end = time.perf_counter()
        self.recorder.append((self._end - self._start) * 1000.0)  # ms
