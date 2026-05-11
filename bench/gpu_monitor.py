"""GPU stats monitoring via nvidia-smi."""

import asyncio
import subprocess
from dataclasses import dataclass


@dataclass
class GPUStats:
    memory_used_mb: float
    memory_total_mb: float
    utilization_pct: float


def sample_gpu_stats() -> GPUStats:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    parts = result.stdout.strip().split(", ")
    if len(parts) < 3:
        return GPUStats(0.0, 0.0, 0.0)
    return GPUStats(
        memory_used_mb=float(parts[0]),
        memory_total_mb=float(parts[1]),
        utilization_pct=float(parts[2]),
    )


async def monitor_gpu(interval: float = 1.0) -> GPUStats:
    """Sample GPU stats in background, return peak memory usage."""
    peak_memory = 0.0
    peak_util = 0.0
    total_memory = 0.0

    async def _sample():
        nonlocal peak_memory, peak_util, total_memory
        loop = asyncio.get_event_loop()
        while True:
            try:
                stats = await loop.run_in_executor(None, sample_gpu_stats)
                peak_memory = max(peak_memory, stats.memory_used_mb)
                peak_util = max(peak_util, stats.utilization_pct)
                total_memory = stats.memory_total_mb
            except Exception:
                pass
            await asyncio.sleep(interval)

    task = asyncio.create_task(_sample())
    try:
        yield GPUStats(peak_memory, total_memory, peak_util)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# Make it usable as async context manager
import types
monitor_gpu.__enter__ = lambda self: None  # type: ignore[attr-defined]
monitor_gpu.__exit__ = lambda self, *a: None  # type: ignore[attr-defined]
