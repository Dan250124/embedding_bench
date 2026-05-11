"""TEI HTTP client for embedding and reranking."""

import asyncio
import time

import aiohttp

from bench.gpu_monitor import sample_gpu_stats
from bench.metrics import BenchmarkResult, LatencyTimer
from bench.workload import RerankerPair, get_batches


class TEIClient:
    def __init__(self, base_url: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    async def embed(self, session: aiohttp.ClientSession, texts: list[str]) -> list:
        async with session.post(
            f"{self.base_url}/embed",
            json={"inputs": texts, "truncate": True},
        ) as resp:
            return await resp.json()

    async def rerank(
        self, session: aiohttp.ClientSession, query: str, documents: list[str]
    ) -> list:
        async with session.post(
            f"{self.base_url}/rerank",
            json={"query": query, "texts": documents, "truncate": True},
        ) as resp:
            return await resp.json()


async def run_embedding_bench(
    base_url: str,
    model_name: str,
    texts: list[str],
    batch_size: int,
    text_length: str,
    warmup_rounds: int = 3,
    measure_rounds: int = 3,
) -> BenchmarkResult:
    client = TEIClient(base_url)

    async with aiohttp.ClientSession(timeout=client.timeout) as session:
        # Warmup
        for _ in range(warmup_rounds):
            for batch in get_batches(texts[:batch_size], batch_size):
                await client.embed(session, batch)

        # Measure
        latencies = []
        for _ in range(measure_rounds):
            for batch in get_batches(texts[:batch_size], batch_size):
                with LatencyTimer(latencies) as t:  # noqa: F841
                    await client.embed(session, batch)

    gpu = sample_gpu_stats()

    return BenchmarkResult(
        model_name=model_name,
        framework="tei",
        test_type="embedding",
        text_length=text_length,
        batch_size=batch_size,
        latencies_ms=latencies,
        total_requests=len(latencies),
        gpu_memory_used_mb=gpu.memory_used_mb,
        gpu_utilization_pct=gpu.utilization_pct,
    )


async def run_reranker_bench(
    base_url: str,
    model_name: str,
    pairs: list[RerankerPair],
    batch_size: int,
    text_length: str,
    warmup_rounds: int = 3,
    measure_rounds: int = 3,
) -> BenchmarkResult:
    client = TEIClient(base_url)

    async with aiohttp.ClientSession(timeout=client.timeout) as session:
        # Warmup
        for _ in range(warmup_rounds):
            for pair in pairs[:1]:
                await client.rerank(session, pair.query, pair.documents[:batch_size])

        # Measure
        latencies = []
        for _ in range(measure_rounds):
            for pair in pairs[:3]:
                docs = pair.documents[:batch_size]
                with LatencyTimer(latencies) as t:  # noqa: F841
                    await client.rerank(session, pair.query, docs)

    gpu = sample_gpu_stats()

    return BenchmarkResult(
        model_name=model_name,
        framework="tei",
        test_type="reranker",
        text_length=text_length,
        batch_size=batch_size,
        latencies_ms=latencies,
        total_requests=len(latencies),
        gpu_memory_used_mb=gpu.memory_used_mb,
        gpu_utilization_pct=gpu.utilization_pct,
    )


async def _send_requests(
    client: TEIClient,
    session: aiohttp.ClientSession,
    texts: list[str],
    batch_size: int,
    duration_sec: float,
) -> tuple[int, float]:
    """Send concurrent requests for duration_sec, return (total_requests, elapsed)."""
    total = 0
    start = time.perf_counter()
    sem = asyncio.Semaphore(16)  # limit concurrent connections

    async def worker():
        nonlocal total
        while time.perf_counter() - start < duration_sec:
            async with sem:
                batch = texts[:batch_size]
                await client.embed(session, batch)
                total += 1

    workers = [asyncio.create_task(worker()) for _ in range(16)]
    await asyncio.gather(*workers)
    elapsed = time.perf_counter() - start
    return total, elapsed


async def run_throughput_test(
    base_url: str,
    model_name: str,
    texts: list[str],
    batch_size: int,
    concurrency: int,
    duration_sec: float = 30.0,
) -> BenchmarkResult:
    client = TEIClient(base_url)

    async with aiohttp.ClientSession(timeout=client.timeout) as session:
        # Warmup
        for _ in range(3):
            await client.embed(session, texts[:batch_size])

        total, elapsed = await _send_requests(
            client, session, texts, batch_size, duration_sec
        )

    gpu = sample_gpu_stats()

    return BenchmarkResult(
        model_name=model_name,
        framework="tei",
        test_type="embedding",
        text_length="medium",
        batch_size=batch_size,
        concurrency=concurrency,
        total_requests=total,
        duration_sec=elapsed,
        gpu_memory_used_mb=gpu.memory_used_mb,
        gpu_utilization_pct=gpu.utilization_pct,
    )
