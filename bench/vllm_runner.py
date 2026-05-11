"""vLLM programmatic API runner for embedding and reranking."""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from vllm import LLM

from bench.gpu_monitor import sample_gpu_stats
from bench.metrics import BenchmarkResult, LatencyTimer
from bench.workload import RerankerPair, get_batches

# Chat template for Qwen3-Reranker models
QWEN3_RERANKER_TEMPLATE = """\
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
{document}<|im_end|>
"""


def _is_qwen_reranker(model_name: str) -> bool:
    return "Qwen3-Reranker" in model_name


class VLLMEmbedder:
    def __init__(self, model_path: str):
        self.llm = LLM(
            model=model_path,
            runner="pooling",
            enforce_eager=True,
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        outputs = self.llm.embed(texts)
        return [o.outputs.embedding for o in outputs]


class VLLMReranker:
    def __init__(self, model_path: str, model_name: str):
        kwargs = dict(
            model=model_path,
            runner="pooling",
            enforce_eager=True,
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
        )
        if _is_qwen_reranker(model_name):
            kwargs["chat_template"] = QWEN3_RERANKER_TEMPLATE
        self.llm = LLM(**kwargs)
        self.model_name = model_name

    def score(self, query: str, documents: list[str]) -> list[float]:
        outputs = self.llm.score(query, documents)
        return [o.outputs.score for o in outputs]


def create_embedder(model_path: str) -> VLLMEmbedder:
    return VLLMEmbedder(model_path)


def create_reranker(model_path: str, model_name: str) -> VLLMReranker:
    return VLLMReranker(model_path, model_name)


def run_embedding_bench(
    embedder: VLLMEmbedder,
    model_name: str,
    texts: list[str],
    batch_size: int,
    text_length: str,
    warmup_rounds: int = 3,
    measure_rounds: int = 3,
) -> BenchmarkResult:
    # Warmup
    for _ in range(warmup_rounds):
        for batch in get_batches(texts[:batch_size], batch_size):
            embedder.embed(batch)

    # Measure
    latencies = []
    for _ in range(measure_rounds):
        for batch in get_batches(texts[:batch_size], batch_size):
            with LatencyTimer(latencies) as t:  # noqa: F841
                embedder.embed(batch)

    gpu = sample_gpu_stats()

    return BenchmarkResult(
        model_name=model_name,
        framework="vllm",
        test_type="embedding",
        text_length=text_length,
        batch_size=batch_size,
        latencies_ms=latencies,
        total_requests=len(latencies),
        gpu_memory_used_mb=gpu.memory_used_mb,
        gpu_utilization_pct=gpu.utilization_pct,
    )


def run_reranker_bench(
    reranker: VLLMReranker,
    model_name: str,
    pairs: list[RerankerPair],
    batch_size: int,
    text_length: str,
    warmup_rounds: int = 3,
    measure_rounds: int = 3,
) -> BenchmarkResult:
    # Warmup
    for _ in range(warmup_rounds):
        for pair in pairs[:1]:
            reranker.score(pair.query, pair.documents[:batch_size])

    # Measure
    latencies = []
    for _ in range(measure_rounds):
        for pair in pairs[:3]:  # test on 3 pairs per round
            docs = pair.documents[:batch_size]
            with LatencyTimer(latencies) as t:  # noqa: F841
                reranker.score(pair.query, docs)

    gpu = sample_gpu_stats()

    return BenchmarkResult(
        model_name=model_name,
        framework="vllm",
        test_type="reranker",
        text_length=text_length,
        batch_size=batch_size,
        latencies_ms=latencies,
        total_requests=len(latencies),
        gpu_memory_used_mb=gpu.memory_used_mb,
        gpu_utilization_pct=gpu.utilization_pct,
    )
