"""Transformers baseline runner for embedding and reranking."""

import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from bench.gpu_monitor import sample_gpu_stats
from bench.metrics import BenchmarkResult, LatencyTimer
from bench.workload import RerankerPair, get_batches


class TransformersEmbedder:
    def __init__(self, model_path: str, pooling: str = "cls"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.float16, trust_remote_code=True
        ).eval()
        self.model.to("cuda")
        self.pooling = pooling

    def embed(self, texts: list[str]) -> list[list[float]]:
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, max_length=8192, return_tensors="pt"
        ).to("cuda")
        with torch.no_grad():
            outputs = self.model(**encoded)
        if self.pooling == "cls":
            embeddings = outputs.last_hidden_state[:, 0, :]
        else:  # mean pooling
            mask = encoded["attention_mask"].unsqueeze(-1)
            embeddings = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
        return embeddings.cpu().float().tolist()


class TransformersReranker:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, torch_dtype=torch.float16, trust_remote_code=True
        ).eval()
        self.model.to("cuda")
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def score(self, query: str, documents: list[str]) -> list[float]:
        pairs = [f"{query} [SEP] {doc}" for doc in documents]
        encoded = self.tokenizer(
            pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to("cuda")
        with torch.no_grad():
            logits = self.model(**encoded).logits.squeeze(-1)
        scores = torch.sigmoid(logits)
        return scores.cpu().float().tolist()


def run_embedding_bench(
    model_path: str,
    model_name: str,
    texts: list[str],
    batch_size: int,
    text_length: str,
    pooling: str = "cls",
    warmup_rounds: int = 3,
    measure_rounds: int = 3,
) -> BenchmarkResult:
    embedder = TransformersEmbedder(model_path, pooling=pooling)

    # Warmup
    for batch in get_batches(texts[:batch_size], batch_size):
        embedder.embed(batch)
    for _ in range(warmup_rounds - 1):
        for batch in get_batches(texts[:batch_size], batch_size):
            embedder.embed(batch)

    torch.cuda.synchronize()

    # Measure
    latencies = []
    for _ in range(measure_rounds):
        for batch in get_batches(texts[:batch_size], batch_size):
            with LatencyTimer(latencies) as t:  # noqa: F841
                embedder.embed(batch)
        torch.cuda.synchronize()

    gpu = sample_gpu_stats()

    del embedder
    torch.cuda.empty_cache()

    return BenchmarkResult(
        model_name=model_name,
        framework="transformers",
        test_type="embedding",
        text_length=text_length,
        batch_size=batch_size,
        latencies_ms=latencies,
        total_requests=len(latencies),
        gpu_memory_used_mb=gpu.memory_used_mb,
        gpu_utilization_pct=gpu.utilization_pct,
    )


def run_reranker_bench(
    model_path: str,
    model_name: str,
    pairs: list[RerankerPair],
    batch_size: int,
    text_length: str,
    warmup_rounds: int = 3,
    measure_rounds: int = 3,
) -> BenchmarkResult:
    reranker = TransformersReranker(model_path)

    # Warmup
    for pair in pairs[:1]:
        reranker.score(pair.query, pair.documents[:batch_size])
    for _ in range(warmup_rounds - 1):
        for pair in pairs[:1]:
            reranker.score(pair.query, pair.documents[:batch_size])

    torch.cuda.synchronize()

    # Measure
    latencies = []
    for _ in range(measure_rounds):
        for pair in pairs[:3]:  # test on 3 pairs per round
            docs = pair.documents[:batch_size]
            with LatencyTimer(latencies) as t:  # noqa: F841
                reranker.score(pair.query, docs)
        torch.cuda.synchronize()

    gpu = sample_gpu_stats()

    del reranker
    torch.cuda.empty_cache()

    return BenchmarkResult(
        model_name=model_name,
        framework="transformers",
        test_type="reranker",
        text_length=text_length,
        batch_size=batch_size,
        latencies_ms=latencies,
        total_requests=len(latencies),
        gpu_memory_used_mb=gpu.memory_used_mb,
        gpu_utilization_pct=gpu.utilization_pct,
    )
