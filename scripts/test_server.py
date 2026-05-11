"""Quick smoke test for TEI / vLLM embedding server.

Supports both TEI (HuggingFace TGI) and vLLM (OpenAI-compatible) backends.

Usage:
    # TEI
    uv run python scripts/test_tei.py --base-url http://localhost:8080 --backend tei --type embedding
    # vLLM
    uv run python scripts/test_tei.py --base-url http://localhost:8080 --backend vllm --type embedding
"""

import argparse
import asyncio
import json
import statistics
import time

import aiohttp


# ── Sample data ──────────────────────────────────────────────────────────────

SHORT_TEXTS = [
    "What is machine learning?",
    "How to deploy a model to production?",
    "Explain the difference between CNN and RNN.",
    "What is attention mechanism?",
    "Best practices for API design.",
]

MEDIUM_TEXTS = [
    "Machine learning is a subset of artificial intelligence that focuses on building systems "
    "that learn from data. Unlike traditional programming where rules are explicitly coded, "
    "ML algorithms identify patterns in data and make decisions with minimal human intervention. "
    "Supervised learning, unsupervised learning, and reinforcement learning are the three main "
    "paradigms. Each has its own strengths and is suited for different types of problems, from "
    "image classification to game playing.",
    "Deploying machine learning models to production requires careful consideration of latency, "
    "throughput, and resource utilization. Containerization with Docker and orchestration with "
    "Kubernetes have become standard practices. Model serving frameworks like TensorFlow Serving, "
    "TorchServe, and Triton Inference Server provide optimized inference pipelines. Key metrics "
    "to monitor include request latency percentiles (P50, P95, P99), GPU utilization, and "
    "memory consumption. Auto-scaling policies should be based on these metrics to handle "
    "variable traffic patterns efficiently.",
    "The attention mechanism, introduced in the 'Attention Is All You Need' paper, revolutionized "
    "natural language processing. Self-attention allows each position in a sequence to attend to "
    "all other positions, capturing long-range dependencies without recurrence. Multi-head attention "
    "enables the model to jointly attend to information from different representation subspaces. "
    "This architecture forms the foundation of transformer models like BERT, GPT, and their "
    "successors, which have achieved state-of-the-art results across numerous NLP benchmarks.",
]

LONG_TEXTS = [
    "Natural language processing (NLP) has undergone a remarkable transformation in recent years, "
    "driven primarily by the advent of large language models and the transformer architecture. "
    "The field has moved from task-specific models to general-purpose foundation models that can "
    "be fine-tuned or prompted for a wide variety of applications. This paradigm shift has been "
    "enabled by three key factors: the availability of massive text corpora for pre-training, "
    "advances in GPU hardware and distributed training, and architectural innovations such as "
    "self-attention and rotary position embeddings.\n\n"
    "Embedding models, which map text to dense vector representations, are a critical component "
    "of modern NLP systems. They power semantic search, retrieval-augmented generation (RAG), "
    "clustering, and recommendation systems. The quality of embeddings directly impacts downstream "
    "task performance. Recent models like BGE-M3, E5, and Qwen3-Embedding have pushed the "
    "boundaries of embedding quality by scaling model size, improving training data diversity, "
    "and introducing techniques like multi-vector retrieval and late interaction.\n\n"
    "Reranking models complement embedding-based retrieval by scoring candidate documents more "
    "precisely. Cross-encoder rerankers jointly encode the query and document, enabling deeper "
    "semantic matching than bi-encoder approaches. The trade-off is higher computational cost, "
    "making them suitable for the final ranking stage rather than initial retrieval. Models like "
    "bge-reranker-v2-m3 and Qwen3-Reranker demonstrate that even relatively small models (0.6B "
    "parameters) can achieve strong reranking performance when properly trained.\n\n"
    "The inference serving infrastructure for these models has also matured significantly. "
    "Hugging Face's Text Embeddings Inference (TEI) provides a purpose-built serving solution "
    "optimized for embedding and reranking workloads. It offers features like automatic batching, "
    "tokenization acceleration with Rust-based tokenizers, and efficient CUDA kernels. "
    "Alternatives like vLLM with its pooling runner mode and traditional Transformers-based "
    "serving each have their own performance characteristics, making framework selection an "
    "important decision for production deployments.",
    "Vector databases have become essential infrastructure for applications leveraging embeddings. "
    "Specialized solutions like Milvus, Pinecone, Weaviate, and Qdrant offer efficient approximate "
    "nearest neighbor (ANN) search at scale, supporting billions of vectors with sub-millisecond "
    "query latency. These systems employ various indexing algorithms such as HNSW, IVF, and "
    "product quantization to balance recall accuracy with computational efficiency. The choice "
    "of vector database depends on factors including dataset size, query throughput requirements, "
    "latency constraints, and whether the application needs hybrid search combining semantic "
    "and keyword-based retrieval. Integration with existing data pipelines and support for "
    "metadata filtering are also important practical considerations.",
    "Fine-tuning embedding models for specific domains has shown significant improvements in "
    "retrieval quality. Contrastive learning objectives like InfoNCE, trained on domain-specific "
    "query-document pairs, can substantially outperform general-purpose embeddings. Techniques "
    "such as hard negative mining, where deliberately challenging negative examples are used "
    "during training, help the model learn finer-grained distinctions. Recent work has also "
    "explored synthetic data generation for fine-tuning, using large language models to create "
    "diverse training pairs that cover edge cases and domain-specific terminology. This approach "
    "reduces the need for expensive human-annotated data while maintaining or improving retrieval "
    "performance on domain-specific benchmarks.",
    "Multi-modal embedding models extend the concept of text embeddings to encompass images, "
    "audio, and other modalities. Models like CLIP and its successors learn a shared embedding "
    "space where text and images can be compared directly, enabling cross-modal retrieval and "
    "zero-shot classification. This has practical applications in image search, content "
    "moderation, recommendation systems, and accessibility tools. The architectural approach "
    "typically involves separate encoders for each modality trained with a contrastive loss "
    "that pulls together embeddings of matching text-image pairs while pushing apart non-matching "
    "pairs. Scaling these models requires careful attention to batch size and training data "
    "quality, as the contrastive loss benefits from large batches with diverse negatives.",
]

RERANK_QUERY = "What is the best framework for serving embedding models?"
RERANK_DOCS = [
    "TEI is a purpose-built inference server for text embeddings and reranking, optimized for "
    "throughput with Rust-based tokenization and automatic batching.",
    "Redis is an in-memory data store often used for caching embedding vectors and managing "
    "vector similarity search at scale.",
    "Transformers is the core library from Hugging Face that provides model loading and inference "
    "for thousands of pretrained models including embedding models.",
    "PostgreSQL with pgvector extension enables storing and querying embedding vectors directly "
    "in a relational database.",
    "vLLM is a high-throughput inference engine that supports embedding models through its "
    "pooling runner mode, offering continuous batching and PagedAttention.",
    "ONNX Runtime provides cross-platform inference optimization for transformer models, "
    "supporting various quantization schemes for deployment.",
]


# ── Backend abstraction ───────────────────────────────────────────────────────


def build_embed_payload(backend: str, texts: list[str]) -> tuple[str, dict]:
    """Return (url_path, json_payload) for the embed request."""
    if backend == "vllm":
        return "/v1/embeddings", {"input": texts, "encoding_format": "float"}
    # TEI
    return "/embed", {"inputs": texts, "truncate": True}


def build_rerank_payload(backend: str, query: str, documents: list[str]) -> tuple[str, dict]:
    """Return (url_path, json_payload) for the rerank request."""
    if backend == "vllm":
        return "/v1/score", {"model": "", "queries": [query], "documents": documents}
    # TEI
    return "/rerank", {"query": query, "texts": documents, "truncate": True}


def extract_embedding_dim(backend: str, result) -> int:
    """Extract embedding dimension from response."""
    if backend == "vllm":
        # {"object":"list","data":[{"embedding":[...],"index":0},...]}
        if isinstance(result, dict) and "data" in result:
            data = result["data"]
            if data and "embedding" in data[0]:
                return len(data[0]["embedding"])
        return 0
    # TEI: [[...], [...]]
    if isinstance(result, list) and result:
        return len(result[0])
    return 0


# ── Helpers ──────────────────────────────────────────────────────────────────


async def check_health(base_url: str, backend: str) -> None:
    async with aiohttp.ClientSession() as session:
        if backend == "vllm":
            url = f"{base_url}/health"
        else:
            url = f"{base_url}/health"
        async with session.get(url) as resp:
            status = resp.status
            body = await resp.text()
    backend_label = backend.upper()
    print(f"  /health  -> {status} {body.strip()}  ({backend_label})")
    assert status == 200, "Health check failed!"


async def test_embed(
    base_url: str, backend: str, texts: list[str], label: str
) -> dict:
    """Send one embed request, return timing + shape info."""
    path, payload = build_embed_payload(backend, texts)
    t0 = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{base_url}{path}", json=payload) as resp:
            result = await resp.json()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert resp.status == 200, f"Embed failed: {resp.status} {json.dumps(result)[:200]}"
    dim = extract_embedding_dim(backend, result)
    return {"label": label, "batch": len(texts), "latency_ms": round(elapsed_ms, 2), "dim": dim}


async def benchmark_embed(
    base_url: str, backend: str, texts: list[str], label: str, rounds: int = 20
) -> None:
    """Run multiple rounds, print trimmed avg/P95 (drop min & max)."""
    times = []
    path, payload = build_embed_payload(backend, texts)
    async with aiohttp.ClientSession() as session:
        # warmup
        for _ in range(3):
            async with session.post(f"{base_url}{path}", json=payload) as resp:
                result = await resp.json()
        # measure
        for _ in range(rounds):
            t0 = time.perf_counter()
            async with session.post(f"{base_url}{path}", json=payload) as resp:
                result = await resp.json()
            times.append((time.perf_counter() - t0) * 1000)

    dim = extract_embedding_dim(backend, result)
    trimmed = sorted(times)[1:-1]  # drop min and max
    avg = statistics.mean(trimmed)
    p95 = sorted(trimmed)[int(len(trimmed) * 0.95)]
    print(
        f"  embed batch={len(texts):>2} dim={dim:>4} [{label:>12s}] | "
        f"avg={avg:>8.2f}ms  p95={p95:>8.2f}ms  "
        f"(rounds={len(times)}, trimmed={len(trimmed)}, "
        f"min={min(times):.2f}ms, max={max(times):.2f}ms)"
    )


async def test_rerank(
    base_url: str, backend: str, query: str, documents: list[str]
) -> None:
    """Test rerank endpoint, print scores."""
    path, payload = build_rerank_payload(backend, query, documents)

    t0 = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{base_url}{path}", json=payload) as resp:
            result = await resp.json()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert resp.status == 200, f"Rerank failed: {resp.status} {json.dumps(result)[:200]}"

    print(f"  rerank batch={len(documents)} | latency={elapsed_ms:.2f}ms")

    # Extract results depending on backend format
    if isinstance(result, dict) and "results" in result:
        entries = result["results"]
    elif isinstance(result, dict) and "data" in result:
        entries = result["data"]
    elif isinstance(result, list):
        entries = result
    else:
        print(f"    Unexpected response format: {type(result)}")
        return

    for r in sorted(entries, key=lambda x: x.get("score", x.get("relevance_score", 0)), reverse=True):
        idx = r.get("index", "?")
        score = r.get("score", r.get("relevance_score", 0))
        text_preview = documents[idx][:60] + "..." if len(documents[idx]) > 60 else documents[idx]
        print(f"    [{idx}] score={score:.4f}  {text_preview}")


async def benchmark_rerank(
    base_url: str, backend: str, query: str, documents: list[str], rounds: int = 20
) -> None:
    """Benchmark rerank over multiple rounds, drop min & max."""
    times = []
    path, payload = build_rerank_payload(backend, query, documents)
    async with aiohttp.ClientSession() as session:
        for _ in range(3):  # warmup
            async with session.post(f"{base_url}{path}", json=payload) as resp:
                await resp.json()
        for _ in range(rounds):
            t0 = time.perf_counter()
            async with session.post(f"{base_url}{path}", json=payload) as resp:
                await resp.json()
            times.append((time.perf_counter() - t0) * 1000)

    trimmed = sorted(times)[1:-1]  # drop min and max
    avg = statistics.mean(trimmed)
    p95 = sorted(trimmed)[int(len(trimmed) * 0.95)]
    print(
        f"  rerank batch={len(documents)} | "
        f"avg={avg:>8.2f}ms  p95={p95:>8.2f}ms  "
        f"(rounds={len(times)}, trimmed={len(trimmed)}, "
        f"min={min(times):.2f}ms, max={max(times):.2f}ms)"
    )


async def benchmark_concurrent_embed(
    base_url: str,
    backend: str,
    texts: list[str],
    length_label: str,
    concurrency: int,
    rounds: int = 10,
) -> None:
    """Benchmark concurrent embed requests.

    Each round sends `concurrency` simultaneous requests (each batch=1).
    Reports wall-clock time, per-request latency, and aggregate QPS.
    """
    path, payload = build_embed_payload(backend, texts[:1])  # batch=1

    async def _single_request(session: aiohttp.ClientSession) -> float:
        t0 = time.perf_counter()
        async with session.post(f"{base_url}{path}", json=payload) as resp:
            await resp.json()
        return (time.perf_counter() - t0) * 1000

    all_per_req_times: list[float] = []  # per-request latencies across all rounds
    wall_times: list[float] = []  # wall-clock per round

    async with aiohttp.ClientSession() as session:
        # warmup (3 rounds)
        for _ in range(3):
            await asyncio.gather(*[_single_request(session) for _ in range(concurrency)])

        # measure
        for _ in range(rounds):
            t_start = time.perf_counter()
            req_times = await asyncio.gather(
                *[_single_request(session) for _ in range(concurrency)]
            )
            wall_ms = (time.perf_counter() - t_start) * 1000
            wall_times.append(wall_ms)
            all_per_req_times.extend(req_times)

    # Stats on wall-clock (total round time)
    wall_trimmed = sorted(wall_times)[1:-1]
    wall_avg = statistics.mean(wall_trimmed)
    wall_p95 = sorted(wall_trimmed)[int(len(wall_trimmed) * 0.95)]

    # Stats on per-request latency
    req_trimmed = sorted(all_per_req_times)[1:-1]
    req_avg = statistics.mean(req_trimmed)
    req_p95 = sorted(req_trimmed)[int(len(req_trimmed) * 0.95)]

    # Aggregate QPS = total_requests / total_wall_time
    total_requests = len(all_per_req_times)
    total_wall_sec = sum(wall_trimmed) / 1000
    qps = total_requests / total_wall_sec if total_wall_sec > 0 else 0

    print(
        f"  concur={concurrency:>2} [{length_label:>12s}] | "
        f"wall_avg={wall_avg:>8.2f}ms  wall_p95={wall_p95:>8.2f}ms  "
        f"req_avg={req_avg:>8.2f}ms  req_p95={req_p95:>8.2f}ms  "
        f"QPS={qps:>7.1f}"
        f"  (rounds={len(wall_times)}, trimmed={len(wall_trimmed)})"
    )


# ── Main ─────────────────────────────────────────────────────────────────────


async def main():
    parser = argparse.ArgumentParser(description="Smoke test for TEI / vLLM server")
    parser.add_argument("--base-url", default="http://localhost:8080", help="Server base URL")
    parser.add_argument(
        "--backend",
        choices=["tei", "vllm"],
        default="tei",
        help="Server backend (tei or vllm)",
    )
    parser.add_argument(
        "--type",
        choices=["embedding", "reranker", "both"],
        default="both",
        help="Which endpoints to test",
    )
    parser.add_argument("--rounds", type=int, default=20, help="Benchmark rounds")
    parser.add_argument("--concurrency-rounds", type=int, default=10, help="Concurrency benchmark rounds")
    parser.add_argument(
        "--mode",
        choices=["all", "batch", "concurrency"],
        default="all",
        help="Benchmark mode: batch (batch size scaling), concurrency (parallel requests), or all",
    )
    args = parser.parse_args()

    print(f"Testing {args.backend.upper()} at {args.base_url}")
    print("=" * 70)

    # Health check
    print("\n[1] Health check")
    await check_health(args.base_url, args.backend)

    # Embedding tests
    if args.type in ("embedding", "both"):
        if args.mode in ("all", "batch"):
            print("\n[2] Embedding endpoint - smoke test")
            r = await test_embed(args.base_url, args.backend, ["Hello world"], "single text")
            print(f"  {r}")
            r = await test_embed(args.base_url, args.backend, SHORT_TEXTS, "short texts")
            print(f"  {r}")

            print("\n[3] Embedding endpoint - benchmark (various batch sizes & lengths)")
            batch_sizes = [1, 2, 3, 4, 8, 12, 32]
            for length_label, corpus in [("short", SHORT_TEXTS), ("medium", MEDIUM_TEXTS), ("long", LONG_TEXTS)]:
                for bs in batch_sizes:
                    await benchmark_embed(args.base_url, args.backend, corpus[:1] * bs, length_label, rounds=args.rounds)

        if args.mode in ("all", "concurrency"):
            if args.mode == "concurrency":
                print("\n[2] Embedding endpoint - smoke test")
                r = await test_embed(args.base_url, args.backend, ["Hello world"], "single text")
                print(f"  {r}")

            print("\n[4] Embedding endpoint - concurrency benchmark (batch=1, various concurrency)")
            concurrency_levels = [1, 3, 5, 10, 20, 32]
            for length_label, corpus in [("short", SHORT_TEXTS), ("medium", MEDIUM_TEXTS), ("long", LONG_TEXTS)]:
                for cc in concurrency_levels:
                    await benchmark_concurrent_embed(
                        args.base_url, args.backend, corpus, length_label,
                        concurrency=cc, rounds=args.concurrency_rounds,
                    )

    # Reranker tests
    if args.type in ("reranker", "both"):
        print("\n[5] Rerank endpoint - smoke test")
        await test_rerank(args.base_url, args.backend, RERANK_QUERY, RERANK_DOCS)

        print("\n[6] Rerank endpoint - benchmark")
        await benchmark_rerank(args.base_url, args.backend, RERANK_QUERY, RERANK_DOCS, rounds=args.rounds)

    print("\n" + "=" * 70)
    print("All tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
