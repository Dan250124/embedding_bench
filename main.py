"""Main entry point for embedding/reranker benchmark."""

import argparse
import asyncio
import importlib
import sys
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from bench.metrics import BenchmarkResult
from bench.reporter import (
    generate_summary_table,
    plot_gpu_memory,
    plot_latency_by_model,
    plot_throughput_by_model,
    save_csv,
)
from bench.workload import Corpus, load_corpus


def load_config(config_path: str = "configs/models.yaml") -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def filter_models(config: dict, model_names: list[str] | None) -> list[dict]:
    models = config["models"]
    if model_names:
        models = [m for m in models if m["name"] in model_names]
    return models


def get_texts(corpus: Corpus, text_length: str) -> list[str]:
    match text_length:
        case "short":
            return corpus.short
        case "medium":
            return corpus.medium
        case "long":
            return corpus.long
        case _:
            raise ValueError(f"Unknown text_length: {text_length}")


# --- Transformers tests ---


def run_transformers_tests(
    model: dict, corpus: Corpus, test_config: dict
) -> list[BenchmarkResult]:
    from bench.transformers_runner import run_embedding_bench, run_reranker_bench

    results = []
    model_path = model["model_id"]
    pooling = model.get("pooling", "cls")

    if model["type"] == "embedding":
        for text_length in test_config["text_lengths"]:
            texts = get_texts(corpus, text_length)
            for bs in test_config["batch_sizes"]:
                # Limit batch to available texts
                actual_bs = min(bs, len(texts))
                print(f"    Transformers | {model['name']} | {text_length} | batch={actual_bs}")
                result = run_embedding_bench(
                    model_path=model_path,
                    model_name=model["name"],
                    texts=texts,
                    batch_size=actual_bs,
                    text_length=text_length,
                    pooling=pooling,
                    warmup_rounds=test_config["warmup_rounds"],
                    measure_rounds=test_config["measure_rounds"],
                )
                results.append(result)
                print(f"      avg={result.avg_latency_ms:.1f}ms  p95={result.p95_latency_ms:.1f}ms  gpu={result.gpu_memory_used_mb:.0f}MB")
    else:
        for text_length in test_config["text_lengths"]:
            pairs = corpus.reranker_pairs
            for bs in test_config["batch_sizes"]:
                actual_bs = min(bs, len(pairs[0].documents))
                print(f"    Transformers | {model['name']} | {text_length} | batch={actual_bs}")
                result = run_reranker_bench(
                    model_path=model_path,
                    model_name=model["name"],
                    pairs=pairs,
                    batch_size=actual_bs,
                    text_length=text_length,
                    warmup_rounds=test_config["warmup_rounds"],
                    measure_rounds=test_config["measure_rounds"],
                )
                results.append(result)
                print(f"      avg={result.avg_latency_ms:.1f}ms  p95={result.p95_latency_ms:.1f}ms  gpu={result.gpu_memory_used_mb:.0f}MB")

    return results


# --- vLLM tests ---


def run_vllm_tests(
    model: dict, corpus: Corpus, test_config: dict
) -> list[BenchmarkResult]:
    from bench.vllm_runner import (
        create_embedder,
        create_reranker,
        run_embedding_bench,
        run_reranker_bench,
    )

    results = []
    model_path = model["model_id"]
    model_name = model["name"]

    if model["type"] == "embedding":
        print(f"    Loading vLLM engine for {model_name}...")
        embedder = create_embedder(model_path)

        for text_length in test_config["text_lengths"]:
            texts = get_texts(corpus, text_length)
            for bs in test_config["batch_sizes"]:
                actual_bs = min(bs, len(texts))
                print(f"    vLLM | {model_name} | {text_length} | batch={actual_bs}")
                result = run_embedding_bench(
                    embedder=embedder,
                    model_name=model_name,
                    texts=texts,
                    batch_size=actual_bs,
                    text_length=text_length,
                    warmup_rounds=test_config["warmup_rounds"],
                    measure_rounds=test_config["measure_rounds"],
                )
                results.append(result)
                print(f"      avg={result.avg_latency_ms:.1f}ms  p95={result.p95_latency_ms:.1f}ms  gpu={result.gpu_memory_used_mb:.0f}MB")

        del embedder
    else:
        print(f"    Loading vLLM engine for {model_name}...")
        reranker = create_reranker(model_path, model_name)

        for text_length in test_config["text_lengths"]:
            pairs = corpus.reranker_pairs
            for bs in test_config["batch_sizes"]:
                actual_bs = min(bs, len(pairs[0].documents))
                print(f"    vLLM | {model_name} | {text_length} | batch={actual_bs}")
                result = run_reranker_bench(
                    reranker=reranker,
                    model_name=model_name,
                    pairs=pairs,
                    batch_size=actual_bs,
                    text_length=text_length,
                    warmup_rounds=test_config["warmup_rounds"],
                    measure_rounds=test_config["measure_rounds"],
                )
                results.append(result)
                print(f"      avg={result.avg_latency_ms:.1f}ms  p95={result.p95_latency_ms:.1f}ms  gpu={result.gpu_memory_used_mb:.0f}MB")

        del reranker

    return results


# --- TEI tests ---


async def run_tei_tests(
    model: dict, corpus: Corpus, test_config: dict, port: int
) -> list[BenchmarkResult]:
    from bench.client import run_embedding_bench, run_reranker_bench

    results = []
    base_url = f"http://localhost:{port}"

    if model["type"] == "embedding":
        for text_length in test_config["text_lengths"]:
            texts = get_texts(corpus, text_length)
            for bs in test_config["batch_sizes"]:
                actual_bs = min(bs, len(texts))
                print(f"    TEI | {model['name']} | {text_length} | batch={actual_bs}")
                result = await run_embedding_bench(
                    base_url=base_url,
                    model_name=model["name"],
                    texts=texts,
                    batch_size=actual_bs,
                    text_length=text_length,
                    warmup_rounds=test_config["warmup_rounds"],
                    measure_rounds=test_config["measure_rounds"],
                )
                results.append(result)
                print(f"      avg={result.avg_latency_ms:.1f}ms  p95={result.p95_latency_ms:.1f}ms  gpu={result.gpu_memory_used_mb:.0f}MB")
    else:
        for text_length in test_config["text_lengths"]:
            pairs = corpus.reranker_pairs
            for bs in test_config["batch_sizes"]:
                actual_bs = min(bs, len(pairs[0].documents))
                print(f"    TEI | {model['name']} | {text_length} | batch={actual_bs}")
                result = await run_reranker_bench(
                    base_url=base_url,
                    model_name=model["name"],
                    pairs=pairs,
                    batch_size=actual_bs,
                    text_length=text_length,
                    warmup_rounds=test_config["warmup_rounds"],
                    measure_rounds=test_config["measure_rounds"],
                )
                results.append(result)
                print(f"      avg={result.avg_latency_ms:.1f}ms  p95={result.p95_latency_ms:.1f}ms  gpu={result.gpu_memory_used_mb:.0f}MB")

    return results


_HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
_TEI_IMAGE = "ghcr.io/huggingface/text-embeddings-inference:120-1.9"


def _resolve_local_model_path(model_id: str) -> str:
    """Resolve HF model_id to local snapshot path."""
    dir_name = f"models--{model_id.replace('/', '--')}"
    hub_dir = _HF_CACHE / dir_name
    if not hub_dir.exists():
        return model_id  # fallback to original ID
    snapshots = list(hub_dir.glob("snapshots/*"))
    return str(snapshots[0]) if snapshots else model_id


def start_tei_container(model_path: str, port: int) -> str:
    import subprocess
    import time

    local_path = _resolve_local_model_path(model_path)
    container_name = f"tei-bench-{port}"

    # Clean up any existing container
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)

    subprocess.run(
        [
            "docker", "run", "--gpus", "all",
            "-p", f"{port}:80",
            "-v", f"{local_path}:/data",
            "--name", container_name,
            "--rm", "-d",
            _TEI_IMAGE,
            "--model-id", "/data",
        ],
        check=True,
    )

    # Wait for ready
    print(f"    Waiting for TEI on port {port}...")
    for i in range(60):
        time.sleep(5)
        try:
            r = subprocess.run(
                ["curl", "-sf", f"http://localhost:{port}/health"],
                capture_output=True, timeout=5,
            )
            if r.returncode == 0:
                print(f"    TEI ready!")
                return container_name
        except Exception:
            pass
        if (i + 1) % 6 == 0:
            print(f"    Still waiting... ({(i+1)*5}s)")

    raise RuntimeError(f"TEI failed to start within 300s on port {port}")


def stop_tei_container(container_name: str) -> None:
    import subprocess
    subprocess.run(["docker", "stop", container_name], capture_output=True)


def main():
    parser = argparse.ArgumentParser(description="Embedding/Reranker Benchmark")
    parser.add_argument(
        "--config", default="configs/models.yaml", help="Path to model config"
    )
    parser.add_argument(
        "--corpus", default="data/corpus.jsonl", help="Path to test corpus"
    )
    parser.add_argument(
        "--models", nargs="+", default=None, help="Model names to test (default: all)"
    )
    parser.add_argument(
        "--frameworks",
        nargs="+",
        default=["transformers", "vllm", "tei"],
        choices=["transformers", "vllm", "tei"],
        help="Frameworks to test",
    )
    parser.add_argument(
        "--test-type",
        default="all",
        choices=["latency", "throughput", "all"],
        help="Type of test to run",
    )
    parser.add_argument(
        "--output-dir", default="results", help="Output directory for results"
    )
    parser.add_argument(
        "--tei-port", type=int, default=8080, help="Base port for TEI containers"
    )
    parser.add_argument(
        "--generate-corpus",
        action="store_true",
        help="Generate corpus and exit",
    )
    args = parser.parse_args()

    # Generate corpus if requested
    if args.generate_corpus:
        from data.generate_corpus import generate_corpus

        generate_corpus(args.corpus)
        return

    # Load config and corpus
    config = load_config(args.config)
    models = filter_models(config, args.models)
    test_config = config["test"]

    if not models:
        print("No models to test. Check --models and config.")
        return

    corpus = load_corpus(args.corpus)
    output_dir = Path(args.output_dir)

    # Load existing results if we're only running a subset of frameworks
    all_results: list[BenchmarkResult] = []
    csv_path = output_dir / "raw" / "benchmark_results.csv"
    if csv_path.exists() and set(args.frameworks) != {"transformers", "vllm", "tei"}:
        import csv as csv_mod
        with open(csv_path, encoding="utf-8") as f:
            for row in csv_mod.DictReader(f):
                all_results.append(BenchmarkResult(
                    model_name=row["model"],
                    framework=row["framework"],
                    test_type=row["test_type"],
                    text_length=row["text_length"],
                    batch_size=int(row["batch_size"]),
                    concurrency=int(row["concurrency"]),
                    gpu_memory_used_mb=float(row["gpu_memory_mb"]),
                    gpu_utilization_pct=float(row["gpu_util_pct"]),
                ))
        print(f"Loaded {len(all_results)} existing results from CSV")

    tei_port = args.tei_port

    print(f"Models to test: {[m['name'] for m in models]}")
    print(f"Frameworks: {args.frameworks}")
    print(f"Test type: {args.test_type}")
    print("=" * 60)

    for model in models:
        print(f"\n{'=' * 60}")
        print(f"Model: {model['name']} ({model['type']}, {model['params']})")
        print(f"{'=' * 60}")

        # Transformers
        if "transformers" in args.frameworks:
            print("\n  [Transformers]")
            results = run_transformers_tests(model, corpus, test_config)
            all_results.extend(results)

        # vLLM
        if "vllm" in args.frameworks:
            print("\n  [vLLM]")
            results = run_vllm_tests(model, corpus, test_config)
            all_results.extend(results)

        # TEI
        if "tei" in args.frameworks:
            print(f"\n  [TEI] (port {tei_port})")
            container_name = start_tei_container(model["model_id"], tei_port)
            try:
                results = asyncio.run(
                    run_tei_tests(model, corpus, test_config, tei_port)
                )
                all_results.extend(results)
            finally:
                stop_tei_container(container_name)
            tei_port += 1

    # Generate reports
    print(f"\n{'=' * 60}")
    print("Generating reports...")

    save_csv(all_results, output_dir / "raw" / "benchmark_results.csv")
    print(f"  CSV saved to {output_dir / 'raw' / 'benchmark_results.csv'}")

    plot_latency_by_model(all_results, output_dir / "figures")
    plot_throughput_by_model(all_results, output_dir / "figures")
    plot_gpu_memory(all_results, output_dir / "figures")
    print(f"  Figures saved to {output_dir / 'figures'}/")

    summary = generate_summary_table(all_results)
    summary_path = output_dir / "summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    print(f"  Summary saved to {summary_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
