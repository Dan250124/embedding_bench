"""Run TEI benchmarks for all models, append results to existing CSV."""

import asyncio
import csv
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from bench.client import run_embedding_bench, run_reranker_bench
from bench.gpu_monitor import sample_gpu_stats
from bench.metrics import BenchmarkResult
from bench.reporter import (
    generate_summary_table,
    plot_gpu_memory,
    plot_latency_by_model,
    plot_throughput_by_model,
    save_csv,
)
from bench.workload import Corpus, RerankerPair, load_corpus


HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
IMAGE = "ghcr.io/huggingface/text-embeddings-inference:120-1.9"
BASE_PORT = 8080


def get_snapshot_path(model_id: str) -> str:
    """Resolve HF model_id to local snapshot path."""
    dir_name = model_id.replace("/", "--")
    hub_dir = HF_CACHE / dir_name
    if not hub_dir.exists():
        raise FileNotFoundError(f"Model not found in HF cache: {hub_dir}")
    snapshots = list(hub_dir.glob("snapshots/*"))
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found in {hub_dir}")
    return str(snapshots[0])


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


def start_tei(model_path: str, port: int) -> str:
    container = f"tei-bench-{port}"
    # Clean up any existing container
    subprocess.run(["docker", "rm", "-f", container], capture_output=True)

    subprocess.run(
        [
            "docker", "run", "--gpus", "all",
            "-p", f"{port}:80",
            "-v", f"{model_path}:/data",
            "--name", container,
            "--rm", "-d",
            IMAGE,
            "--model-id", "/data",
        ],
        check=True,
    )

    # Wait for ready
    print(f"  Waiting for TEI on port {port}...")
    for i in range(60):
        time.sleep(5)
        try:
            result = subprocess.run(
                ["curl", "-s", f"http://localhost:{port}/health"],
                capture_output=True, timeout=5,
            )
            if result.returncode == 0:
                print(f"  TEI ready!")
                return container
        except Exception:
            pass
        print(f"  Still waiting... ({(i+1)*5}s)")

    raise RuntimeError(f"TEI failed to start within 300s on port {port}")


def stop_tei(container: str):
    subprocess.run(["docker", "stop", container], capture_output=True)


async def run_tei_model(model: dict, corpus: Corpus, test_config: dict, port: int) -> list[BenchmarkResult]:
    base_url = f"http://localhost:{port}"
    results = []

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


def load_existing_results(csv_path: Path) -> list[BenchmarkResult]:
    """Load existing results from CSV."""
    results = []
    if not csv_path.exists():
        return results
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(BenchmarkResult(
                model_name=row["model"],
                framework=row["framework"],
                test_type=row["test_type"],
                text_length=row["text_length"],
                batch_size=int(row["batch_size"]),
                concurrency=int(row["concurrency"]),
                latencies_ms=[float(row["avg_latency_ms"])],  # single value for avg
                total_requests=0,
                duration_sec=0.0,
                gpu_memory_used_mb=float(row["gpu_memory_mb"]),
                gpu_utilization_pct=float(row["gpu_util_pct"]),
            ))
    return results


def main():
    config_path = "configs/models.yaml"
    corpus_path = "data/corpus.jsonl"
    output_dir = Path("results")
    csv_path = output_dir / "raw" / "benchmark_results.csv"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    corpus = load_corpus(corpus_path)
    test_config = config["test"]
    port = BASE_PORT

    all_results = load_existing_results(csv_path)
    tei_results: list[BenchmarkResult] = []

    for model in config["models"]:
        print(f"\n{'=' * 60}")
        print(f"TEI | {model['name']} ({model['type']}, {model['params']})")
        print(f"{'=' * 60}")

        model_path = get_snapshot_path(model["model_id"])
        container = start_tei(model_path, port)

        try:
            results = asyncio.run(run_tei_model(model, corpus, test_config, port))
            tei_results.extend(results)
        finally:
            stop_tei(container)
            time.sleep(2)

        port += 1

    # Merge: replace old TEI results, keep TF+vLLM
    non_tei = [r for r in all_results if r.framework != "tei"]
    merged = non_tei + tei_results

    # Save
    save_csv(merged, csv_path)
    print(f"\nCSV saved to {csv_path} ({len(merged)} rows)")

    plot_latency_by_model(merged, output_dir / "figures")
    plot_throughput_by_model(merged, output_dir / "figures")
    plot_gpu_memory(merged, output_dir / "figures")
    print(f"Figures saved to {output_dir / 'figures'}/")

    summary = generate_summary_table(merged)
    summary_path = output_dir / "summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    print(f"Summary saved to {summary_path}")

    print("\nAll TEI tests completed!")


if __name__ == "__main__":
    main()
