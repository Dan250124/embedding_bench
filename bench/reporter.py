"""Report generation: CSV + matplotlib charts."""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bench.metrics import BenchmarkResult

# Chinese font support
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def save_csv(results: list[BenchmarkResult], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "framework", "test_type", "text_length",
            "batch_size", "concurrency",
            "avg_latency_ms", "p50_ms", "p95_ms", "p99_ms",
            "requests_per_sec", "gpu_memory_mb", "gpu_util_pct",
        ])
        for r in results:
            writer.writerow([
                r.model_name, r.framework, r.test_type, r.text_length,
                r.batch_size, r.concurrency,
                f"{r.avg_latency_ms:.2f}",
                f"{r.p50_latency_ms:.2f}",
                f"{r.p95_latency_ms:.2f}",
                f"{r.p99_latency_ms:.2f}",
                f"{r.requests_per_sec:.2f}",
                f"{r.gpu_memory_used_mb:.0f}",
                f"{r.gpu_utilization_pct:.0f}",
            ])


def plot_latency_by_model(
    results: list[BenchmarkResult], output_dir: str | Path
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(dict)
    for r in results:
        if not r.latencies_ms:
            continue
        key = (r.model_name, r.test_type, r.text_length, r.batch_size)
        groups[key][r.framework] = r.avg_latency_ms

    if not groups:
        return

    # Overview: medium text, batch_size=32
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for idx, test_type in enumerate(["embedding", "reranker"]):
        ax = axes[idx]
        models = sorted(set(k[0] for k in groups if k[1] == test_type and k[2] == "medium" and k[3] == 32))
        if not models:
            ax.set_visible(False)
            continue

        framework_order = ["transformers", "vllm", "tei"]
        x = np.arange(len(models))
        width = 0.25

        for fi, fw in enumerate(framework_order):
            vals = []
            for m in models:
                lat = groups.get((m, test_type, "medium", 32), {}).get(fw, 0)
                vals.append(lat)
            bars = ax.bar(x + fi * width, vals, width, label=fw)
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.1f}", ha="center", va="bottom", fontsize=7,
                    )

        ax.set_xlabel("Model")
        ax.set_ylabel("Avg Latency (ms)")
        ax.set_title(f"{test_type.capitalize()} - Latency (batch=32, medium)")
        ax.set_xticks(x + width)
        ax.set_xticklabels(models, rotation=15, ha="right", fontsize=8)
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "latency_overview.png", dpi=150)
    plt.close()

    # Per-model latency vs batch_size
    for (model, test_type, text_length, _), frameworks in groups.items():
        fig, ax = plt.subplots(figsize=(10, 5))
        all_bs = defaultdict(dict)
        for (m, tt, tl, bs), fws in groups.items():
            if m == model and tt == test_type and tl == text_length:
                for fw, lat in fws.items():
                    all_bs[bs][fw] = lat

        if not all_bs:
            plt.close(fig)
            continue

        for fw in ["transformers", "vllm", "tei"]:
            bs_sorted = sorted(all_bs.keys())
            vals = [all_bs[bs].get(fw, 0) for bs in bs_sorted]
            if any(v > 0 for v in vals):
                ax.plot(bs_sorted, vals, marker="o", label=fw)

        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Avg Latency (ms)")
        ax.set_title(f"{model} - {test_type} - {text_length}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fname = f"latency_{model}_{test_type}_{text_length}.png".replace(" ", "_")
        plt.savefig(output_dir / fname, dpi=150)
        plt.close()


def plot_throughput_by_model(
    results: list[BenchmarkResult], output_dir: str | Path
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(dict)
    for r in results:
        if r.requests_per_sec > 0:
            key = (r.model_name, r.test_type)
            groups[key][r.framework] = r.requests_per_sec

    if not groups:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for idx, test_type in enumerate(["embedding", "reranker"]):
        ax = axes[idx]
        models = sorted(set(k[0] for k in groups if k[1] == test_type))
        if not models:
            ax.set_visible(False)
            continue

        x = np.arange(len(models))
        width = 0.25

        for fi, fw in enumerate(["transformers", "vllm", "tei"]):
            vals = [groups.get((m, test_type), {}).get(fw, 0) for m in models]
            bars = ax.bar(x + fi * width, vals, width, label=fw)
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.1f}", ha="center", va="bottom", fontsize=7,
                    )

        ax.set_xlabel("Model")
        ax.set_ylabel("Requests/sec")
        ax.set_title(f"{test_type.capitalize()} - Throughput")
        ax.set_xticks(x + width)
        ax.set_xticklabels(models, rotation=15, ha="right", fontsize=8)
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "throughput_overview.png", dpi=150)
    plt.close()


def plot_gpu_memory(
    results: list[BenchmarkResult], output_dir: str | Path
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(dict)
    for r in results:
        if r.gpu_memory_used_mb > 0:
            key = (r.model_name, r.test_type)
            groups[key][r.framework] = r.gpu_memory_used_mb

    if not groups:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    models_test = sorted(groups.keys())
    x = np.arange(len(models_test))
    width = 0.25

    for fi, fw in enumerate(["transformers", "vllm", "tei"]):
        vals = [groups[k].get(fw, 0) for k in models_test]
        bars = ax.bar(x + fi * width, vals, width, label=fw)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.0f}", ha="center", va="bottom", fontsize=7,
                )

    labels = [f"{m} ({t})" for m, t in models_test]
    ax.set_xlabel("Model (type)")
    ax.set_ylabel("GPU Memory (MB)")
    ax.set_title("GPU Memory Usage")
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "gpu_memory.png", dpi=150)
    plt.close()


def generate_summary_table(results: list[BenchmarkResult]) -> str:
    lines = ["# Benchmark Results Summary\n"]

    for test_type in ["embedding", "reranker"]:
        type_results = [r for r in results if r.test_type == test_type]
        if not type_results:
            continue

        lines.append(f"## {test_type.capitalize()}\n")
        lines.append(
            "| Model | Framework | Text Len | Batch | Avg(ms) | P50(ms) | "
            "P95(ms) | P99(ms) | Req/s | GPU(MB) |"
        )
        lines.append(
            "|-------|-----------|----------|-------|---------|---------|"
            "--------|--------|-------|---------|"
        )

        for r in type_results:
            lines.append(
                f"| {r.model_name} | {r.framework} | {r.text_length} | "
                f"{r.batch_size} | {r.avg_latency_ms:.1f} | "
                f"{r.p50_latency_ms:.1f} | {r.p95_latency_ms:.1f} | "
                f"{r.p99_latency_ms:.1f} | {r.requests_per_sec:.1f} | "
                f"{r.gpu_memory_used_mb:.0f} |"
            )

        lines.append("")

    return "\n".join(lines)
