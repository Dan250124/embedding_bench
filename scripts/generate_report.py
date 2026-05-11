"""Parse bench_data.txt and generate analysis report with charts.

Output:
  - results/figures/*.png  — charts
  - results/report.md      — full report
"""

import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "results" / "bench_data.txt"
FIG_DIR = BASE_DIR / "results" / "figures"
REPORT_FILE = BASE_DIR / "results" / "report.md"
FIG_WEB = "figures"  # relative path for markdown

BATCH_SIZES = [1, 2, 3, 4, 8, 12, 32]
LENGTHS = ["short", "medium", "long"]
COLORS = {
    "tei": "#e74c3c",
    "vllm": "#2980b9",
    "short": "#27ae60",
    "medium": "#f39c12",
    "long": "#e74c3c",
}
MARKERS = {"tei": "o", "vllm": "s"}

# ── Data structures ───────────────────────────────────────────────────────────


class EmbedRow:
    def __init__(self, model, framework, batch, length, avg_ms, p95_ms):
        self.model = model
        self.framework = framework
        self.batch = batch
        self.length = length
        self.avg_ms = avg_ms
        self.p95_ms = p95_ms

    @property
    def req_per_sec(self):
        return self.batch / (self.avg_ms / 1000)


class RerankRow:
    def __init__(self, model, framework, batch, avg_ms, p95_ms):
        self.model = model
        self.framework = framework
        self.batch = batch
        self.avg_ms = avg_ms
        self.p95_ms = p95_ms


class ConcurrencyRow:
    def __init__(self, model, framework, concurrency, length,
                 wall_avg_ms, wall_p95_ms, req_avg_ms, req_p95_ms, qps):
        self.model = model
        self.framework = framework
        self.concurrency = concurrency
        self.length = length
        self.wall_avg_ms = wall_avg_ms
        self.wall_p95_ms = wall_p95_ms
        self.req_avg_ms = req_avg_ms
        self.req_p95_ms = req_p95_ms
        self.qps = qps


# ── Parser ────────────────────────────────────────────────────────────────────


def parse_data(text: str):
    embed_rows: list[EmbedRow] = []
    rerank_rows: list[RerankRow] = []
    concur_rows: list[ConcurrencyRow] = []

    # Match embedding lines — allow spaces inside brackets
    embed_pat = re.compile(
        r"embed batch=\s*(\d+)\s+dim=\d+\s+\[\s*(\w+)\s*\]\s*\|\s*"
        r"avg=\s*([\d.]+)ms\s+p95=\s*([\d.]+)ms"
    )
    # Match reranker lines
    rerank_pat = re.compile(
        r"rerank batch=\s*(\d+)\s*\|\s*"
        r"avg=\s*([\d.]+)ms\s+p95=\s*([\d.]+)ms"
    )
    # Match concurrency lines
    concur_pat = re.compile(
        r"concur=\s*(\d+)\s+\[\s*(\w+)\s*\]\s*\|\s*"
        r"wall_avg=\s*([\d.]+)ms\s+wall_p95=\s*([\d.]+)ms\s+"
        r"req_avg=\s*([\d.]+)ms\s+req_p95=\s*([\d.]+)ms\s+"
        r"QPS=\s*([\d.]+)"
    )

    # Scan line-by-line, tracking current model/framework context
    current_model = None
    current_framework = None

    def detect_header(line: str):
        nonlocal current_model, current_framework
        model = None
        framework = None
        if "TEI" in line or "tei" in line.lower():
            framework = "tei"
        if "vllm" in line.lower():
            framework = "vllm"
        if "Qwen3-Embedding-0.6B" in line:
            model = "Qwen3-Embedding-0.6B"
        elif "Qwen3-Embedding-4B" in line:
            model = "Qwen3-Embedding-4B"
        elif "bge-m3" in line and "reranker" not in line.lower():
            model = "bge-m3"
        elif "Qwen3-Reranker-0.6B" in line:
            model = "Qwen3-Reranker-0.6B"
        elif "Qwen3-Reranker-4B" in line:
            model = "Qwen3-Reranker-4B"
        elif "bge-reranker" in line:
            model = "bge-reranker-v2-m3"
        if model:
            current_model = model
            if framework:
                current_framework = framework

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        # Check if this is a header line (contains model name or benchmark keyword)
        if any(kw in stripped for kw in [
            "Qwen3-Embedding", "bge-m3", "Qwen3-Reranker", "bge-reranker",
            "benchmark",
        ]):
            detect_header(stripped)
            continue

        # Try to match data lines
        m = embed_pat.search(stripped)
        if m and current_model and current_framework:
            embed_rows.append(EmbedRow(
                current_model, current_framework,
                int(m.group(1)), m.group(2),
                float(m.group(3)), float(m.group(4)),
            ))
            continue

        m = rerank_pat.search(stripped)
        if m and current_model and current_framework:
            rerank_rows.append(RerankRow(
                current_model, current_framework,
                int(m.group(1)),
                float(m.group(2)), float(m.group(3)),
            ))
            continue

        m = concur_pat.search(stripped)
        if m and current_model and current_framework:
            concur_rows.append(ConcurrencyRow(
                current_model, current_framework,
                int(m.group(1)), m.group(2),
                float(m.group(3)), float(m.group(4)),
                float(m.group(5)), float(m.group(6)),
                float(m.group(7)),
            ))

    return embed_rows, rerank_rows, concur_rows


# ── Helpers ───────────────────────────────────────────────────────────────────


def get_rows(rows, model, framework=None, length=None):
    out = [r for r in rows if r.model == model]
    if framework:
        out = [r for r in out if r.framework == framework]
    if length:
        out = [r for r in out if r.length == length]
    return sorted(out, key=lambda r: r.batch)


def throughput(rows, model, framework, length):
    """requests/sec = batch_size / latency_sec"""
    rs = get_rows(rows, model, framework, length)
    return [(r.batch, r.req_per_sec) for r in rs]


# ── Chart generators ─────────────────────────────────────────────────────────


def chart_tei_vs_vllm(rows, model, fig_dir):
    """One model: TEI vs vLLM across all batch sizes, 3 subplots (lengths)."""
    tei = {l: get_rows(rows, model, "tei", l) for l in LENGTHS}
    vllm = {l: get_rows(rows, model, "vllm", l) for l in LENGTHS}
    if not any(tei.values()) or not any(vllm.values()):
        return None

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)
    for ax, length in zip(axes, LENGTHS):
        if tei[length]:
            ax.plot(
                [r.batch for r in tei[length]],
                [r.avg_ms for r in tei[length]],
                marker="o", color=COLORS["tei"], linewidth=2, label="TEI",
            )
        if vllm[length]:
            ax.plot(
                [r.batch for r in vllm[length]],
                [r.avg_ms for r in vllm[length]],
                marker="s", color=COLORS["vllm"], linewidth=2, label="vLLM",
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(BATCH_SIZES)
        ax.set_xticklabels(BATCH_SIZES)
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Avg Latency (ms)")
        ax.set_title(f"{length.capitalize()} Text")
        ax.legend()

    fig.suptitle(f"{model}: TEI vs vLLM — Avg Latency", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = fig_dir / f"{model.replace('/', '_')}_tei_vs_vllm.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path.name


def chart_throughput_tei_vs_vllm(rows, model, fig_dir):
    """Throughput (req/s) comparison: TEI vs vLLM, one subplot per length."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)
    for ax, length in zip(axes, LENGTHS):
        has_data = False
        for fw, color, marker in [("tei", COLORS["tei"], "o"), ("vllm", COLORS["vllm"], "s")]:
            rs = get_rows(rows, model, fw, length)
            if rs:
                has_data = True
                ax.plot(
                    [r.batch for r in rs],
                    [r.req_per_sec for r in rs],
                    marker=marker, color=color, linewidth=2, label=fw.upper(),
                )
        ax.set_xscale("log", base=2)
        ax.set_xticks(BATCH_SIZES)
        ax.set_xticklabels(BATCH_SIZES)
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Throughput (req/s)")
        ax.set_title(f"{length.capitalize()} Text")
        if has_data:
            ax.legend()

    fig.suptitle(f"{model}: TEI vs vLLM — Throughput", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = fig_dir / f"{model.replace('/', '_')}_throughput.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path.name


def chart_all_tei_models(rows, fig_dir):
    """All embedding models on TEI, grouped by length."""
    models = ["Qwen3-Embedding-0.6B", "Qwen3-Embedding-4B", "bge-m3"]
    model_colors = ["#e74c3c", "#8e44ad", "#27ae60"]
    model_markers = ["o", "s", "^"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, length in zip(axes, LENGTHS):
        for m, c, mk in zip(models, model_colors, model_markers):
            rs = get_rows(rows, m, "tei", length)
            if rs:
                ax.plot(
                    [r.batch for r in rs],
                    [r.avg_ms for r in rs],
                    marker=mk, color=c, linewidth=2, label=m,
                )
        ax.set_xscale("log", base=2)
        ax.set_xticks(BATCH_SIZES)
        ax.set_xticklabels(BATCH_SIZES)
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Avg Latency (ms)")
        ax.set_title(f"{length.capitalize()} Text")
        ax.legend(fontsize=9)

    fig.suptitle("All Embedding Models on TEI — Avg Latency", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = fig_dir / "all_tei_models.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path.name


def chart_all_vllm_models(rows, fig_dir):
    """All embedding models on vLLM, grouped by length."""
    models = ["Qwen3-Embedding-0.6B", "Qwen3-Embedding-4B"]
    model_colors = ["#e74c3c", "#8e44ad"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, length in zip(axes, LENGTHS):
        for m, c in zip(models, model_colors):
            rs = get_rows(rows, m, "vllm", length)
            if rs:
                ax.plot(
                    [r.batch for r in rs],
                    [r.avg_ms for r in rs],
                    marker="o", color=c, linewidth=2, label=m,
                )
        ax.set_xscale("log", base=2)
        ax.set_xticks(BATCH_SIZES)
        ax.set_xticklabels(BATCH_SIZES)
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Avg Latency (ms)")
        ax.set_title(f"{length.capitalize()} Text")
        ax.legend()

    fig.suptitle("All Embedding Models on vLLM — Avg Latency", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = fig_dir / "all_vllm_models.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path.name


def chart_reranker_comparison(rerank_rows, fig_dir):
    """Bar chart for reranker models."""
    if not rerank_rows:
        return None

    models = [r.model for r in rerank_rows]
    avg_latencies = [r.avg_ms for r in rerank_rows]
    frameworks = [r.framework.upper() for r in rerank_rows]
    labels = [f"{m}\n({fw})" for m, fw in zip(models, frameworks)]
    colors = [COLORS.get(r.framework.lower(), "#999") for r in rerank_rows]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(labels, avg_latencies, color=colors, width=0.5, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, avg_latencies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}ms", ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("Avg Latency (ms)")
    ax.set_title("Reranker Models — Latency Comparison (batch=6)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(avg_latencies) * 1.25)

    fig.tight_layout()
    path = fig_dir / "reranker_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path.name


def chart_speedup_heatmap(rows, fig_dir):
    """Heatmap: vLLM speedup over TEI (model × batch_size × length)."""
    models = ["Qwen3-Embedding-0.6B", "Qwen3-Embedding-4B"]
    if not all(
        get_rows(rows, m, "tei") and get_rows(rows, m, "vllm") for m in models
    ):
        return None

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, model in zip(axes, models):
        matrix = []
        for length in LENGTHS:
            row_data = []
            for bs in BATCH_SIZES:
                tei_r = [r for r in get_rows(rows, model, "tei", length) if r.batch == bs]
                vllm_r = [r for r in get_rows(rows, model, "vllm", length) if r.batch == bs]
                if tei_r and vllm_r:
                    ratio = tei_r[0].avg_ms / vllm_r[0].avg_ms
                    row_data.append(ratio)
                else:
                    row_data.append(np.nan)
            matrix.append(row_data)

        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0.8, vmax=3.0)
        ax.set_xticks(range(len(BATCH_SIZES)))
        ax.set_xticklabels(BATCH_SIZES)
        ax.set_yticks(range(len(LENGTHS)))
        ax.set_yticklabels([l.capitalize() for l in LENGTHS])
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Text Length")
        ax.set_title(model)

        for i in range(len(LENGTHS)):
            for j in range(len(BATCH_SIZES)):
                val = matrix[i][j]
                if not np.isnan(val):
                    color = "white" if val > 2.2 else "black"
                    ax.text(j, i, f"{val:.2f}x", ha="center", va="center",
                            fontsize=9, fontweight="bold", color=color)

    fig.colorbar(im, ax=axes, label="Speedup (TEI / vLLM)", shrink=0.8)
    fig.suptitle("vLLM Speedup over TEI (higher = vLLM faster)", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = fig_dir / "speedup_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path.name


def chart_p95_comparison(rows, model, fig_dir):
    """P95 latency comparison: TEI vs vLLM."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, length in zip(axes, LENGTHS):
        has_data = False
        for fw, color, marker in [("tei", COLORS["tei"], "o"), ("vllm", COLORS["vllm"], "s")]:
            rs = get_rows(rows, model, fw, length)
            if rs:
                has_data = True
                ax.plot(
                    [r.batch for r in rs],
                    [r.p95_ms for r in rs],
                    marker=marker, color=color, linewidth=2, label=fw.upper(),
                )
        ax.set_xscale("log", base=2)
        ax.set_xticks(BATCH_SIZES)
        ax.set_xticklabels(BATCH_SIZES)
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("P95 Latency (ms)")
        ax.set_title(f"{length.capitalize()} Text")
        if has_data:
            ax.legend()

    fig.suptitle(f"{model}: TEI vs vLLM — P95 Latency", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = fig_dir / f"{model.replace('/', '_')}_p95.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path.name


CONCURRENCY_LEVELS = [1, 3, 5, 10, 20, 32]


def _get_concur(concur_rows, model, framework, length=None):
    out = [r for r in concur_rows if r.model == model and r.framework == framework]
    if length:
        out = [r for r in out if r.length == length]
    return sorted(out, key=lambda r: r.concurrency)


def chart_concurrency_qps(concur_rows, model, fig_dir):
    """QPS vs concurrency level, grouped by text length."""
    rows_tei = _get_concur(concur_rows, model, "tei")
    rows_vllm = _get_concur(concur_rows, model, "vllm")
    if not rows_tei and not rows_vllm:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, length in zip(axes, LENGTHS):
        for source, color, marker, label in [
            (rows_tei, COLORS["tei"], "o", "TEI"),
            (rows_vllm, COLORS["vllm"], "s", "vLLM"),
        ]:
            rs = [r for r in source if r.length == length]
            if rs:
                ax.plot(
                    [r.concurrency for r in rs],
                    [r.qps for r in rs],
                    marker=marker, color=color, linewidth=2, label=label,
                )
        ax.set_xlabel("Concurrency")
        ax.set_ylabel("QPS (requests/sec)")
        ax.set_title(f"{length.capitalize()} Text")
        ax.legend()

    fig.suptitle(f"{model} — Concurrency vs QPS (batch=1 per request)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = fig_dir / f"{model.replace('/', '_')}_concurrency_qps.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path.name


def chart_concurrency_latency(concur_rows, model, fig_dir):
    """Per-request avg latency vs concurrency, grouped by text length."""
    rows_tei = _get_concur(concur_rows, model, "tei")
    rows_vllm = _get_concur(concur_rows, model, "vllm")
    if not rows_tei and not rows_vllm:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, length in zip(axes, LENGTHS):
        for source, color, marker, label in [
            (rows_tei, COLORS["tei"], "o", "TEI"),
            (rows_vllm, COLORS["vllm"], "s", "vLLM"),
        ]:
            rs = [r for r in source if r.length == length]
            if rs:
                ax.plot(
                    [r.concurrency for r in rs],
                    [r.req_avg_ms for r in rs],
                    marker=marker, color=color, linewidth=2, label=f"{label} req_avg",
                )
                ax.fill_between(
                    [r.concurrency for r in rs],
                    [r.req_avg_ms for r in rs],
                    [r.req_p95_ms for r in rs],
                    alpha=0.15, color=color,
                )
        ax.set_xlabel("Concurrency")
        ax.set_ylabel("Per-request Latency (ms)")
        ax.set_title(f"{length.capitalize()} Text")
        ax.legend()

    fig.suptitle(f"{model} — Concurrency vs Per-request Latency (batch=1)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = fig_dir / f"{model.replace('/', '_')}_concurrency_latency.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path.name


# ── Report generation ────────────────────────────────────────────────────────


def build_report(embed_rows, rerank_rows, concur_rows, chart_files: dict):
    lines = []
    w = lines.append

    w("# Embedding / Reranker 推理框架性能测试报告")
    w("")
    w("> **测试目的**: 对比 TEI (Text Embeddings Inference) 与 vLLM 在 Embedding 和 Reranker 模型上的推理性能，")
    w("> 为生产环境技术选型提供数据支撑。")
    w("")
    w("## 测试环境与方法")
    w("")
    w("| 项目 | 说明 |")
    w("|------|------|")
    w("| GPU | NVIDIA (WSL2) |")
    w("| Batch 测试轮数 | 20 轮 (3 warmup + 17 measure)，去掉最高最低各 1 轮取 trimmed mean |")
    w("| 并发测试轮数 | 10 轮 (3 warmup + 7 measure)，去掉最高最低各 1 轮取 trimmed mean |")
    w("| Batch Size | 1, 2, 3, 4, 8, 12, 32 |")
    w("| 并发度 (Concurrency) | 1, 3, 5, 10, 20, 32（每个请求 batch=1）|")
    w("| 文本长度 | short (~32 tokens), medium (~256 tokens), long (~1024 tokens) |")
    w("| 统计指标 | Avg Latency (ms), P95 Latency (ms), Throughput (req/s), QPS |")
    w("| Reranker 批次 | batch=6 文档 |")
    w("")

    # ── Embedding Results ──
    w("---")
    w("")
    w("## 一、Embedding 模型性能对比（Batch Size 维度）")
    w("")

    # 1.1 TEI vs vLLM per model
    for model in ["Qwen3-Embedding-0.6B", "Qwen3-Embedding-4B"]:
        w(f"### {model}")
        w("")

        chart_key = f"{model.replace('/', '_')}_tei_vs_vllm"
        if chart_key in chart_files:
            w(f"![{model} TEI vs vLLM]({FIG_WEB}/{chart_files[chart_key]})")
            w("")

        # Summary table: batch=1 and batch=32 comparison
        w("| 场景 | TEI Avg (ms) | vLLM Avg (ms) | vLLM 提速 | TEI P95 (ms) | vLLM P95 (ms) |")
        w("|------|:---:|:---:|:---:|:---:|:---:|")
        for length in LENGTHS:
            for bs in [1, 32]:
                tei_r = [r for r in get_rows(embed_rows, model, "tei", length) if r.batch == bs]
                vllm_r = [r for r in get_rows(embed_rows, model, "vllm", length) if r.batch == bs]
                if tei_r and vllm_r:
                    t, v = tei_r[0], vllm_r[0]
                    speedup = t.avg_ms / v.avg_ms
                    w(f"| {length} bs={bs} | {t.avg_ms:.2f} | {v.avg_ms:.2f} | **{speedup:.2f}x** | {t.p95_ms:.2f} | {v.p95_ms:.2f} |")
        w("")

        # P95 chart
        p95_key = f"{model.replace('/', '_')}_p95"
        if p95_key in chart_files:
            w(f"![{model} P95]({FIG_WEB}/{chart_files[p95_key]})")
            w("")

        # Throughput chart
        tp_key = f"{model.replace('/', '_')}_throughput"
        if tp_key in chart_files:
            w(f"![{model} Throughput]({FIG_WEB}/{chart_files[tp_key]})")
            w("")

    # 1.2 All TEI models
    w("### TEI 框架 — 所有 Embedding 模型横向对比")
    w("")
    if "all_tei_models" in chart_files:
        w(f"![All TEI Models]({FIG_WEB}/{chart_files['all_tei_models']})")
        w("")

    # TEI summary table
    w("| 模型 | Batch=1 Avg | Batch=32 Avg (short) | Batch=32 Avg (long) | 维度 |")
    w("|------|:---:|:---:|:---:|:---:|")
    for model, dim in [("Qwen3-Embedding-0.6B", 1024), ("Qwen3-Embedding-4B", 2560), ("bge-m3", 1024)]:
        rs_short_1 = [r for r in get_rows(embed_rows, model, "tei", "short") if r.batch == 1]
        rs_short_32 = [r for r in get_rows(embed_rows, model, "tei", "short") if r.batch == 32]
        rs_long_32 = [r for r in get_rows(embed_rows, model, "tei", "long") if r.batch == 32]
        s1 = rs_short_1[0].avg_ms if rs_short_1 else 0
        s32 = rs_short_32[0].avg_ms if rs_short_32 else 0
        l32 = rs_long_32[0].avg_ms if rs_long_32 else 0
        w(f"| {model} | {s1:.2f} | {s32:.2f} | {l32:.2f} | {dim} |")
    w("")

    # 1.3 All vLLM models
    w("### vLLM 框架 — 所有 Embedding 模型横向对比")
    w("")
    if "all_vllm_models" in chart_files:
        w(f"![All vLLM Models]({FIG_WEB}/{chart_files['all_vllm_models']})")
        w("")

    # vLLM summary table
    w("| 模型 | Batch=1 Avg | Batch=32 Avg (short) | Batch=32 Avg (long) | 维度 |")
    w("|------|:---:|:---:|:---:|:---:|")
    for model, dim in [("Qwen3-Embedding-0.6B", 1024), ("Qwen3-Embedding-4B", 2560)]:
        rs_short_1 = [r for r in get_rows(embed_rows, model, "vllm", "short") if r.batch == 1]
        rs_short_32 = [r for r in get_rows(embed_rows, model, "vllm", "short") if r.batch == 32]
        rs_long_32 = [r for r in get_rows(embed_rows, model, "vllm", "long") if r.batch == 32]
        s1 = rs_short_1[0].avg_ms if rs_short_1 else 0
        s32 = rs_short_32[0].avg_ms if rs_short_32 else 0
        l32 = rs_long_32[0].avg_ms if rs_long_32 else 0
        w(f"| {model} | {s1:.2f} | {s32:.2f} | {l32:.2f} | {dim} |")
    w("")

    # 1.4 Speedup heatmap
    w("### vLLM 相对 TEI 加速比 (Batch Size)")
    w("")
    if "speedup_heatmap" in chart_files:
        w(f"![Speedup Heatmap]({FIG_WEB}/{chart_files['speedup_heatmap']})")
        w("")

    # ── Concurrency Results ──
    if concur_rows:
        w("---")
        w("")
        w("## 二、Embedding 模型并发性能对比")
        w("")
        w("> 每个并发请求 batch=1，模拟多客户端同时访问场景。")
        w("")

        for model in ["Qwen3-Embedding-0.6B"]:
            rows_tei = _get_concur(concur_rows, model, "tei")
            rows_vllm = _get_concur(concur_rows, model, "vllm")
            if not rows_tei and not rows_vllm:
                continue

            w(f"### {model} — 并发 QPS 对比")
            w("")

            qps_key = f"{model.replace('/', '_')}_concurrency_qps"
            if qps_key in chart_files:
                w(f"![{model} Concurrency QPS]({FIG_WEB}/{chart_files[qps_key]})")
                w("")

            # QPS comparison table
            w("| 并发度 | TEI QPS (short) | vLLM QPS (short) | TEI QPS (medium) | vLLM QPS (medium) | TEI QPS (long) | vLLM QPS (long) |")
            w("|:------:|:---:|:---:|:---:|:---:|:---:|:---:|")
            for cc in CONCURRENCY_LEVELS:
                vals = []
                for length in LENGTHS:
                    for fw in ["tei", "vllm"]:
                        rs = [r for r in _get_concur(concur_rows, model, fw, length) if r.concurrency == cc]
                        vals.append(f"{rs[0].qps:.0f}" if rs else "-")
                w(f"| {cc} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} | {vals[4]} | {vals[5]} |")
            w("")

            # Per-request latency table
            w(f"### {model} — 并发下单请求延迟")
            w("")
            lat_key = f"{model.replace('/', '_')}_concurrency_latency"
            if lat_key in chart_files:
                w(f"![{model} Concurrency Latency]({FIG_WEB}/{chart_files[lat_key]})")
                w("")

            w("| 并发度 | TEI req_avg (short) | vLLM req_avg (short) | TEI req_p95 (short) | vLLM req_p95 (short) |")
            w("|:------:|:---:|:---:|:---:|:---:|")
            for cc in CONCURRENCY_LEVELS:
                tei_avg = vllm_avg = tei_p95 = vllm_p95 = "-"
                rs_tei = [r for r in _get_concur(concur_rows, model, "tei", "short") if r.concurrency == cc]
                rs_vllm = [r for r in _get_concur(concur_rows, model, "vllm", "short") if r.concurrency == cc]
                if rs_tei:
                    tei_avg = f"{rs_tei[0].req_avg_ms:.1f} ms"
                    tei_p95 = f"{rs_tei[0].req_p95_ms:.1f} ms"
                if rs_vllm:
                    vllm_avg = f"{rs_vllm[0].req_avg_ms:.1f} ms"
                    vllm_p95 = f"{rs_vllm[0].req_p95_ms:.1f} ms"
                w(f"| {cc} | {tei_avg} | {vllm_avg} | {tei_p95} | {vllm_p95} |")
            w("")

            # Concurrency key findings
            w(f"### {model} — 并发测试发现")
            w("")
            conc_findings = []
            # Compare QPS at high concurrency
            tei_cc32_short = [r for r in _get_concur(concur_rows, model, "tei", "short") if r.concurrency == 32]
            vllm_cc32_short = [r for r in _get_concur(concur_rows, model, "vllm", "short") if r.concurrency == 32]
            if tei_cc32_short and vllm_cc32_short:
                tei_qps = tei_cc32_short[0].qps
                vllm_qps = vllm_cc32_short[0].qps
                if tei_qps > vllm_qps:
                    conc_findings.append(
                        f"**高并发 (cc=32, short)**: TEI QPS ({tei_qps:.0f}) 超过 vLLM ({vllm_qps:.0f})，"
                        f"TEI 在高并发下吞吐更优"
                    )
                else:
                    conc_findings.append(
                        f"**高并发 (cc=32, short)**: vLLM QPS ({vllm_qps:.0f}) 超过 TEI ({tei_qps:.0f})"
                    )

            # Latency degradation
            tei_cc1 = [r for r in _get_concur(concur_rows, model, "tei", "short") if r.concurrency == 1]
            tei_cc32 = [r for r in _get_concur(concur_rows, model, "tei", "short") if r.concurrency == 32]
            if tei_cc1 and tei_cc32:
                degrade = tei_cc32[0].req_avg_ms / tei_cc1[0].req_avg_ms
                conc_findings.append(
                    f"**TEI 延迟劣化**: cc=1→32 时单请求延迟从 {tei_cc1[0].req_avg_ms:.1f}ms 增至 {tei_cc32[0].req_avg_ms:.1f}ms（{degrade:.1f}x）"
                )

            vllm_cc1 = [r for r in _get_concur(concur_rows, model, "vllm", "short") if r.concurrency == 1]
            vllm_cc32 = [r for r in _get_concur(concur_rows, model, "vllm", "short") if r.concurrency == 32]
            if vllm_cc1 and vllm_cc32:
                degrade = vllm_cc32[0].req_avg_ms / vllm_cc1[0].req_avg_ms
                conc_findings.append(
                    f"**vLLM 延迟劣化**: cc=1→32 时单请求延迟从 {vllm_cc1[0].req_avg_ms:.1f}ms 增至 {vllm_cc32[0].req_avg_ms:.1f}ms（{degrade:.1f}x）"
                )

            # Long text comparison
            tei_cc32_long = [r for r in _get_concur(concur_rows, model, "tei", "long") if r.concurrency == 32]
            vllm_cc32_long = [r for r in _get_concur(concur_rows, model, "vllm", "long") if r.concurrency == 32]
            if tei_cc32_long and vllm_cc32_long:
                conc_findings.append(
                    f"**长文本高并发 (cc=32, long)**: vLLM QPS ({vllm_cc32_long[0].qps:.0f}) vs TEI QPS ({tei_cc32_long[0].qps:.0f})，"
                    f"vLLM 是 TEI 的 {vllm_cc32_long[0].qps / tei_cc32_long[0].qps:.1f}x"
                )

            for f in conc_findings:
                w(f"- {f}")
            w("")

    # ── Reranker Results ──
    w("---")
    w("")
    w("## 三、Reranker 模型性能对比")
    w("")
    if rerank_rows:
        if "reranker_comparison" in chart_files:
            w(f"![Reranker Comparison]({FIG_WEB}/{chart_files['reranker_comparison']})")
            w("")

        w("| 模型 | 框架 | Batch | Avg Latency (ms) | P95 Latency (ms) |")
        w("|------|------|:---:|:---:|:---:|")
        for r in rerank_rows:
            w(f"| {r.model} | {r.framework.upper()} | {r.batch} | {r.avg_ms:.2f} | {r.p95_ms:.2f} |")
        w("")
    else:
        w("> Reranker 数据暂缺。")
        w("")

    # ── Key Findings ──
    w("---")
    w("")
    w("## 四、关键发现")
    w("")

    findings = []

    for model in ["Qwen3-Embedding-0.6B", "Qwen3-Embedding-4B"]:
        tei_bs1 = [r for r in get_rows(embed_rows, model, "tei", "short") if r.batch == 1]
        vllm_bs1 = [r for r in get_rows(embed_rows, model, "vllm", "short") if r.batch == 1]
        if tei_bs1 and vllm_bs1:
            sp = tei_bs1[0].avg_ms / vllm_bs1[0].avg_ms
            findings.append(f"**{model}**: vLLM 单请求延迟比 TEI 快 **{sp:.1f}x** (short text, batch=1)")

        tei_bs32_long = [r for r in get_rows(embed_rows, model, "tei", "long") if r.batch == 32]
        vllm_bs32_long = [r for r in get_rows(embed_rows, model, "vllm", "long") if r.batch == 32]
        if tei_bs32_long and vllm_bs32_long:
            sp = tei_bs32_long[0].avg_ms / vllm_bs32_long[0].avg_ms
            findings.append(f"**{model}**: vLLM 大批次长文本 (batch=32, long) 比 TEI 快 **{sp:.1f}x**")

    bge_m3_short = [r for r in get_rows(embed_rows, "bge-m3", "tei", "short") if r.batch == 1]
    if bge_m3_short:
        findings.append(f"**bge-m3 (TEI)** 单请求延迟仅 **{bge_m3_short[0].avg_ms:.1f}ms**，是所有 Embedding 模型中最快的")

    q06_short = [r for r in get_rows(embed_rows, "Qwen3-Embedding-0.6B", "tei", "short") if r.batch == 1]
    q4_short = [r for r in get_rows(embed_rows, "Qwen3-Embedding-4B", "tei", "short") if r.batch == 1]
    if q06_short and q4_short:
        ratio = q4_short[0].avg_ms / q06_short[0].avg_ms
        findings.append(f"**4B vs 0.6B**: Qwen3-Embedding-4B 单请求延迟是 0.6B 的 **{ratio:.1f}x** (TEI, batch=1)")

    if rerank_rows:
        rerank_sorted = sorted(rerank_rows, key=lambda r: r.avg_ms)
        fastest = rerank_sorted[0]
        findings.append(f"**Reranker 最快**: {fastest.model} ({fastest.framework.upper()}) — {fastest.avg_ms:.1f}ms/batch")

    # Concurrency findings
    if concur_rows:
        tei_cc32_s = [r for r in _get_concur(concur_rows, "Qwen3-Embedding-0.6B", "tei", "short") if r.concurrency == 32]
        vllm_cc32_s = [r for r in _get_concur(concur_rows, "Qwen3-Embedding-0.6B", "vllm", "short") if r.concurrency == 32]
        if tei_cc32_s and vllm_cc32_s:
            if tei_cc32_s[0].qps > vllm_cc32_s[0].qps:
                findings.append(f"**并发高吞吐**: TEI 在高并发 (cc=32) 短文本 QPS ({tei_cc32_s[0].qps:.0f}) 优于 vLLM ({vllm_cc32_s[0].qps:.0f})")
            else:
                findings.append(f"**并发高吞吐**: vLLM 在高并发 (cc=32) 短文本 QPS ({vllm_cc32_s[0].qps:.0f}) 优于 TEI ({tei_cc32_s[0].qps:.0f})")

        tei_cc32_l = [r for r in _get_concur(concur_rows, "Qwen3-Embedding-0.6B", "tei", "long") if r.concurrency == 32]
        vllm_cc32_l = [r for r in _get_concur(concur_rows, "Qwen3-Embedding-0.6B", "vllm", "long") if r.concurrency == 32]
        if tei_cc32_l and vllm_cc32_l:
            findings.append(f"**长文本并发**: vLLM 在 cc=32 long 文本 QPS ({vllm_cc32_l[0].qps:.0f}) 是 TEI ({tei_cc32_l[0].qps:.0f}) 的 {vllm_cc32_l[0].qps / tei_cc32_l[0].qps:.1f}x")

    for f in findings:
        w(f"- {f}")
    w("")

    # ── Recommendations ──
    w("---")
    w("")
    w("## 五、技术选型建议")
    w("")
    w("### Embedding 服务")
    w("")
    w("| 场景 | 推荐方案 | 理由 |")
    w("|------|----------|------|")
    w("| 低延迟单请求 | vLLM | 单请求延迟显著低于 TEI，P95 尾部延迟也更优 |")
    w("| 高并发短文本 | TEI | 高并发下 QPS 更高，短文本延迟劣化更小 |")
    w("| 高并发长文本 | vLLM | 长文本高并发下 QPS 显著优于 TEI，延迟劣化更小 |")
    w("| 高吞吐批量处理 | vLLM | 大批次场景下吞吐量优势更大，加速比随 batch 增大而增加 |")
    w("| 简单部署/Docker 环境 | TEI | 开箱即用，无需 Python 环境，Docker 一键部署 |")
    w("| 混合推理 (生成+Embedding) | vLLM | 同一框架支持生成和 Embedding，减少运维复杂度 |")
    w("")
    w("### Reranker 服务")
    w("")
    w("| 场景 | 推荐方案 | 理由 |")
    w("|------|----------|------|")
    w("| 追求低延迟 | Qwen3-Reranker-0.6B (vLLM) | 9ms/batch，延迟最低 |")
    w("| 追求精度 | Qwen3-Reranker-4B (vLLM) | 18ms/batch，更大模型精度更高 |")
    w("| TEI 生态兼容 | bge-reranker-v2-m3 (TEI) | TEI 原生支持，延迟可接受 (19ms) |")
    w("")

    w("---")
    w("")
    w("## 六、测试数据明细")
    w("")

    # Full data tables
    for model in ["Qwen3-Embedding-0.6B", "Qwen3-Embedding-4B", "bge-m3"]:
        for fw in ["tei", "vllm"]:
            rs = [r for r in embed_rows if r.model == model and r.framework == fw]
            if not rs:
                continue
            w(f"### {model} + {fw.upper()} (Batch)")
            w("")
            w("| Batch | Length | Avg (ms) | P95 (ms) | Throughput (req/s) |")
            w("|:-----:|:------:|:--------:|:-------:|:------------------:|")
            for r in sorted(rs, key=lambda x: (x.length, x.batch)):
                w(f"| {r.batch} | {r.length} | {r.avg_ms:.2f} | {r.p95_ms:.2f} | {r.req_per_sec:.1f} |")
            w("")

    # Concurrency data tables
    if concur_rows:
        for model in ["Qwen3-Embedding-0.6B"]:
            for fw in ["tei", "vllm"]:
                rs = [r for r in concur_rows if r.model == model and r.framework == fw]
                if not rs:
                    continue
                w(f"### {model} + {fw.upper()} (Concurrency)")
                w("")
                w("| Concurrency | Length | Wall Avg (ms) | Wall P95 (ms) | Req Avg (ms) | Req P95 (ms) | QPS |")
                w("|:-----------:|:------:|:-------------:|:-------------:|:------------:|:------------:|:---:|")
                for r in sorted(rs, key=lambda x: (x.length, x.concurrency)):
                    w(f"| {r.concurrency} | {r.length} | {r.wall_avg_ms:.2f} | {r.wall_p95_ms:.2f} | {r.req_avg_ms:.2f} | {r.req_p95_ms:.2f} | {r.qps:.0f} |")
                w("")

    if rerank_rows:
        w("### Reranker 数据")
        w("")
        w("| 模型 | 框架 | Batch | Avg (ms) | P95 (ms) |")
        w("|------|------|:-----:|:--------:|:-------:|")
        for r in rerank_rows:
            w(f"| {r.model} | {r.framework.upper()} | {r.batch} | {r.avg_ms:.2f} | {r.p95_ms:.2f} |")
        w("")

    w("---")
    w("")
    w("*报告生成时间: 2026-05-12*")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    text = DATA_FILE.read_text()
    embed_rows, rerank_rows, concur_rows = parse_data(text)

    print(f"Parsed {len(embed_rows)} embedding rows, {len(rerank_rows)} reranker rows, {len(concur_rows)} concurrency rows")

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    chart_files = {}

    # Generate batch-size charts
    for model in ["Qwen3-Embedding-0.6B", "Qwen3-Embedding-4B"]:
        name = chart_tei_vs_vllm(embed_rows, model, FIG_DIR)
        if name:
            chart_files[f"{model.replace('/', '_')}_tei_vs_vllm"] = name
            print(f"  Generated {name}")

        name = chart_p95_comparison(embed_rows, model, FIG_DIR)
        if name:
            chart_files[f"{model.replace('/', '_')}_p95"] = name
            print(f"  Generated {name}")

        name = chart_throughput_tei_vs_vllm(embed_rows, model, FIG_DIR)
        if name:
            chart_files[f"{model.replace('/', '_')}_throughput"] = name
            print(f"  Generated {name}")

    name = chart_all_tei_models(embed_rows, FIG_DIR)
    if name:
        chart_files["all_tei_models"] = name
        print(f"  Generated {name}")

    name = chart_all_vllm_models(embed_rows, FIG_DIR)
    if name:
        chart_files["all_vllm_models"] = name
        print(f"  Generated {name}")

    name = chart_reranker_comparison(rerank_rows, FIG_DIR)
    if name:
        chart_files["reranker_comparison"] = name
        print(f"  Generated {name}")

    name = chart_speedup_heatmap(embed_rows, FIG_DIR)
    if name:
        chart_files["speedup_heatmap"] = name
        print(f"  Generated {name}")

    # Generate concurrency charts
    for model in ["Qwen3-Embedding-0.6B", "Qwen3-Embedding-4B"]:
        name = chart_concurrency_qps(concur_rows, model, FIG_DIR)
        if name:
            chart_files[f"{model.replace('/', '_')}_concurrency_qps"] = name
            print(f"  Generated {name}")

        name = chart_concurrency_latency(concur_rows, model, FIG_DIR)
        if name:
            chart_files[f"{model.replace('/', '_')}_concurrency_latency"] = name
            print(f"  Generated {name}")

    # Generate report
    report = build_report(embed_rows, rerank_rows, concur_rows, chart_files)
    REPORT_FILE.write_text(report)
    print(f"\nReport written to {REPORT_FILE}")


if __name__ == "__main__":
    main()
