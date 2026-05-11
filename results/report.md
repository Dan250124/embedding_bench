# Embedding / Reranker 推理框架性能测试报告

> **测试目的**: 对比 TEI (Text Embeddings Inference) 与 vLLM 在 Embedding 和 Reranker 模型上的推理性能，
> 为生产环境技术选型提供数据支撑。

## 测试环境与方法

| 项目 | 说明 |
|------|------|
| GPU | NVIDIA (WSL2) |
| Batch 测试轮数 | 20 轮 (3 warmup + 17 measure)，去掉最高最低各 1 轮取 trimmed mean |
| 并发测试轮数 | 10 轮 (3 warmup + 7 measure)，去掉最高最低各 1 轮取 trimmed mean |
| Batch Size | 1, 2, 3, 4, 8, 12, 32 |
| 并发度 (Concurrency) | 1, 3, 5, 10, 20, 32（每个请求 batch=1）|
| 文本长度 | short (~32 tokens), medium (~256 tokens), long (~1024 tokens) |
| 统计指标 | Avg Latency (ms), P95 Latency (ms), Throughput (req/s), QPS |
| Reranker 批次 | batch=6 文档 |

---

## 一、Embedding 模型性能对比（Batch Size 维度）

### Qwen3-Embedding-0.6B

![Qwen3-Embedding-0.6B TEI vs vLLM](figures/Qwen3-Embedding-0.6B_tei_vs_vllm.png)

| 场景 | TEI Avg (ms) | vLLM Avg (ms) | vLLM 提速 | TEI P95 (ms) | vLLM P95 (ms) |
|------|:---:|:---:|:---:|:---:|:---:|
| short bs=1 | 11.42 | 5.27 | **2.17x** | 12.78 | 6.07 |
| short bs=32 | 25.28 | 28.50 | **0.89x** | 32.27 | 31.40 |
| medium bs=1 | 11.84 | 5.59 | **2.12x** | 13.03 | 6.60 |
| medium bs=32 | 50.05 | 32.82 | **1.52x** | 53.34 | 34.39 |
| long bs=1 | 13.46 | 6.78 | **1.99x** | 14.26 | 7.23 |
| long bs=32 | 126.97 | 46.07 | **2.76x** | 131.36 | 48.66 |

![Qwen3-Embedding-0.6B P95](figures/Qwen3-Embedding-0.6B_p95.png)

![Qwen3-Embedding-0.6B Throughput](figures/Qwen3-Embedding-0.6B_throughput.png)

### Qwen3-Embedding-4B

![Qwen3-Embedding-4B TEI vs vLLM](figures/Qwen3-Embedding-4B_tei_vs_vllm.png)

| 场景 | TEI Avg (ms) | vLLM Avg (ms) | vLLM 提速 | TEI P95 (ms) | vLLM P95 (ms) |
|------|:---:|:---:|:---:|:---:|:---:|
| short bs=1 | 18.13 | 10.76 | **1.68x** | 18.64 | 11.56 |
| short bs=32 | 51.97 | 63.48 | **0.82x** | 65.02 | 65.91 |
| medium bs=1 | 18.94 | 10.27 | **1.84x** | 19.39 | 11.76 |
| medium bs=32 | 160.70 | 65.21 | **2.46x** | 165.79 | 66.26 |
| long bs=1 | 33.25 | 12.08 | **2.75x** | 34.09 | 12.82 |
| long bs=32 | 581.13 | 84.79 | **6.85x** | 587.23 | 86.59 |

![Qwen3-Embedding-4B P95](figures/Qwen3-Embedding-4B_p95.png)

![Qwen3-Embedding-4B Throughput](figures/Qwen3-Embedding-4B_throughput.png)

### TEI 框架 — 所有 Embedding 模型横向对比

![All TEI Models](figures/all_tei_models.png)

| 模型 | Batch=1 Avg | Batch=32 Avg (short) | Batch=32 Avg (long) | 维度 |
|------|:---:|:---:|:---:|:---:|
| Qwen3-Embedding-0.6B | 11.42 | 25.28 | 126.97 | 1024 |
| Qwen3-Embedding-4B | 18.13 | 51.97 | 581.13 | 2560 |
| bge-m3 | 7.29 | 17.27 | 84.81 | 1024 |

### vLLM 框架 — 所有 Embedding 模型横向对比

![All vLLM Models](figures/all_vllm_models.png)

| 模型 | Batch=1 Avg | Batch=32 Avg (short) | Batch=32 Avg (long) | 维度 |
|------|:---:|:---:|:---:|:---:|
| Qwen3-Embedding-0.6B | 5.27 | 28.50 | 46.07 | 1024 |
| Qwen3-Embedding-4B | 10.76 | 63.48 | 84.79 | 2560 |

### vLLM 相对 TEI 加速比 (Batch Size)

![Speedup Heatmap](figures/speedup_heatmap.png)

---

## 二、Embedding 模型并发性能对比

> 每个并发请求 batch=1，模拟多客户端同时访问场景。

### Qwen3-Embedding-0.6B — 并发 QPS 对比

![Qwen3-Embedding-0.6B Concurrency QPS](figures/Qwen3-Embedding-0.6B_concurrency_qps.png)

| 并发度 | TEI QPS (short) | vLLM QPS (short) | TEI QPS (medium) | vLLM QPS (medium) | TEI QPS (long) | vLLM QPS (long) |
|:------:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 127 | 67 | 102 | 176 | 93 | 184 |
| 3 | 134 | 333 | 133 | 341 | 111 | 303 |
| 5 | 220 | 453 | 205 | 462 | 150 | 447 |
| 10 | 433 | 637 | 362 | 615 | 215 | 593 |
| 20 | 795 | 781 | 580 | 816 | 284 | 757 |
| 32 | 1196 | 924 | 814 | 899 | 324 | 880 |

### Qwen3-Embedding-0.6B — 并发下单请求延迟

![Qwen3-Embedding-0.6B Concurrency Latency](figures/Qwen3-Embedding-0.6B_concurrency_latency.png)

| 并发度 | TEI req_avg (short) | vLLM req_avg (short) | TEI req_p95 (short) | vLLM req_p95 (short) |
|:------:|:---:|:---:|:---:|:---:|
| 1 | 9.8 ms | 18.5 ms | 10.0 ms | 19.6 ms |
| 3 | 19.8 ms | 9.9 ms | 32.9 ms | 12.1 ms |
| 5 | 22.0 ms | 12.3 ms | 28.8 ms | 15.0 ms |
| 10 | 25.1 ms | 17.9 ms | 29.1 ms | 19.8 ms |
| 20 | 28.3 ms | 29.4 ms | 31.5 ms | 35.1 ms |
| 32 | 29.7 ms | 37.7 ms | 34.0 ms | 43.2 ms |

### Qwen3-Embedding-0.6B — 并发测试发现

- **高并发 (cc=32, short)**: TEI QPS (1196) 超过 vLLM (924)，TEI 在高并发下吞吐更优
- **TEI 延迟劣化**: cc=1→32 时单请求延迟从 9.8ms 增至 29.7ms（3.0x）
- **vLLM 延迟劣化**: cc=1→32 时单请求延迟从 18.5ms 增至 37.7ms（2.0x）
- **长文本高并发 (cc=32, long)**: vLLM QPS (880) vs TEI QPS (324)，vLLM 是 TEI 的 2.7x

---

## 三、Reranker 模型性能对比

![Reranker Comparison](figures/reranker_comparison.png)

| 模型 | 框架 | Batch | Avg Latency (ms) | P95 Latency (ms) |
|------|------|:---:|:---:|:---:|
| Qwen3-Reranker-0.6B | VLLM | 6 | 9.02 | 9.88 |
| Qwen3-Reranker-4B | VLLM | 6 | 17.93 | 19.38 |
| bge-reranker-v2-m3 | TEI | 6 | 18.78 | 32.39 |

---

## 四、关键发现

- **Qwen3-Embedding-0.6B**: vLLM 单请求延迟比 TEI 快 **2.2x** (short text, batch=1)
- **Qwen3-Embedding-0.6B**: vLLM 大批次长文本 (batch=32, long) 比 TEI 快 **2.8x**
- **Qwen3-Embedding-4B**: vLLM 单请求延迟比 TEI 快 **1.7x** (short text, batch=1)
- **Qwen3-Embedding-4B**: vLLM 大批次长文本 (batch=32, long) 比 TEI 快 **6.9x**
- **bge-m3 (TEI)** 单请求延迟仅 **7.3ms**，是所有 Embedding 模型中最快的
- **4B vs 0.6B**: Qwen3-Embedding-4B 单请求延迟是 0.6B 的 **1.6x** (TEI, batch=1)
- **Reranker 最快**: Qwen3-Reranker-0.6B (VLLM) — 9.0ms/batch
- **并发高吞吐**: TEI 在高并发 (cc=32) 短文本 QPS (1196) 优于 vLLM (924)
- **长文本并发**: vLLM 在 cc=32 long 文本 QPS (880) 是 TEI (324) 的 2.7x

---

## 五、技术选型建议

### Embedding 服务

| 场景 | 推荐方案 | 理由 |
|------|----------|------|
| 低延迟单请求 | vLLM | 单请求延迟显著低于 TEI，P95 尾部延迟也更优 |
| 高并发短文本 | TEI | 高并发下 QPS 更高，短文本延迟劣化更小 |
| 高并发长文本 | vLLM | 长文本高并发下 QPS 显著优于 TEI，延迟劣化更小 |
| 高吞吐批量处理 | vLLM | 大批次场景下吞吐量优势更大，加速比随 batch 增大而增加 |
| 简单部署/Docker 环境 | TEI | 开箱即用，无需 Python 环境，Docker 一键部署 |
| 混合推理 (生成+Embedding) | vLLM | 同一框架支持生成和 Embedding，减少运维复杂度 |

### Reranker 服务

| 场景 | 推荐方案 | 理由 |
|------|----------|------|
| 追求低延迟 | Qwen3-Reranker-0.6B (vLLM) | 9ms/batch，延迟最低 |
| 追求精度 | Qwen3-Reranker-4B (vLLM) | 18ms/batch，更大模型精度更高 |
| TEI 生态兼容 | bge-reranker-v2-m3 (TEI) | TEI 原生支持，延迟可接受 (19ms) |

---

## 六、测试数据明细

### Qwen3-Embedding-0.6B + TEI (Batch)

| Batch | Length | Avg (ms) | P95 (ms) | Throughput (req/s) |
|:-----:|:------:|:--------:|:-------:|:------------------:|
| 1 | long | 13.46 | 14.26 | 74.3 |
| 2 | long | 20.71 | 25.39 | 96.6 |
| 3 | long | 26.59 | 29.84 | 112.8 |
| 4 | long | 31.63 | 40.52 | 126.5 |
| 8 | long | 53.74 | 55.48 | 148.9 |
| 12 | long | 65.85 | 68.97 | 182.2 |
| 32 | long | 126.97 | 131.36 | 252.0 |
| 1 | medium | 11.84 | 13.03 | 84.5 |
| 2 | medium | 12.09 | 13.55 | 165.4 |
| 3 | medium | 16.14 | 21.71 | 185.9 |
| 4 | medium | 18.48 | 23.55 | 216.5 |
| 8 | medium | 30.78 | 33.72 | 259.9 |
| 12 | medium | 33.42 | 36.46 | 359.1 |
| 32 | medium | 50.05 | 53.34 | 639.4 |
| 1 | short | 11.42 | 12.78 | 87.6 |
| 2 | short | 11.04 | 11.69 | 181.2 |
| 3 | short | 12.27 | 20.12 | 244.5 |
| 4 | short | 15.23 | 20.24 | 262.6 |
| 8 | short | 20.60 | 21.85 | 388.3 |
| 12 | short | 22.13 | 29.24 | 542.3 |
| 32 | short | 25.28 | 32.27 | 1265.8 |

### Qwen3-Embedding-0.6B + VLLM (Batch)

| Batch | Length | Avg (ms) | P95 (ms) | Throughput (req/s) |
|:-----:|:------:|:--------:|:-------:|:------------------:|
| 1 | long | 6.78 | 7.23 | 147.5 |
| 2 | long | 8.75 | 12.62 | 228.6 |
| 3 | long | 9.65 | 13.54 | 310.9 |
| 4 | long | 11.76 | 13.09 | 340.1 |
| 8 | long | 18.09 | 20.13 | 442.2 |
| 12 | long | 22.05 | 23.52 | 544.2 |
| 32 | long | 46.07 | 48.66 | 694.6 |
| 1 | medium | 5.59 | 6.60 | 178.9 |
| 2 | medium | 8.43 | 10.36 | 237.2 |
| 3 | medium | 9.93 | 11.73 | 302.1 |
| 4 | medium | 11.38 | 12.23 | 351.5 |
| 8 | medium | 14.32 | 15.03 | 558.7 |
| 12 | medium | 18.02 | 19.77 | 665.9 |
| 32 | medium | 32.82 | 34.39 | 975.0 |
| 1 | short | 5.27 | 6.07 | 189.8 |
| 2 | short | 8.37 | 10.74 | 238.9 |
| 3 | short | 9.95 | 11.08 | 301.5 |
| 4 | short | 11.14 | 12.87 | 359.1 |
| 8 | short | 13.41 | 14.49 | 596.6 |
| 12 | short | 15.94 | 17.07 | 752.8 |
| 32 | short | 28.50 | 31.40 | 1122.8 |

### Qwen3-Embedding-4B + TEI (Batch)

| Batch | Length | Avg (ms) | P95 (ms) | Throughput (req/s) |
|:-----:|:------:|:--------:|:-------:|:------------------:|
| 1 | long | 33.25 | 34.09 | 30.1 |
| 2 | long | 55.41 | 66.05 | 36.1 |
| 3 | long | 73.16 | 85.78 | 41.0 |
| 4 | long | 96.96 | 113.85 | 41.3 |
| 8 | long | 180.18 | 184.51 | 44.4 |
| 12 | long | 245.49 | 248.52 | 48.9 |
| 32 | long | 581.13 | 587.23 | 55.1 |
| 1 | medium | 18.94 | 19.39 | 52.8 |
| 2 | medium | 22.20 | 22.57 | 90.1 |
| 3 | medium | 24.74 | 40.01 | 121.3 |
| 4 | medium | 35.80 | 44.20 | 111.7 |
| 8 | medium | 70.40 | 73.66 | 113.6 |
| 12 | medium | 85.92 | 88.22 | 139.7 |
| 32 | medium | 160.70 | 165.79 | 199.1 |
| 1 | short | 18.13 | 18.64 | 55.2 |
| 2 | short | 19.51 | 35.75 | 102.5 |
| 3 | short | 20.60 | 36.67 | 145.6 |
| 4 | short | 29.97 | 36.15 | 133.5 |
| 8 | short | 39.73 | 53.88 | 201.4 |
| 12 | short | 41.46 | 55.66 | 289.4 |
| 32 | short | 51.97 | 65.02 | 615.7 |

### Qwen3-Embedding-4B + VLLM (Batch)

| Batch | Length | Avg (ms) | P95 (ms) | Throughput (req/s) |
|:-----:|:------:|:--------:|:-------:|:------------------:|
| 1 | long | 12.08 | 12.82 | 82.8 |
| 2 | long | 15.71 | 23.26 | 127.3 |
| 3 | long | 19.76 | 23.69 | 151.8 |
| 4 | long | 24.12 | 29.38 | 165.8 |
| 8 | long | 31.73 | 34.77 | 252.1 |
| 12 | long | 39.57 | 40.68 | 303.3 |
| 32 | long | 84.79 | 86.59 | 377.4 |
| 1 | medium | 10.27 | 11.76 | 97.4 |
| 2 | medium | 15.53 | 19.63 | 128.8 |
| 3 | medium | 19.00 | 21.93 | 157.9 |
| 4 | medium | 22.52 | 24.69 | 177.6 |
| 8 | medium | 29.14 | 30.21 | 274.5 |
| 12 | medium | 35.21 | 36.39 | 340.8 |
| 32 | medium | 65.21 | 66.26 | 490.7 |
| 1 | short | 10.76 | 11.56 | 92.9 |
| 2 | short | 18.46 | 21.02 | 108.3 |
| 3 | short | 20.84 | 22.34 | 144.0 |
| 4 | short | 22.64 | 23.88 | 176.7 |
| 8 | short | 27.30 | 28.30 | 293.0 |
| 12 | short | 33.09 | 35.15 | 362.6 |
| 32 | short | 63.48 | 65.91 | 504.1 |

### bge-m3 + TEI (Batch)

| Batch | Length | Avg (ms) | P95 (ms) | Throughput (req/s) |
|:-----:|:------:|:--------:|:-------:|:------------------:|
| 1 | long | 9.43 | 9.67 | 106.0 |
| 2 | long | 13.35 | 17.43 | 149.8 |
| 3 | long | 18.26 | 19.87 | 164.3 |
| 4 | long | 19.63 | 27.12 | 203.8 |
| 8 | long | 37.22 | 40.80 | 214.9 |
| 12 | long | 45.45 | 47.68 | 264.0 |
| 32 | long | 84.81 | 88.15 | 377.3 |
| 1 | medium | 7.86 | 9.93 | 127.2 |
| 2 | medium | 8.13 | 8.94 | 246.0 |
| 3 | medium | 9.04 | 13.64 | 331.9 |
| 4 | medium | 12.39 | 15.04 | 322.8 |
| 8 | medium | 20.47 | 23.08 | 390.8 |
| 12 | medium | 20.07 | 24.58 | 597.9 |
| 32 | medium | 34.81 | 37.20 | 919.3 |
| 1 | short | 7.29 | 12.18 | 137.2 |
| 2 | short | 6.93 | 7.24 | 288.6 |
| 3 | short | 7.67 | 12.86 | 391.1 |
| 4 | short | 11.40 | 13.88 | 350.9 |
| 8 | short | 12.66 | 14.80 | 631.9 |
| 12 | short | 14.31 | 15.14 | 838.6 |
| 32 | short | 17.27 | 18.52 | 1852.9 |

### Qwen3-Embedding-0.6B + TEI (Concurrency)

| Concurrency | Length | Wall Avg (ms) | Wall P95 (ms) | Req Avg (ms) | Req P95 (ms) | QPS |
|:-----------:|:------:|:-------------:|:-------------:|:------------:|:------------:|:---:|
| 1 | long | 13.47 | 13.92 | 13.44 | 13.89 | 93 |
| 3 | long | 33.78 | 35.61 | 23.69 | 35.37 | 111 |
| 5 | long | 41.73 | 42.26 | 31.58 | 41.73 | 150 |
| 10 | long | 58.21 | 61.79 | 49.68 | 65.05 | 215 |
| 20 | long | 87.97 | 89.96 | 76.97 | 89.25 | 284 |
| 32 | long | 123.37 | 124.75 | 107.51 | 123.22 | 324 |
| 1 | medium | 12.28 | 13.31 | 12.24 | 13.27 | 102 |
| 3 | medium | 28.17 | 29.68 | 19.32 | 29.29 | 133 |
| 5 | medium | 30.55 | 31.31 | 23.73 | 30.84 | 205 |
| 10 | medium | 34.52 | 35.09 | 29.02 | 35.96 | 362 |
| 20 | medium | 43.11 | 44.26 | 38.61 | 42.59 | 580 |
| 32 | medium | 49.16 | 49.97 | 43.01 | 47.92 | 814 |
| 1 | short | 9.84 | 10.07 | 9.81 | 10.04 | 127 |
| 3 | short | 27.91 | 28.41 | 19.78 | 32.87 | 134 |
| 5 | short | 28.47 | 29.16 | 22.02 | 28.79 | 220 |
| 10 | short | 28.87 | 29.62 | 25.11 | 29.07 | 433 |
| 20 | short | 31.46 | 33.12 | 28.30 | 31.53 | 795 |
| 32 | short | 33.46 | 34.93 | 29.68 | 34.03 | 1196 |

### Qwen3-Embedding-0.6B + VLLM (Concurrency)

| Concurrency | Length | Wall Avg (ms) | Wall P95 (ms) | Req Avg (ms) | Req P95 (ms) | QPS |
|:-----------:|:------:|:-------------:|:-------------:|:------------:|:------------:|:---:|
| 1 | long | 6.81 | 7.15 | 6.78 | 7.10 | 184 |
| 3 | long | 12.37 | 13.09 | 11.09 | 12.74 | 303 |
| 5 | long | 13.97 | 15.48 | 12.98 | 16.10 | 447 |
| 10 | long | 21.09 | 23.20 | 18.99 | 22.20 | 593 |
| 20 | long | 33.04 | 35.59 | 28.82 | 34.61 | 757 |
| 32 | long | 45.44 | 48.85 | 39.13 | 46.42 | 880 |
| 1 | medium | 7.09 | 8.59 | 7.05 | 8.55 | 176 |
| 3 | medium | 10.99 | 11.81 | 9.99 | 11.58 | 341 |
| 5 | medium | 13.54 | 15.08 | 12.27 | 14.81 | 462 |
| 10 | medium | 20.33 | 23.36 | 18.76 | 23.70 | 615 |
| 20 | medium | 30.64 | 31.51 | 27.30 | 30.65 | 816 |
| 32 | medium | 44.48 | 46.41 | 39.97 | 46.91 | 899 |
| 1 | short | 18.56 | 19.63 | 18.52 | 19.59 | 67 |
| 3 | short | 11.26 | 12.31 | 9.86 | 12.07 | 333 |
| 5 | short | 13.80 | 15.36 | 12.33 | 14.97 | 453 |
| 10 | short | 19.61 | 20.02 | 17.87 | 19.81 | 637 |
| 20 | short | 32.01 | 34.48 | 29.35 | 35.10 | 781 |
| 32 | short | 43.31 | 44.66 | 37.72 | 43.16 | 924 |

### Reranker 数据

| 模型 | 框架 | Batch | Avg (ms) | P95 (ms) |
|------|------|:-----:|:--------:|:-------:|
| Qwen3-Reranker-0.6B | VLLM | 6 | 9.02 | 9.88 |
| Qwen3-Reranker-4B | VLLM | 6 | 17.93 | 19.38 |
| bge-reranker-v2-m3 | TEI | 6 | 18.78 | 32.39 |

---

*报告生成时间: 2026-05-12*