# Embedding / Reranker 推理框架对比分析

## 测试环境

- **GPU**: NVIDIA RTX 5090 (32GB GDDR7, Blackwell)
- **框架**: Transformers (基线) vs vLLM (pooling runner)
- **模型**: 6 个 (3 Embedding + 3 Reranker)
- **维度**: 3 种文本长度 (short/medium/long) × 9 种 batch size (1~64)
- **指标**: 平均延迟 (ms)、P95 延迟、GPU 显存

---

## 一、Embedding 模型对比

### 1.1 单请求延迟 (batch=1)

| 模型 | Transformers | vLLM | vLLM 领先 |
|------|------------|------|-----------|
| Qwen3-Embedding-0.6B | 20.9~22.9ms | 12.7~13.5ms | 1.5~1.8x |
| Qwen3-Embedding-4B | 27.6~60.4ms | 15.7~17.0ms | 1.7~3.6x |
| bge-m3 | 7.3~10.6ms | 9.2~12.2ms | TF 更快 0.8~0.9x |

**结论**: 单请求场景下，Qwen3 系列 vLLM 有 1.5~3.6x 加速；bge-m3 因为模型小、计算量低，Transformers 反而更快（无引擎开销）。

### 1.2 批量延迟 (batch=32, medium 文本)

| 模型 | Transformers | vLLM | vLLM 加速比 |
|------|------------|------|-------------|
| Qwen3-Embedding-0.6B | 95.4ms | 37.5ms | **2.5x** |
| Qwen3-Embedding-4B | 452.2ms | 45.8ms | **9.9x** |
| bge-m3 | 49.1ms | 51.0ms | 持平 |

### 1.3 大批量长文本 (batch=64, long 文本) — 极端场景

| 模型 | Transformers | vLLM | vLLM 加速比 |
|------|------------|------|-------------|
| Qwen3-Embedding-0.6B | 1049.9ms | 76.6ms | **13.7x** |
| Qwen3-Embedding-4B | 4204.7ms | 110.1ms | **38.2x** |
| bge-m3 | 478.8ms | 264.8ms | **1.8x** |

**结论**: 这是 vLLM 的主场。批量+长文本场景下，vLLM 对 Qwen3-4B 有 **38 倍加速**，从 4.2 秒降到 110ms。核心原因是 vLLM 的 continuous batching 和优化的 CUDA kernel 能高效并行处理长序列。

### 1.4 吞吐量估算 (batch=64, medium 文本)

| 模型 | Transformers | vLLM |
|------|------------|------|
| Qwen3-Embedding-0.6B | 318 req/s | 1194 req/s |
| Qwen3-Embedding-4B | 68 req/s | 920 req/s |
| bge-m3 | 692 req/s | 703 req/s |

---

## 二、Reranker 模型对比

### 2.1 单请求延迟 (batch=1)

| 模型 | Transformers | vLLM | 说明 |
|------|------------|------|------|
| Qwen3-Reranker-0.6B | 23.6~24.7ms | 19.2~31.6ms | 接近，TF 短文本略优 |
| Qwen3-Reranker-4B | 30.6~32.0ms | 18.8~50.5ms | vLLM 首次请求慢 (冷启动) |
| bge-reranker-v2-m3 | 9.7~12.3ms | 8.8~12.2ms | 基本持平 |

### 2.2 批量延迟 (batch=6)

| 模型 | Transformers | vLLM | vLLM 加速比 |
|------|------------|------|-------------|
| Qwen3-Reranker-0.6B | 33.2~34.4ms | 23.0~24.8ms | **1.4x** |
| Qwen3-Reranker-4B | 116.7~118.3ms | 27.9~28.1ms | **4.2x** |
| bge-reranker-v2-m3 | 16.3~24.9ms | 16.3~18.1ms | 持平 |

**结论**: Reranker 场景中，Qwen3-Reranker-4B 在 vLLM 下加速最显著 (4.2x)。bge-reranker-v2-m3 两个框架表现接近。值得注意的是 Qwen3-Reranker-4B 在 Transformers 下 batch=6 时延迟飙升至 118ms，而 vLLM 仅 28ms，说明 vLLM 的 pooling runner 对 4B 级 reranker 有显著优化。

---

## 三、GPU 显存对比

### 3.1 Transformers 显存 (随 batch/文本长度动态增长)

| 模型 | 短文本 bs=1 | 长文本 bs=64 | 峰值 |
|------|------------|-------------|------|
| Qwen3-Embedding-0.6B | 1,768 MB | 13,280 MB | 13.3 GB |
| Qwen3-Embedding-4B | 8,360 MB | 29,538 MB | 29.5 GB |
| bge-m3 | 1,722 MB | 4,442 MB | 4.4 GB |

### 3.2 vLLM 显存 (预分配，不随 batch 变化)

| 模型 | 固定占用 |
|------|---------|
| Qwen3-Embedding-0.6B | 30,671 MB (~30 GB) |
| Qwen3-Embedding-4B | 30,769 MB (~30 GB) |
| bge-m3 | 2,663 MB (~2.6 GB) |

**关键发现**:
- vLLM 的 `gpu_memory_utilization=0.9` 导致启动即预分配 90% 显存 (30GB)，**无法反映真实显存消耗**
- Transformers 显存随 batch size 和文本长度线性增长，数据更有参考价值
- bge-m3 (0.6B) 在 vLLM 下仅占 2.6GB，说明 vLLM 对小模型也能合理管理显存
- Qwen3-Embedding-4B 在 Transformers + long + batch=64 时已达 29.5GB，接近 32GB 上限

---

## 四、核心结论

### 框架选型建议

| 场景 | 推荐框架 | 理由 |
|------|---------|------|
| 小模型 (bge 系列) + 低并发 | **Transformers** | 单请求更快，显存占用低，无引擎开销 |
| Qwen3 系列 + 高并发/大 batch | **vLLM** | 批量场景加速 2.5~38x，延迟更稳定 |
| 长文本 embedding | **vLLM** | 长序列处理优势巨大 (Qwen3-4B 加速 38x) |
| 显存敏感/多模型共享 GPU | **Transformers** | 按需占用，vLLM 预分配 30GB |
| Reranker (4B 级) | **vLLM** | Qwen3-Reranker-4B 加速 4.2x |

### 模型选型建议

| 维度 | 推荐 | 说明 |
|------|------|------|
| 延迟优先 (短文本) | bge-m3 | 单请求 7.3ms，全场最快 |
| 吞吐优先 | Qwen3-Embedding-0.6B + vLLM | 1194 req/s |
| 效果优先 (4B) | Qwen3-Embedding-4B + vLLM | 920 req/s，效果优于 0.6B |
| 显存受限 | bge-m3 | 峰值仅 4.4GB |
| Reranker 效果+速度 | Qwen3-Reranker-0.6B | 两个框架都稳定在 24~35ms |

### 注意事项

1. **vLLM 显存数据不可用**: `nvidia-smi` 采样到的 30GB 是预分配值，非真实消耗。建议通过 vLLM 日志中的 `Model loading took X GiB memory` 获取真实数据
2. **vLLM 首次请求延迟高**: 部分场景 batch=1 时 vLLM 比 Transformers 慢 (如 Qwen3-Reranker-4B 首次 50ms vs 30ms)，属引擎初始化开销
3. **Transformers 大 batch OOM 风险**: Qwen3-Embedding-4B + long + batch=64 已占 29.5GB，再大可能 OOM
