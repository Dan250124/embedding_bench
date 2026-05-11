# Qwen3-Embedding-0.6B 压测记录

## 硬件

- GPU: NVIDIA RTX 5090 (32GB GDDR7, Blackwell sm_120)
- CUDA: 12.x

## 模型

- Qwen3-Embedding-0.6B (310M params, 1.12 GiB)

## 延迟对比 (avg ms)

| batch | short TF | short vLLM | medium TF | medium vLLM | long TF | long vLLM |
|-------|---------|-----------|----------|------------|--------|-----------|
| 1 | 20.8 | 17.8 | 37.1 | 13.1 | 31.0 | 13.1 |
| 2 | 29.7 | 15.2 | 47.3 | 22.8 | 33.7 | 20.1 |
| 3 | 24.1 | 22.7 | 33.9 | 19.8 | 48.2 | 16.9 |
| 4 | 25.7 | 21.2 | 36.9 | 21.3 | 60.1 | 15.5 |
| 6 | 28.2 | 23.1 | 34.9 | 23.3 | 82.6 | 26.7 |
| 10 | 26.3 | 22.0 | 33.0 | 27.7 | 141.5 | 29.0 |
| 20 | 26.8 | 24.9 | 57.6 | 32.0 | 296.0 | 38.0 |
| 32 | 25.0 | 28.6 | 90.4 | 34.1 | 482.8 | 46.7 |
| 64 | 28.7 | 32.7 | 186.8 | 47.8 | 992.8 | 73.2 |

## 显存对比

### Transformers (实际占用)

| 文本长度 | batch=1 | batch=32 | batch=64 |
|---------|--------|---------|---------|
| short | 1768 MB | 1852 MB | 1910 MB |
| medium | 1796 MB | 3184 MB | 4552 MB |
| long | 1952 MB | 7538 MB | 13280 MB |

- 基础模型加载: ~1.7 GB
- 显存随 batch size 和文本长度线性增长
- batch=64 + long: 13280 MB (约 13 GB), 仍未超出 32GB

### vLLM

- **始终显示 30042 MB** (约 30 GB)
- 原因: `gpu_memory_utilization=0.9` 让 vLLM 预分配 90% 显存用于 KV cache
- 模型实际只占用 ~1.12 GiB
- 剩余空间用于 KV cache 和计算缓冲区
- 即使 batch=1 也显示 30GB, 无法反映真实显存消耗

### 显存记录问题

vLLM 的 `nvidia-smi` 采样无法准确反映模型实际显存占用, 因为:
1. vLLM 预分配大量 KV cache 内存
2. PyTorch 的 CUDA 内存分配器会缓存已释放的显存
3. `gpu_memory_utilization=0.9` 导致 vLLM 启动时就占满大部分显存

**建议**: 对 vLLM, 显存指标参考模型加载日志中的 `Model loading took X GiB memory`, 或在创建 LLM 前后分别采样 GPU 显存差值作为真实模型大小。
