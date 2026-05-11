# Embedding / Reranker 推理框架测试方案

## 方案一：命令行手动测试

适用于快速验证单个模型在单个框架下的性能，灵活可控。

### 1. 启动推理服务

> 同一时间只运行一个模型服务，避免 GPU 资源争抢。

#### TEI (Docker)

**Embedding 模型：**

```bash
# Qwen3-Embedding-0.6B
docker run --gpus all -p 8080:80 \
  -v ~/.cache/modelscope/hub/Qwen/Qwen3-Embedding-0___6B:/data \
  --rm ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id /data

# Qwen3-Embedding-4B
docker run --gpus all -p 8080:80 \
  -v ~/.cache/modelscope/hub/Qwen/Qwen3-Embedding-4B:/data \
  --rm ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id /data

# bge-m3
docker run --gpus all -p 8080:80 \
  -v ~/.cache/modelscope/hub/BAAI/bge-m3:/data \
  --rm ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id /data
```

**Reranker 模型**（需加 `--workflow-classify`）：

```bash
# Qwen3-Reranker-0.6B
docker run --gpus all -p 8081:80 \
  -v ~/.cache/modelscope/hub/Qwen/Qwen3-Reranker-0___6B:/data \
  --rm ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id /data --workflow-classify

# Qwen3-Reranker-4B
docker run --gpus all -p 8081:80 \
  -v ~/.cache/modelscope/hub/Qwen/Qwen3-Reranker-4B:/data \
  --rm ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id /data --workflow-classify

# bge-reranker-v2-m3
docker run --gpus all -p 8081:80 \
  -v ~/.cache/modelscope/hub/BAAI/bge-reranker-v2-m3:/data \
  --rm ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id /data --workflow-classify
```

> **注意**：路径中的 `___` 是 ModelScope 对 `/` 的替换，先 `ls` 确认实际目录名。

#### vLLM

**Embedding 模型：**

```bash
# Qwen3-Embedding-0.6B (远程)
uv run vllm serve Qwen/Qwen3-Embedding-0.6B --host 0.0.0.0 --port 8080 --runner pooling

# Qwen3-Embedding-0.6B (本地)
uv run vllm serve ~/.cache/modelscope/hub/Qwen/Qwen3-Embedding-0___6B --host 0.0.0.0 --port 8080 --runner pooling

# Qwen3-Embedding-4B
uv run vllm serve ~/.cache/modelscope/hub/Qwen/Qwen3-Embedding-4B --host 0.0.0.0 --port 8080 --runner pooling

# bge-m3
uv run vllm serve ~/.cache/modelscope/hub/BAAI/bge-m3 --host 0.0.0.0 --port 8080 --runner pooling
```

**Reranker 模型：**

```bash
# Qwen3-Reranker-0.6B
uv run vllm serve ~/.cache/modelscope/hub/Qwen/Qwen3-Reranker-0___6B --host 0.0.0.0 --port 8081 --runner pooling

# Qwen3-Reranker-4B
uv run vllm serve ~/.cache/modelscope/hub/Qwen/Qwen3-Reranker-4B --host 0.0.0.0 --port 8081 --runner pooling

# bge-reranker-v2-m3
uv run vllm serve ~/.cache/modelscope/hub/BAAI/bge-reranker-v2-m3 --host 0.0.0.0 --port 8081 --runner pooling
```

### 2. 健康检查

服务启动后，新开终端确认就绪：

```bash
curl http://localhost:8080/health
```

返回 `200 OK` 即可开始测试。

### 3. 运行测试脚本

```bash
cd /mnt/e/labs/embedding_bench

# 测试 embedding 端口
uv run python scripts/test_server.py --base-url http://localhost:8080 --type embedding

# 测试 reranker 端口
uv run python scripts/test_server.py --base-url http://localhost:8081 --type reranker

# 两个都测
uv run python scripts/test_server.py --base-url http://localhost:8080 --type both

# 自定义测试轮数（默认 20）
uv run python scripts/test_server.py --base-url http://localhost:8080 --rounds 30
```

### 4. 测试内容

| 项目 | 说明 |
|------|------|
| batch size | 1, 2, 3, 4, 8, 12, 32 |
| 文本长度 | short (~32 tokens), medium (~256 tokens), long (~1024 tokens) |
| 轮数 | 20 轮 warmup(3) + measure(20)，去掉最高最低各 1 轮 |
| 指标 | avg, P95 延迟 (ms) |
| 共计 | 7 batch × 3 length = 21 组 embedding 测试 |

---

## 方案二：全自动批量测试

适用于一次性跑完所有模型 × 所有框架的组合，自动管理 TEI 容器的启停。

```bash
cd /mnt/e/labs/embedding_bench

# 全量测试（所有模型 × 所有框架）
uv run python main.py

# 只测指定模型
uv run python main.py --models bge-m3 Qwen3-Embedding-0.6B

# 只测指定框架
uv run python main.py --frameworks tei vllm

# 只跑延迟测试
uv run python main.py --test-type latency

# 自定义 TEI 端口
uv run python main.py --tei-port 9000
```

自动流程：启动 TEI Docker → 健康检查 → 延迟/吞吐测试 → 采集 GPU 指标 → 停止容器 → 生成 CSV + 图表 + 汇总表。

结果输出到 `results/` 目录。
