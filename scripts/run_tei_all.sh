#!/bin/bash
# Run TEI tests for all models sequentially
set -euo pipefail

HF_CACHE="$HOME/.cache/huggingface/hub"
IMAGE="ghcr.io/huggingface/text-embeddings-inference:120-1.9"
BASE_PORT=8080
PYTHON="uv run python"

declare -A MODELS
MODELS=(
    ["Qwen3-Embedding-0.6B"]="models--Qwen--Qwen3-Embedding-0.6B:embedding"
    ["Qwen3-Embedding-4B"]="models--Qwen--Qwen3-Embedding-4B:embedding"
    ["bge-m3"]="models--BAAI--bge-m3:embedding"
    ["Qwen3-Reranker-0.6B"]="models--Qwen--Qwen3-Reranker-0.6B:reranker"
    ["Qwen3-Reranker-4B"]="models--Qwen--Qwen3-Reranker-4B:reranker"
    ["bge-reranker-v2-m3"]="models--BAAI--bge-reranker-v2-m3:reranker"
)

PORT=$BASE_PORT

for NAME in "${!MODELS[@]}"; do
    IFS=':' read -r DIR TYPE <<< "${MODELS[$NAME]}"
    MODEL_PATH="$HF_CACHE/$DIR/snapshots/$(ls "$HF_CACHE/$DIR/snapshots/")"
    CONTAINER="tei-bench-$PORT"

    echo ""
    echo "============================================================"
    echo "TEI | $NAME ($TYPE) | port=$PORT"
    echo "============================================================"

    # Start container
    echo "Starting TEI container..."
    docker run --gpus all \
        -p "$PORT:80" \
        -v "$MODEL_PATH:/data" \
        --name "$CONTAINER" \
        --rm \
        -d \
        "$IMAGE" \
        --model-id /data

    # Wait for ready
    echo "Waiting for TEI to be ready..."
    ELAPSED=0
    while true; do
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo "TEI ready on port $PORT"
            break
        fi
        if [ "$ELAPSED" -ge 300 ]; then
            echo "ERROR: TEI failed to start within 300s"
            docker logs "$CONTAINER" 2>&1 | tail -20
            exit 1
        fi
        sleep 5
        ELAPSED=$((ELAPSED + 5))
    done

    # Run benchmark for this model
    $PYTHON main.py \
        --frameworks tei \
        --models "$NAME" \
        --tei-port "$PORT" \
        --output-dir results

    # Stop container
    echo "Stopping TEI container..."
    docker stop "$CONTAINER" 2>/dev/null || true
    sleep 2

    PORT=$((PORT + 1))
done

echo ""
echo "All TEI tests completed!"
