#!/bin/bash
# Start TEI Docker container for a given model
# Usage: ./start_tei.sh <model_path> <port> [container_name]

set -euo pipefail

MODEL_PATH="${1:?Usage: $0 <model_path> <port> [container_name]}"
PORT="${2:?Usage: $0 <model_path> <port> [container_name]}"
CONTAINER_NAME="${3:-tei-bench-${PORT}}"

IMAGE="ghcr.io/huggingface/text-embeddings-inference:latest"

echo "Starting TEI container: ${CONTAINER_NAME}"
echo "  Model: ${MODEL_PATH}"
echo "  Port: ${PORT}"
echo "  Image: ${IMAGE}"

# Ensure model path is absolute
MODEL_PATH="$(cd "$(dirname "$MODEL_PATH")" && pwd)/$(basename "$MODEL_PATH")"

docker run --gpus all \
    -p "${PORT}:80" \
    -v "${MODEL_PATH}:/data" \
    --name "${CONTAINER_NAME}" \
    --rm \
    -d \
    "${IMAGE}" \
    --model-id /data

# Health check - wait for the server to be ready
echo "Waiting for TEI to be ready on port ${PORT}..."
MAX_WAIT=300  # 5 minutes
ELAPSED=0
while true; do
    if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "TEI is ready on http://localhost:${PORT}"
        break
    fi
    if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
        echo "ERROR: TEI failed to start within ${MAX_WAIT}s"
        docker logs "${CONTAINER_NAME}" 2>&1 | tail -20
        exit 1
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    echo "  Still waiting... (${ELAPSED}s)"
done
