#!/bin/bash
# Stop TEI Docker container
# Usage: ./stop_tei.sh <container_name>

set -euo pipefail

CONTAINER_NAME="${1:?Usage: $0 <container_name>}"

if docker ps -q -f "name=${CONTAINER_NAME}" | grep -q .; then
    echo "Stopping container: ${CONTAINER_NAME}"
    docker stop "${CONTAINER_NAME}"
    echo "Container stopped."
else
    echo "Container ${CONTAINER_NAME} is not running."
fi
