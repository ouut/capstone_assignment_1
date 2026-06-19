#!/usr/bin/env bash
# ============================================================
# docker-build.sh — Build FashionMNIST image and smoke test
# ============================================================
set -euo pipefail

IMAGE="fashionmnist-server"
TAG="${TAG:-latest}"
PORT="${PORT:-8000}"
CONTAINER="fashionmnist-test-$$"
MODEL_DIR="$(cd "$(dirname "$0")" && pwd)/models"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

cleanup() {
    echo -e "\n${YELLOW}[cleanup]${NC} stopping container..."
    docker stop "$CONTAINER" 2>/dev/null || true
    docker rm "$CONTAINER" 2>/dev/null || true
}
trap cleanup EXIT

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Build & Test: $IMAGE:$TAG${NC}"
echo -e "${GREEN}========================================${NC}"

# ---- Step 1: Build ----
echo -e "\n${YELLOW}[1/4]${NC} Building image..."
docker build -t "$IMAGE:$TAG" .

# ---- Step 2: Start ----
echo -e "\n${YELLOW}[2/4]${NC} Starting container..."
docker run -d --name "$CONTAINER" \
    -p "$PORT:8000" \
    -v "$MODEL_DIR:/app/models:ro" \
    "$IMAGE:$TAG"

# ---- Step 3: Wait ----
echo -e "\n${YELLOW}[3/4]${NC} Waiting for server..."
for i in $(seq 1 60); do
    if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo -e "${GREEN}  Server ready${NC}"
        break
    fi
    sleep 1
done

# ---- Step 4: Smoke tests ----
echo -e "\n${YELLOW}[4/4]${NC} Running smoke tests..."

echo -e "\n--- /health ---"
curl -s "http://localhost:$PORT/health" | python3 -m json.tool

echo -e "\n--- /classes ---"
curl -s "http://localhost:$PORT/classes" | python3 -m json.tool

echo -e "\n--- /predict-json ---"
PAYLOAD=$(python3 -c "
import json, random
random.seed(42)
pixels = [random.randint(0, 255) for _ in range(784)]
print(json.dumps({'image': pixels}))
")
curl -s -X POST "http://localhost:$PORT/predict-json" \
    -H 'Content-Type: application/json' \
    -d "$PAYLOAD" | python3 -m json.tool

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  All tests passed!${NC}"
echo -e "${GREEN}  Image: $IMAGE:$TAG${NC}"
echo -e "${GREEN}========================================${NC}"
