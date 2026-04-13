#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# run_local.sh — Run MLflow + YOLOv8 training WITHOUT Docker
# Uses a local Python venv and file-based MLflow storage.
#
# Usage:
#   ./run_local.sh                        # default config
#   YOLO_EPOCHS=3 YOLO_BATCH=4 ./run_local.sh
# ─────────────────────────────────────────────────────────────
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"
MLFLOW_PORT=5001
MLFLOW_STORE="$PROJECT_ROOT/mlruns"

# ── 1. Create or reuse venv ───────────────────────────────────
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "[local] Creating venv at .venv ..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# ── 2. Install dependencies ───────────────────────────────────
echo "[local] Installing dependencies ..."
pip install -q -r "$PROJECT_ROOT/requirements.txt"

# ── 3. Load .env.local if it exists, else fall back to .env ──
if [ -f "$PROJECT_ROOT/.env.local" ]; then
    echo "[local] Loading .env.local"
    set -o allexport
    source "$PROJECT_ROOT/.env.local"
    set +o allexport
elif [ -f "$PROJECT_ROOT/.env" ]; then
    echo "[local] Loading .env"
    set -o allexport
    source "$PROJECT_ROOT/.env"
    set +o allexport
fi

# ── 4. Force local MLflow tracking (file-based, no Docker) ───
export MLFLOW_TRACKING_URI="http://localhost:$MLFLOW_PORT"

# ── 5. Start MLflow server in background ─────────────────────
echo "[local] Starting MLflow server on http://localhost:$MLFLOW_PORT ..."
mlflow server \
    --host 0.0.0.0 \
    --port "$MLFLOW_PORT" \
    --backend-store-uri "sqlite:///$MLFLOW_STORE/mlflow.db" \
    --default-artifact-root "$MLFLOW_STORE/artifacts" \
    &
MLFLOW_PID=$!

# Wait for the server to be ready
echo "[local] Waiting for MLflow server ..."
for i in $(seq 1 20); do
    if python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:$MLFLOW_PORT/health')" 2>/dev/null; then
        break
    fi
    sleep 1
done

echo "[local] MLflow UI → http://localhost:$MLFLOW_PORT"

# ── 6. Run training ───────────────────────────────────────────
echo "[local] Starting training ..."
cd "$PROJECT_ROOT/src"
python train.py

# ── 7. Keep server alive so you can browse the UI ─────────────
echo ""
echo "[local] Training done. MLflow UI still running at http://localhost:$MLFLOW_PORT"
echo "[local] Press Ctrl+C to stop."
wait $MLFLOW_PID
