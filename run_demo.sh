#!/bin/bash
# ── Démo : 3 runs avec hyperparamètres différents ──────────────
# Lance 3 entraînements séquentiels pour comparer dans l'UI MLflow.
# Usage : ./run_demo.sh (depuis la racine du projet)

set -e

COMPOSE_FILE="docker/docker-compose.yml"
ENV_FILE=".env"

# Vérifier que le fichier .env existe
if [ ! -f "$ENV_FILE" ]; then
    echo "Erreur : fichier .env introuvable."
    echo "Copie .env.example vers .env et renseigne tes credentials OVH."
    exit 1
fi

echo "=========================================="
echo "  POC MLflow + YOLOv8 — Démo 3 runs"
echo "=========================================="

# S'assurer que le serveur MLflow tourne
echo ""
echo "[1/4] Démarrage du serveur MLflow..."
docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d mlflow-server

# Attendre que le serveur soit prêt
echo "[1/4] Attente du healthcheck MLflow..."
timeout 60 bash -c 'until docker inspect --format="{{.State.Health.Status}}" mlflow-server 2>/dev/null | grep -q healthy; do sleep 2; done' \
    || echo "Warning: healthcheck timeout, on tente quand même..."

echo ""
echo "UI MLflow disponible sur : http://localhost:5000"
echo ""

# ── Run 1 : YOLOv8n (nano), baseline ───────────────────────────
echo "[2/4] Run 1 — YOLOv8n, 10 epochs, lr=0.01"
docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" run --rm \
    -e YOLO_MODEL=yolov8n.pt \
    -e YOLO_EPOCHS=10 \
    -e YOLO_BATCH=16 \
    -e YOLO_LR0=0.01 \
    yolo-trainer

# ── Run 2 : YOLOv8s (small), plus gros modèle ─────────────────
echo ""
echo "[3/4] Run 2 — YOLOv8s, 10 epochs, lr=0.01"
docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" run --rm \
    -e YOLO_MODEL=yolov8s.pt \
    -e YOLO_EPOCHS=10 \
    -e YOLO_BATCH=16 \
    -e YOLO_LR0=0.01 \
    yolo-trainer

# ── Run 3 : YOLOv8n avec learning rate plus élevé ──────────────
echo ""
echo "[4/4] Run 3 — YOLOv8n, 10 epochs, lr=0.001 + AdamW"
docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" run --rm \
    -e YOLO_MODEL=yolov8n.pt \
    -e YOLO_EPOCHS=10 \
    -e YOLO_BATCH=16 \
    -e YOLO_LR0=0.001 \
    -e YOLO_OPTIMIZER=AdamW \
    yolo-trainer

echo ""
echo "=========================================="
echo "  3 runs terminés !"
echo "  Compare les résultats : http://localhost:5000"
echo "=========================================="
