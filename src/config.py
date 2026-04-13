"""
Configuration centralisée du POC MLflow + YOLOv8.
Lit les variables d'environnement et expose les paramètres S3 / MLflow.
"""

import os
from pathlib import Path

import yaml


# ── Chemins projet ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
LOCAL_DATA_DIR = PROJECT_ROOT / "data_local"  # données téléchargées depuis S3
RUNS_DIR = PROJECT_ROOT / "runs"              # résultats YOLO (hors data_local pour éviter les soucis de permissions)

# ── OVH S3 ──────────────────────────────────────────────────────
OVH_ENDPOINT = os.getenv("OVH_ENDPOINT", "https://s3.gra.io.cloud.ovh.net")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
BUCKET_NAME = os.getenv("BUCKET_NAME", "mlops-bucket")
DATA_PREFIX = os.getenv("DATA_PREFIX", "data/")

# ── MLflow ──────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "yolo-poc")


def load_yolo_config() -> dict:
    """Charge les hyperparamètres YOLO depuis configs/yolo_config.yaml."""
    config_path = CONFIGS_DIR / "yolo_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Fichier de config introuvable : {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_dataset_config() -> dict:
    """Charge la config dataset YOLO depuis configs/dataset.yaml."""
    config_path = CONFIGS_DIR / "dataset.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Fichier dataset.yaml introuvable : {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
