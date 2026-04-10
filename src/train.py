"""
Script principal d'entraînement YOLOv8 avec logging MLflow.
Télécharge les données depuis S3, entraîne le modèle,
et log tous les paramètres / métriques / artifacts vers MLflow.
"""

import os
import subprocess
from pathlib import Path

import mlflow
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

from config import (
    LOCAL_DATA_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    load_dataset_config,
    load_yolo_config,
)
from data_loader import download_dataset, prepare_dataset_yaml


# Désactiver le callback MLflow intégré à ultralytics pour éviter le double-logging
# Notre train.py gère entièrement le logging MLflow
SETTINGS.update({"mlflow": False})


def get_git_commit() -> str:
    """Récupère le hash du commit git courant (ou 'unknown')."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def train(overrides: dict | None = None):
    """
    Lance un entraînement YOLOv8 complet avec logging MLflow.

    Args:
        overrides: Dict d'hyperparamètres à surcharger (optionnel).
                   Ex: {"epochs": 5, "batch": 8, "model": "yolov8s.pt"}
    """
    # ── 1. Charger la configuration ─────────────────────────────
    yolo_config = load_yolo_config()
    dataset_config = load_dataset_config()

    # Appliquer les surcharges si fournies
    if overrides:
        yolo_config.update(overrides)

    model_variant = yolo_config.get("model", "yolov8n.pt")
    epochs = yolo_config.get("epochs", 10)
    batch = yolo_config.get("batch", 16)
    imgsz = yolo_config.get("imgsz", 640)
    lr0 = yolo_config.get("lr0", 0.01)
    optimizer = yolo_config.get("optimizer", "SGD")

    # ── 2. Préparer les données ───────────────────────────────────
    # Mode local : si data_local/ existe déjà (COCO8 pré-téléchargé), on l'utilise
    # Mode S3 : sinon, on télécharge depuis le bucket OVH
    if LOCAL_DATA_DIR.exists() and (LOCAL_DATA_DIR / "images").exists():
        print(f"[train] Données locales trouvées dans {LOCAL_DATA_DIR}")
        data_dir = LOCAL_DATA_DIR
    else:
        data_dir = download_dataset()
    dataset_yaml = prepare_dataset_yaml(data_dir, dataset_config)

    # ── 3. Configurer MLflow ────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"{model_variant}-ep{epochs}-bs{batch}"):
        # ── 4. Logger les paramètres ────────────────────────────
        mlflow.log_params({
            "model": model_variant,
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "lr0": lr0,
            "optimizer": optimizer,
        })

        # Tags pour le filtrage dans l'UI
        mlflow.set_tags({
            "dataset_version": yolo_config.get("dataset_version", "v1.0"),
            "yolo_variant": model_variant.replace("yolov8", "").replace(".pt", ""),
            "git_commit": get_git_commit(),
        })

        # ── 5. Entraîner le modèle ─────────────────────────────
        print(f"[train] Démarrage : {model_variant}, {epochs} epochs, batch={batch}")
        model = YOLO(model_variant)

        results = model.train(
            data=str(dataset_yaml),
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            lr0=lr0,
            optimizer=optimizer,
            project=str(LOCAL_DATA_DIR / "runs"),
            name="detect",
            exist_ok=True,
            verbose=True,
        )

        # ── 6. Logger les métriques finales ─────────────────────
        # Récupérer les métriques depuis les résultats YOLO
        metrics = results.results_dict if hasattr(results, "results_dict") else {}

        metrics_to_log = {
            "mAP50": metrics.get("metrics/mAP50(B)", 0.0),
            "mAP50-95": metrics.get("metrics/mAP50-95(B)", 0.0),
            "precision": metrics.get("metrics/precision(B)", 0.0),
            "recall": metrics.get("metrics/recall(B)", 0.0),
            "box_loss": metrics.get("train/box_loss", 0.0),
            "cls_loss": metrics.get("train/cls_loss", 0.0),
            "dfl_loss": metrics.get("train/dfl_loss", 0.0),
        }
        mlflow.log_metrics(metrics_to_log)
        print(f"[train] Métriques finales : {metrics_to_log}")

        # ── 7. Logger les métriques par epoch (CSV) ─────────────
        results_csv = LOCAL_DATA_DIR / "runs" / "detect" / "results.csv"
        if results_csv.exists():
            import csv
            with open(results_csv, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    epoch = int(row.get("epoch", row.get("                  epoch", 0)))
                    epoch_metrics = {}
                    for col, mlflow_name in [
                        ("metrics/mAP50(B)", "epoch_mAP50"),
                        ("metrics/mAP50-95(B)", "epoch_mAP50-95"),
                        ("train/box_loss", "epoch_box_loss"),
                        ("train/cls_loss", "epoch_cls_loss"),
                        ("train/dfl_loss", "epoch_dfl_loss"),
                    ]:
                        # Les colonnes CSV YOLO ont parfois des espaces
                        val = row.get(col) or row.get(f"                   {col}")
                        if val is not None:
                            try:
                                epoch_metrics[mlflow_name] = float(val)
                            except ValueError:
                                pass
                    if epoch_metrics:
                        mlflow.log_metrics(epoch_metrics, step=epoch)

        # ── 8. Logger les artifacts ─────────────────────────────
        train_dir = LOCAL_DATA_DIR / "runs" / "detect"

        try:
            # Matrice de confusion
            confusion = train_dir / "confusion_matrix.png"
            if confusion.exists():
                mlflow.log_artifact(str(confusion), "plots")

            # Courbes de résultats
            results_png = train_dir / "results.png"
            if results_png.exists():
                mlflow.log_artifact(str(results_png), "plots")

            # Meilleurs poids du modèle
            best_weights = train_dir / "weights" / "best.pt"
            if best_weights.exists():
                mlflow.log_artifact(str(best_weights), "model")
                print(f"[train] Poids best.pt loggés ({best_weights.stat().st_size / 1e6:.1f} MB)")

            print("[train] Run MLflow terminé avec succès.")
        except Exception as e:
            print(f"[train] Warning : upload artifacts S3 échoué ({e})")
            print("[train] Métriques et paramètres sont loggés. Vérifier les credentials S3.")


if __name__ == "__main__":
    # Supporte les surcharges via variables d'env pour run_demo.sh
    overrides = {}
    if os.getenv("YOLO_MODEL"):
        overrides["model"] = os.getenv("YOLO_MODEL")
    if os.getenv("YOLO_EPOCHS"):
        overrides["epochs"] = int(os.getenv("YOLO_EPOCHS"))
    if os.getenv("YOLO_BATCH"):
        overrides["batch"] = int(os.getenv("YOLO_BATCH"))
    if os.getenv("YOLO_LR0"):
        overrides["lr0"] = float(os.getenv("YOLO_LR0"))
    if os.getenv("YOLO_IMGSZ"):
        overrides["imgsz"] = int(os.getenv("YOLO_IMGSZ"))
    if os.getenv("YOLO_OPTIMIZER"):
        overrides["optimizer"] = os.getenv("YOLO_OPTIMIZER")

    train(overrides=overrides or None)
