import csv
import os

import mlflow
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

from config import (
    LOCAL_DATA_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    RUNS_DIR,
    load_dataset_config,
    load_yolo_config,
)
from data_loader import download_dataset, prepare_dataset_yaml

# Disable ultralytics built-in MLflow callback — we handle logging ourselves
SETTINGS.update({"mlflow": False})


def train(overrides: dict | None = None):
    yolo_cfg = load_yolo_config()
    dataset_cfg = load_dataset_config()
    if overrides:
        yolo_cfg.update(overrides)

    data_dir = LOCAL_DATA_DIR if (LOCAL_DATA_DIR / "images").exists() else download_dataset()
    dataset_yaml = prepare_dataset_yaml(data_dir, dataset_cfg)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"{yolo_cfg['model']}-ep{yolo_cfg['epochs']}-bs{yolo_cfg['batch']}"):
        mlflow.log_params(yolo_cfg)

        results = YOLO(yolo_cfg["model"]).train(
            data=str(dataset_yaml),
            epochs=yolo_cfg["epochs"],
            batch=yolo_cfg["batch"],
            imgsz=yolo_cfg["imgsz"],
            lr0=yolo_cfg["lr0"],
            optimizer=yolo_cfg["optimizer"],
            project=str(RUNS_DIR),
            name="detect",
            exist_ok=True,
            verbose=True,
        )

        # Final metrics
        metrics = results.results_dict if hasattr(results, "results_dict") else {}
        mlflow.log_metrics({
            "mAP50":    metrics.get("metrics/mAP50(B)", 0.0),
            "mAP50-95": metrics.get("metrics/mAP50-95(B)", 0.0),
            "precision": metrics.get("metrics/precision(B)", 0.0),
            "recall":   metrics.get("metrics/recall(B)", 0.0),
            "box_loss": metrics.get("train/box_loss", 0.0),
            "cls_loss": metrics.get("train/cls_loss", 0.0),
            "dfl_loss": metrics.get("train/dfl_loss", 0.0),
        })

        # Per-epoch metrics from results.csv
        results_csv = RUNS_DIR / "detect" / "results.csv"
        if results_csv.exists():
            with open(results_csv) as f:
                reader = csv.DictReader(f)
                # Strip whitespace from column names (YOLO pads them)
                reader.fieldnames = [k.strip() for k in reader.fieldnames]
                for row in reader:
                    row = {k.strip(): v for k, v in row.items()}
                    epoch = int(float(row.get("epoch", 0)))
                    epoch_metrics = {}
                    for col, name in [
                        ("metrics/mAP50(B)",    "epoch_mAP50"),
                        ("metrics/mAP50-95(B)", "epoch_mAP50-95"),
                        ("train/box_loss",      "epoch_box_loss"),
                        ("train/cls_loss",      "epoch_cls_loss"),
                        ("train/dfl_loss",      "epoch_dfl_loss"),
                    ]:
                        if col in row:
                            try:
                                epoch_metrics[name] = float(row[col])
                            except ValueError:
                                pass
                    if epoch_metrics:
                        mlflow.log_metrics(epoch_metrics, step=epoch)

        # Artifacts
        train_dir = RUNS_DIR / "detect"
        for artifact, folder in [
            (train_dir / "confusion_matrix.png", "plots"),
            (train_dir / "results.png",          "plots"),
            (train_dir / "weights" / "best.pt",  "model"),
        ]:
            if artifact.exists():
                mlflow.log_artifact(str(artifact), folder)

        print("[train] MLflow run complete.")


if __name__ == "__main__":
    overrides = {}
    env_map = {
        "YOLO_MODEL":     ("model",     str),
        "YOLO_EPOCHS":    ("epochs",    int),
        "YOLO_BATCH":     ("batch",     int),
        "YOLO_LR0":       ("lr0",       float),
        "YOLO_IMGSZ":     ("imgsz",     int),
        "YOLO_OPTIMIZER": ("optimizer", str),
    }
    for env_var, (key, cast) in env_map.items():
        if val := os.getenv(env_var):
            overrides[key] = cast(val)

    train(overrides=overrides or None)
