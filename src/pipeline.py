import csv
import datetime
import os
from pathlib import Path

import boto3
import mlflow
from ultralytics import YOLO
from ultralytics.utils import SETTINGS
from zenml import pipeline, step
from zenml.integrations.mlflow.flavors.mlflow_experiment_tracker_flavor import (
    MLFlowExperimentTrackerSettings,
)

from config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    BUCKET_NAME,
    LOCAL_DATA_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    OVH_ENDPOINT,
    RUNS_DIR,
    load_dataset_config,
    load_yolo_config,
)
from data_loader import download_dataset, prepare_dataset_yaml


@step
def load_data() -> Path:
    if LOCAL_DATA_DIR.exists() and (LOCAL_DATA_DIR / "images").exists():
        print(f"[load_data] Local data found at {LOCAL_DATA_DIR}")
        return LOCAL_DATA_DIR
    print("[load_data] Downloading from S3 ...")
    return download_dataset()


@step(
    experiment_tracker="mlflow_tracker",
    settings={"experiment_tracker": MLFlowExperimentTrackerSettings(
        experiment_name=MLFLOW_EXPERIMENT_NAME,
    )},
)
def train_model(data_dir: Path) -> dict:
    SETTINGS.update({"mlflow": False})

    yolo_cfg = load_yolo_config()
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
            yolo_cfg[key] = cast(val)

    dataset_yaml = prepare_dataset_yaml(data_dir, load_dataset_config())
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(
        run_name=f"{yolo_cfg['model']}-ep{yolo_cfg['epochs']}-bs{yolo_cfg['batch']}"
    ):
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

        metrics = results.results_dict if hasattr(results, "results_dict") else {}
        final_metrics = {
            "mAP50":     metrics.get("metrics/mAP50(B)", 0.0),
            "mAP50-95":  metrics.get("metrics/mAP50-95(B)", 0.0),
            "precision": metrics.get("metrics/precision(B)", 0.0),
            "recall":    metrics.get("metrics/recall(B)", 0.0),
            "box_loss":  metrics.get("train/box_loss", 0.0),
            "cls_loss":  metrics.get("train/cls_loss", 0.0),
            "dfl_loss":  metrics.get("train/dfl_loss", 0.0),
        }
        mlflow.log_metrics(final_metrics)

        # Per-epoch metrics from results.csv
        results_csv = RUNS_DIR / "detect" / "results.csv"
        if results_csv.exists():
            with open(results_csv) as f:
                reader = csv.DictReader(f)
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

        # MLflow Model Registry — register best.pt as a versioned model
        run_id = mlflow.active_run().info.run_id
        registered = mlflow.register_model(f"runs:/{run_id}/model", "yolo-detector")
        print(f"[train_model] Registered: yolo-detector v{registered.version}")

    return final_metrics


@step
def save_artifacts(metrics: dict) -> None:
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    s3_prefix = f"yolo-runs/{ts}/"
    train_dir = RUNS_DIR / "detect"

    if not train_dir.exists():
        print(f"[save_artifacts] {train_dir} not found, skipping.")
        return

    s3 = boto3.client(
        "s3",
        endpoint_url=OVH_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name="gra",
        config=boto3.session.Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
        ),
    )

    uploaded = 0
    for f in train_dir.rglob("*"):
        if f.is_file():
            key = f"{s3_prefix}{f.relative_to(train_dir)}"
            s3.upload_file(str(f), BUCKET_NAME, key)
            print(f"[save_artifacts] → s3://{BUCKET_NAME}/{key}")
            uploaded += 1

    print(f"[save_artifacts] {uploaded} files uploaded. Metrics: {metrics}")


@pipeline(name="yolo_training_pipeline")
def yolo_training_pipeline():
    data_dir = load_data()
    metrics = train_model(data_dir)
    save_artifacts(metrics)


if __name__ == "__main__":
    yolo_training_pipeline()
