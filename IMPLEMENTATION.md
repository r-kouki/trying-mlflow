# MLflow + YOLOv8 + ZenML — Implementation Handoff

## Project Overview
YOLOv8 object detection training pipeline with:
- **MLflow 3.x** for experiment tracking, artifact storage, and model registry
- **ZenML** for pipeline orchestration (`load_data → train_model → save_artifacts`)
- **OVH S3** for remote artifact storage (SigV4 + path-style required)
- Everything **Dockerized** in separate stacks (old `docker/`, new `docker-zenml/`)

---

## Repo Structure

```
mlflow_yolo_poc/
├── src/
│   ├── pipeline.py       ← ZenML 3-step pipeline (main entrypoint for docker-zenml)
│   ├── train.py          ← Standalone local training script (no ZenML)
│   ├── config.py         ← All env-var-based config
│   └── data_loader.py    ← S3 download + dataset YAML generation
├── configs/
│   ├── dataset.yaml      ← COCO8 dataset config
│   └── yolo.yaml         ← YOLO hyperparams (model, epochs, batch, lr0, imgsz, optimizer)
├── docker/               ← OLD stack (MLflow 2.x, standalone trainer, port 5000)
│   ├── docker-compose.yml
│   ├── Dockerfile.mlflow
│   └── Dockerfile.trainer
├── docker-zenml/         ← NEW stack (MLflow 3.x + ZenML, port 5002)
│   ├── docker-compose.yml
│   ├── Dockerfile.mlflow
│   ├── Dockerfile.pipeline
│   └── entrypoint.sh
└── .env                  ← OVH credentials (not in git)
```

## Git Branches

| Branch | Purpose |
|---|---|
| `main` | Original Docker stack (old `docker/` folder, MLflow 2.x) |
| `local-venv` | Simplified `train.py` without git tracking / French comments |
| `zenml-integration` | **Current branch** — ZenML pipeline + MLflow 3.x + new `docker-zenml/` |

---

## What Is Working (as of last session)

### Full pipeline runs end-to-end

```bash
cd /home/core/Pictures/mlopsss/mlflow_yolo_poc

# Start servers (MLflow on 5002, ZenML on 8080)
docker compose -f docker-zenml/docker-compose.yml --env-file .env up -d mlflow-server zenml-server

# Run pipeline
docker compose -f docker-zenml/docker-compose.yml --env-file .env --profile train run --rm zenml-pipeline
```

**Verified outputs:**
- `load_data` — downloads from OVH S3 or uses local cache at `data_local/`
- `train_model` — trains YOLOv8n for 10 epochs, logs to MLflow, registers `yolo-detector v1` in Model Registry
- `save_artifacts` — uploads 17 files to `s3://stage-mlops/yolo-runs/<timestamp>/` via boto3

### MLflow UI
- URL: `http://localhost:5002`
- Shows: `yolo-poc` experiment, per-epoch metrics, plots, artifacts, `yolo-detector` model versions

### ZenML Server
- URL: `http://localhost:8080`
- Version: 0.94.2 (latest)
- Credentials: `admin` / `admin`

---

## What Is NOT Yet Verified

### ZenML Dashboard Showing Pipeline Runs

The ZenML server connection from the pipeline was just fixed (see below). Not yet verified that pipeline runs appear in the dashboard after the fix.

**What was fixed:**
1. ZenML 0.94.2 requires server **activation** on first boot (no user is created from env vars alone)  
   → `entrypoint.sh` now calls `PUT /api/v1/activate` if `info.active == False`
2. API key response structure: the `key` value is at `body.key`, not top-level `key`  
   → `entrypoint.sh` now parses `json.load(sys.stdin)['body']['key']`
3. `zenml connect` is deprecated in 0.94.2 — replaced by `zenml login <url> --api-key <key>`

**To verify it works:** rebuild and run the pipeline, then check `http://localhost:8080` for pipeline runs.

```bash
docker compose -f docker-zenml/docker-compose.yml --env-file .env build zenml-pipeline
docker compose -f docker-zenml/docker-compose.yml --env-file .env --profile train run --rm zenml-pipeline
```

---

## Key Design Decisions

### Why `network_mode: host` for both services?
MLflow 3.x has DNS rebinding protection — it rejects HTTP requests where the `Host:` header isn't `localhost`/`127.0.0.1`. Using Docker bridge networking means the container hostname appears in the `Host:` header → `403 Invalid Host header`. Host networking makes `localhost:5002` work from both host and pipeline container.

### Why `mlflow.pyfunc.log_model()` instead of `mlflow.log_artifact()`?
MLflow 3.x `register_model()` requires a proper MLflow model (with `MLmodel` manifest). Raw `log_artifact()` just stores a file — no manifest. The fix wraps `best.pt` in a minimal `PythonModel` subclass before logging.

### Why ZenML `[server]` extra?
`zenml` alone is missing `sqlmodel` and `sqlalchemy_utils` that ZenML's internal metadata store needs. `zenml[server]` pulls in all required dependencies.

### OVH S3 requirements
- `signature_version = s3v4` and `addressing_style = path` in `/root/.aws/config`
- Both `OVH_ACCESS_KEY → AWS_ACCESS_KEY_ID` and `OVH_SECRET_KEY → AWS_SECRET_ACCESS_KEY` must be mapped (config.py reads `AWS_*` prefix)
- Endpoint URL must have no trailing slash in `.env`

---

## `.env` file (not in git — on host at `/home/core/Pictures/mlopsss/mlflow_yolo_poc/.env`)

```
OVH_ENDPOINT=https://s3.gra.io.cloud.ovh.net
OVH_ACCESS_KEY=<key>
OVH_SECRET_KEY=<secret>
BUCKET_NAME=stage-mlops
DATA_PREFIX=data/
```

---

## Relevant File Paths

| File | Key detail |
|---|---|
| [docker-zenml/entrypoint.sh](docker-zenml/entrypoint.sh) | **Fix here** — JSON path for API key parsing |
| [docker-zenml/docker-compose.yml](docker-zenml/docker-compose.yml) | All three services: zenml-server, mlflow-server, zenml-pipeline |
| [docker-zenml/Dockerfile.pipeline](docker-zenml/Dockerfile.pipeline) | `zenml[server]==0.94.2`, `mlflow>=3.0.0`, ultralytics base |
| [src/pipeline.py](src/pipeline.py) | ZenML steps + `mlflow.pyfunc.log_model` for registry |
| [src/config.py](src/config.py) | All config constants read from env |
| [src/data_loader.py](src/data_loader.py) | S3 download + YAML generation |
