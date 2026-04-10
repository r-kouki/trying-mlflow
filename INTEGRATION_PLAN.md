# Plan d'intégration MLflow dans stage_mlops
### Architecture cible pour la Streamlit app

---

## Vue d'ensemble

L'objectif est d'enrichir l'application Streamlit existante (`stage_mlops`) avec un module de suivi d'expériences MLflow, en **réutilisant tous les patterns existants** (credential hierarchy, caching, composants UI) et sans toucher aux pages actuelles.

```
stage_mlops (existant)             +  MLflow (nouveau)
────────────────────────────────      ──────────────────────────
home.py          ← inchangé          experiments.py   ← NOUVEAU
kpi.py           ← inchangé          model_registry.py ← NOUVEAU
manifest_browser ← inchangé          mlflow_client.py  ← NOUVEAU
training.py      ← +1 bouton         train-yolo.yml    ← NOUVEAU
config.py        ← +5 lignes
components.py    ← +2 entrées PAGES
docker-compose   ← +1 service
```

---

## Composants à créer / modifier

### 1. `docker-compose.yml` — Ajouter le service MLflow

**Fichier :** `/home/core/mlops/stage_mlops/docker-compose.yml`

Actuellement, ce fichier n'a qu'un seul service (`streamlit-app`). Il faut y ajouter `mlflow-server` pour qu'il tourne en parallèle sur la même machine.

```yaml
version: "3.8"

services:
  # ── Streamlit app (existant, inchangé) ─────────────────────────
  streamlit-app:
    image: ${GHCR_IMAGE}
    container_name: data-versioning-app
    ports:
      - "9753:8501"
    env_file:
      - .env (optional)
    volumes:
      - ./secrets:/secrets:ro
      - ./manifests:/app/manifests
    restart: unless-stopped

  # ── MLflow Server (NOUVEAU) ─────────────────────────────────────
  # Démarre avec : docker compose up -d mlflow-server
  # UI accessible sur : http://localhost:5000
  mlflow-server:
    image: python:3.11-slim
    container_name: mlflow-server
    ports:
      - "5000:5000"
    environment:
      - AWS_ACCESS_KEY_ID=${OVH_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${OVH_SECRET_ACCESS_KEY}
      - MLFLOW_S3_ENDPOINT_URL=${OVH_ENDPOINT}
      - AWS_DEFAULT_REGION=gra
    volumes:
      - mlflow_data:/mlflow
    command: >
      bash -c "pip install -q mlflow boto3 &&
               mkdir -p /root/.aws &&
               printf '[default]\ns3 =\n    signature_version = s3v4\n    addressing_style = path\n' > /root/.aws/config &&
               mlflow server --host 0.0.0.0 --port 5000
               --backend-store-uri sqlite:///mlflow/mlflow.db
               --default-artifact-root /mlflow/artifacts"
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"]
      interval: 15s
      timeout: 5s
      retries: 5
      start_period: 30s
    restart: unless-stopped

volumes:
  mlflow_data:
    name: stage_mlops_mlflow_data
```

> **Note :** Pour la prod, remplacer le `command` inline par un `Dockerfile.mlflow` dédié (comme dans `mlflow_yolo_poc/docker/`).

---

### 2. `app/mlflow_client.py` — Nouveau fichier

**Fichier à créer :** `/home/core/mlops/stage_mlops/app/mlflow_client.py`

Ce module centralise tous les appels à l'API MLflow. Il suit le même pattern de credential hierarchy que `config.py`.

```python
"""
Client MLflow pour l'app Streamlit.
Suit la même hiérarchie de credentials que config.py :
    session state → st.secrets → env var → défaut local
"""
import os

import mlflow
import pandas as pd
import streamlit as st
from mlflow.tracking import MlflowClient


# ── Résolution du Tracking URI ──────────────────────────────────

def get_tracking_uri() -> str:
    """Hiérarchie 3-niveaux pour le tracking URI."""
    return (
        st.session_state.get("mlflow_tracking_uri")
        or st.secrets.get("MLFLOW_TRACKING_URI", None)
        or os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
    )


def get_client() -> MlflowClient:
    """Crée un client MLflow connecté au tracking server."""
    mlflow.set_tracking_uri(get_tracking_uri())
    return MlflowClient()


# ── Chargement des données (avec cache Streamlit) ───────────────

@st.cache_data(ttl=60)
def list_experiments() -> list[dict]:
    """Retourne la liste des expériences MLflow."""
    client = get_client()
    experiments = client.search_experiments()
    return [
        {"experiment_id": e.experiment_id, "name": e.name}
        for e in experiments
        if e.name != "Default"
    ]


@st.cache_data(ttl=60)
def list_runs(experiment_name: str) -> pd.DataFrame:
    """
    Retourne tous les runs d'une expérience sous forme de DataFrame.
    Colonnes : run_id, run_name, status, start_time, params.*, metrics.*
    """
    client = get_client()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return pd.DataFrame()

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )

    rows = []
    for run in runs:
        row = {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "status": run.info.status,
            "start_time": pd.to_datetime(run.info.start_time, unit="ms"),
        }
        # Aplatir params et metrics directement dans le DataFrame
        row.update({f"param_{k}": v for k, v in run.data.params.items()})
        row.update({f"metric_{k}": v for k, v in run.data.metrics.items()})
        row.update({f"tag_{k}": v for k, v in run.data.tags.items()
                    if not k.startswith("mlflow.")})
        rows.append(row)

    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def get_run_metrics_history(run_id: str, metric_key: str) -> pd.DataFrame:
    """Retourne l'historique d'une métrique par epoch pour un run."""
    client = get_client()
    history = client.get_metric_history(run_id, metric_key)
    return pd.DataFrame([
        {"step": m.step, "value": m.value} for m in history
    ])


@st.cache_data(ttl=300)
def list_run_artifacts(run_id: str, path: str = "") -> list[str]:
    """Liste les artifacts d'un run (chemins relatifs)."""
    client = get_client()
    artifacts = client.list_artifacts(run_id, path)
    return [a.path for a in artifacts]


def compare_runs(run_ids: list[str]) -> pd.DataFrame:
    """Retourne un DataFrame de comparaison pour les runs sélectionnés."""
    client = get_client()
    rows = []
    for run_id in run_ids:
        run = client.get_run(run_id)
        row = {"run_id": run_id, "run_name": run.info.run_name}
        row.update(run.data.params)
        row.update(run.data.metrics)
        rows.append(row)
    return pd.DataFrame(rows)


# ── Model Registry ──────────────────────────────────────────────

@st.cache_data(ttl=120)
def list_registered_models() -> list[dict]:
    """Liste tous les modèles enregistrés dans le Model Registry."""
    client = get_client()
    models = client.search_registered_models()
    return [
        {
            "name": m.name,
            "latest_versions": [
                {"version": v.version, "stage": v.current_stage, "run_id": v.run_id}
                for v in m.latest_versions
            ],
        }
        for m in models
    ]


def promote_model(name: str, version: str, stage: str) -> None:
    """Transite un modèle vers un nouveau stage (Staging/Production/Archived)."""
    client = get_client()
    client.transition_model_version_stage(
        name=name, version=version, stage=stage
    )
    st.cache_data.clear()  # Invalide le cache
```

---

### 3. `app/pages/experiments.py` — Nouvelle page

**Fichier à créer :** `/home/core/mlops/stage_mlops/app/pages/experiments.py`

```python
"""
Page : Expériences MLflow
Browse et compare les runs d'entraînement YOLOv8.
"""

import plotly.express as px
import streamlit as st

from app.components import page_with_right_rail, render_page_header
from app.mlflow_client import (
    compare_runs,
    get_run_metrics_history,
    list_experiments,
    list_run_artifacts,
    list_runs,
)


def render(use_s3: bool = False) -> None:
    render_page_header("experiments")

    # ── Chargement des expériences ──────────────────────────────
    experiments = list_experiments()
    if not experiments:
        st.warning("Aucune expérience MLflow trouvée. "
                   "Le serveur MLflow est-il démarré sur http://localhost:5000 ?")
        return

    # ── Layout principal ────────────────────────────────────────
    main_col, rail_col = page_with_right_rail()

    with rail_col:
        st.caption("Filtres")
        selected_exp = st.selectbox(
            "Expérience",
            options=[e["name"] for e in experiments],
            index=0,
        )
        metric_sort = st.selectbox(
            "Trier par",
            ["metric_mAP50", "metric_mAP50-95", "start_time"],
        )

    # ── Données des runs ────────────────────────────────────────
    df = list_runs(selected_exp)
    if df.empty:
        main_col.info("Aucun run dans cette expérience.")
        return

    with rail_col:
        st.metric("Runs total", len(df))
        if "metric_mAP50" in df.columns:
            st.metric("Meilleur mAP50", f"{df['metric_mAP50'].max():.3f}")
        st.metric("Dernier run", df["start_time"].max().strftime("%d/%m %H:%M"))

    with main_col:
        # ── Tableau des runs ────────────────────────────────────
        st.subheader("Tous les runs")

        display_cols = ["run_name", "status", "start_time"]
        for col in ["metric_mAP50", "metric_mAP50-95", "metric_precision",
                    "metric_recall", "param_model", "param_epochs",
                    "param_optimizer", "param_lr0"]:
            if col in df.columns:
                display_cols.append(col)

        selected_rows = st.dataframe(
            df[display_cols].sort_values(
                metric_sort if metric_sort in df.columns else "start_time",
                ascending=False,
            ),
            use_container_width=True,
            on_select="rerun",
            selection_mode="multi-row",
        )

        # ── Comparaison des runs sélectionnés ───────────────────
        selected_indices = selected_rows.selection.rows if selected_rows else []
        if len(selected_indices) >= 2:
            st.divider()
            st.subheader(f"Comparaison — {len(selected_indices)} runs")

            selected_run_ids = df.iloc[selected_indices]["run_id"].tolist()
            comparison_df = compare_runs(selected_run_ids)

            # Graphes par epoch
            metric_keys = [
                ("epoch_mAP50", "mAP50 par epoch"),
                ("epoch_box_loss", "Box Loss par epoch"),
                ("epoch_cls_loss", "Cls Loss par epoch"),
            ]
            for metric_key, title in metric_keys:
                traces = []
                for run_id in selected_run_ids:
                    hist = get_run_metrics_history(run_id, metric_key)
                    if not hist.empty:
                        run_name = df[df["run_id"] == run_id]["run_name"].iloc[0]
                        hist["run"] = run_name
                        traces.append(hist)

                if traces:
                    import pandas as pd
                    combined = pd.concat(traces)
                    fig = px.line(
                        combined, x="step", y="value", color="run",
                        title=title,
                        labels={"step": "Epoch", "value": metric_key, "run": "Run"},
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # ── Artifacts du run sélectionné ─────────────────────────
        elif len(selected_indices) == 1:
            st.divider()
            run_id = df.iloc[selected_indices[0]]["run_id"]
            run_name = df.iloc[selected_indices[0]]["run_name"]
            st.subheader(f"Artifacts — {run_name}")

            artifacts = list_run_artifacts(run_id)
            for artifact_path in artifacts:
                st.text(f"📄 {artifact_path}")

            st.info("Pour télécharger les artifacts, ouvrir l'UI MLflow : "
                    "http://localhost:5000")
```

---

### 4. `app/pages/model_registry.py` — Nouvelle page

**Fichier à créer :** `/home/core/mlops/stage_mlops/app/pages/model_registry.py`

```python
"""
Page : Model Registry MLflow
Visualise et promeut les modèles enregistrés.
"""

import streamlit as st

from app.components import page_with_right_rail, render_page_header
from app.mlflow_client import list_registered_models, promote_model


def render(use_s3: bool = False) -> None:
    render_page_header("model_registry")

    main_col, rail_col = page_with_right_rail()

    with rail_col:
        st.caption("Model Registry")
        if st.button("🔄 Rafraîchir"):
            st.cache_data.clear()
            st.rerun()

    models = list_registered_models()

    with main_col:
        if not models:
            st.info("Aucun modèle enregistré dans le Model Registry. "
                    "Enregistrez un modèle depuis train.py avec mlflow.register_model().")
            st.code(
                'mlflow.register_model(\n'
                '    f"runs:/{run_id}/model/best.pt",\n'
                '    "yolov8-detection"\n'
                ')',
                language="python",
            )
            return

        for model in models:
            st.subheader(f"🏆 {model['name']}")

            for version_info in model["latest_versions"]:
                col1, col2, col3, col4 = st.columns([1, 2, 2, 2])
                col1.metric("Version", version_info["version"])
                col2.metric("Stage", version_info["stage"])
                col3.caption(f"Run ID: {version_info['run_id'][:8]}...")

                new_stage = col4.selectbox(
                    "Promouvoir vers",
                    ["Staging", "Production", "Archived"],
                    key=f"stage_{model['name']}_{version_info['version']}",
                )
                if col4.button(
                    "Appliquer",
                    key=f"promote_{model['name']}_{version_info['version']}",
                ):
                    promote_model(
                        model["name"],
                        version_info["version"],
                        new_stage,
                    )
                    st.success(
                        f"Modèle {model['name']} v{version_info['version']} "
                        f"→ {new_stage}"
                    )
                    st.rerun()

            st.divider()
```

---

### 5. `app/config.py` — Ajout de 8 lignes

**Fichier :** `/home/core/mlops/stage_mlops/app/config.py`

Ajouter à la fin du fichier (après les constantes existantes) :

```python
# ── MLflow ──────────────────────────────────────────────────────
# Même hiérarchie de credentials que pour S3 :
#   1. session state (saisie dans l'UI)
#   2. st.secrets (fichier /secrets/secrets.toml)
#   3. variable d'environnement
#   4. défaut : serveur local sur le réseau Docker
DEFAULT_MLFLOW_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", "http://mlflow-server:5000"
)


def mlflow_uri() -> str:
    """Retourne le tracking URI MLflow selon la hiérarchie de credentials."""
    return (
        st.session_state.get("mlflow_tracking_uri")
        or st.secrets.get("MLFLOW_TRACKING_URI", None)
        or DEFAULT_MLFLOW_URI
    )
```

---

### 6. `app/components.py` — Ajout de 2 entrées PAGES

**Fichier :** `/home/core/mlops/stage_mlops/app/components.py`

Trouver la liste `PAGES` et ajouter après la page `training` :

```python
# Dans la liste PAGES (après "training")
{"id": "experiments",    "icon": "🧪", "label": "Expériences MLflow"},
{"id": "model_registry", "icon": "🏆", "label": "Model Registry"},
```

Ajouter dans `PAGE_META` :

```python
"experiments": {
    "kicker": "MLflow",
    "title": "Expériences d'entraînement",
    "subtitle": "Compare les runs YOLOv8 — métriques, paramètres, artifacts",
},
"model_registry": {
    "kicker": "MLflow",
    "title": "Model Registry",
    "subtitle": "Gestion et promotion des modèles en production",
},
```

---

### 7. `app/pages/training.py` — Ajout du bouton GitHub Actions

**Fichier :** `/home/core/mlops/stage_mlops/app/pages/training.py`

À la fin de la fonction `render()`, après la génération du CSV, ajouter :

```python
# ── Lancer l'entraînement YOLO via GitHub Actions ───────────────
st.divider()
st.subheader("🚀 Lancer l'entraînement YOLO")

col1, col2, col3, col4 = st.columns(4)
yolo_model  = col1.selectbox("Modèle", ["yolov8n", "yolov8s", "yolov8m"])
yolo_epochs = col2.number_input("Epochs", min_value=5, max_value=100, value=10)
yolo_lr     = col3.number_input("Learning rate", min_value=0.0001,
                                 max_value=0.1, value=0.01, format="%.4f")
yolo_batch  = col4.number_input("Batch size", min_value=4, max_value=64, value=16)

if st.button("▶ Lancer via GitHub Actions", type="primary"):
    from app.github import dispatch_workflow
    dispatch_workflow(
        "train-yolo.yml",
        ref="main",
        inputs={
            "model": yolo_model,
            "epochs": str(yolo_epochs),
            "lr0": str(yolo_lr),
            "batch": str(yolo_batch),
            "dataset_version": str(selected_version),  # version sélectionnée plus haut
        },
    )
    st.success(
        "Pipeline d'entraînement déclenché ! "
        "Suivre l'avancement dans **Pipelines** puis voir les résultats dans **Expériences MLflow**."
    )
```

---

### 8. `.github/workflows/train-yolo.yml` — Nouveau workflow GitHub Actions

**Fichier à créer :** `/home/core/mlops/stage_mlops/.github/workflows/train-yolo.yml`

```yaml
name: Entraînement YOLO + Logging MLflow

on:
  workflow_dispatch:
    inputs:
      model:
        description: "Variante YOLOv8 (yolov8n / yolov8s / yolov8m)"
        default: "yolov8n"
        required: true
      epochs:
        description: "Nombre d'epochs"
        default: "10"
        required: true
      lr0:
        description: "Learning rate initial"
        default: "0.01"
        required: true
      batch:
        description: "Taille de batch"
        default: "16"
        required: true
      dataset_version:
        description: "Version du dataset (ex: v5)"
        default: "latest"
        required: false

jobs:
  train:
    runs-on: self-hosted           # Runner avec Docker disponible
    timeout-minutes: 60

    steps:
      - name: Checkout du code
        uses: actions/checkout@v4

      - name: Démarrer le serveur MLflow
        run: |
          docker compose -f mlflow_yolo_poc/docker/docker-compose.yml \
            --env-file mlflow_yolo_poc/.env \
            up -d mlflow-server
          # Attendre que le serveur soit prêt
          timeout 60 bash -c 'until docker inspect --format="{{.State.Health.Status}}" mlflow-server 2>/dev/null | grep -q healthy; do sleep 3; done'

      - name: Lancer l'entraînement
        run: |
          docker compose -f mlflow_yolo_poc/docker/docker-compose.yml \
            --env-file mlflow_yolo_poc/.env \
            run --rm \
            -e YOLO_MODEL=${{ inputs.model }}.pt \
            -e YOLO_EPOCHS=${{ inputs.epochs }} \
            -e YOLO_LR0=${{ inputs.lr0 }} \
            -e YOLO_BATCH=${{ inputs.batch }} \
            yolo-trainer

      - name: Résumé
        run: |
          echo "Entraînement terminé."
          echo "Voir les résultats sur http://localhost:5000 (ou ton serveur MLflow)"
```

---

## Flux de données complet

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit stage_mlops                        │
│                                                                 │
│  training.py                                                    │
│  ├── Sélectionne dataset version (annotation_v5)               │
│  ├── Génère training CSV → S3 (Manifests/Training/)            │
│  └── [Bouton] Déclenche GitHub Actions train-yolo.yml          │
│                          │                                      │
│                          ▼                                      │
│              GitHub Actions Runner (self-hosted)                │
│              ├── docker compose up mlflow-server               │
│              └── docker compose run yolo-trainer               │
│                          │           │                          │
│                          │           ▼                          │
│                          │    mlflow-server:5000               │
│                          │    ├── reçoit params/metrics        │
│                          │    └── stocke dans SQLite + volume  │
│                          │                                      │
│  experiments.py          │                                      │
│  ├── MlflowClient.search_runs()                                │
│  ├── Tableau des runs (mAP50, epochs, optimizer...)            │
│  ├── Graphes comparatifs (Plotly)                              │
│  └── Liens vers artifacts                                      │
│                                                                 │
│  model_registry.py                                              │
│  ├── Liste des modèles enregistrés                             │
│  └── Promotion Staging → Production                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Checklist d'implémentation

### Phase 1 — Infrastructure (P0, ~1h)
- [ ] Ajouter `mlflow-server` dans `stage_mlops/docker-compose.yml`
- [ ] Vérifier que `http://localhost:5000` est accessible depuis le conteneur Streamlit
- [ ] Ajouter `MLFLOW_TRACKING_URI` dans `secrets/secrets.toml.example`

### Phase 2 — Backend Python (P1, ~2h)
- [ ] Créer `app/mlflow_client.py`
- [ ] Modifier `app/config.py` (+ mlflow_uri())
- [ ] Modifier `app/components.py` (+ 2 PAGES entries)
- [ ] Ajouter `mlflow` dans `requirements.txt`

### Phase 3 — Pages UI (P1, ~3h)
- [ ] Créer `app/pages/experiments.py`
- [ ] Créer `app/pages/model_registry.py`
- [ ] Tester manuellement les deux pages

### Phase 4 — Intégration CI/CD (P2, ~1h)
- [ ] Créer `.github/workflows/train-yolo.yml`
- [ ] Modifier `app/pages/training.py` (bouton dispatch)
- [ ] Tester le dispatch depuis l'UI

### Phase 5 — Vérification (P0, ~30min)
- [ ] Toutes les pages existantes fonctionnent encore (home, kpi, manifest_browser...)
- [ ] La page Experiments affiche les runs du POC mlflow_yolo_poc
- [ ] Le bouton training.py dispatche un workflow visible dans Pipelines
- [ ] model_registry.py liste et promeut un modèle correctement

---

## Dépendances Python à ajouter dans `requirements.txt`

```
mlflow>=2.16.0     # Client MLflow (MlflowClient, search_runs, etc.)
```

> `boto3` est déjà dans les dépendances de `stage_mlops` via `dvc[s3]`.

---

## Notes pour l'équipe

### Pourquoi réutiliser les patterns existants ?
L'app `stage_mlops` a une architecture claire et cohérente. En suivant les mêmes patterns (credential hierarchy, `@st.cache_data`, `page_with_right_rail()`), les nouvelles pages s'intègrent naturellement et sont maintenables par toute l'équipe.

### Isolation vs intégration
Le POC `mlflow_yolo_poc` reste indépendant et sert de référence. L'intégration dans `stage_mlops` **lit** les données MLflow via le client, sans modifier le pipeline d'entraînement.

### Évolution vers la production
1. **Artifact store S3** : créer un compte OVH avec droits `s3:PutObject` sur le préfixe `mlflow-artifacts/` → changer `--default-artifact-root` dans docker-compose
2. **PostgreSQL** : remplacer SQLite par Postgres pour la concurrence multi-utilisateur
3. **Auth MLflow** : ajouter `--app-name basic-auth` si l'UI est exposée publiquement
