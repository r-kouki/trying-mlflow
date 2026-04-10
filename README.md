# POC MLflow + YOLOv8 — Architecture conteneurisée

Pipeline d'entraînement YOLOv8 minimaliste et conteneurisé, avec tracking d'expériences via MLflow et stockage des artifacts sur OVH S3.

## Pourquoi ce POC ?

Démontrer que **MLflow** est la meilleure option pour notre stack MLOps :
- Self-hosted, open-source, RGPD-compliant
- Intégration native avec OVH S3 (notre provider existant)
- Compatible avec notre pipeline DVC + Docker

---

## Architecture

```
                ┌──────────────────────────────────┐
                │   OVH S3 Bucket (unique)         │
                │   mlops-bucket/                  │
                │   ├── data/          ← DVC       │
                │   │   ├── clips/                 │
                │   │   └── annotations/           │
                │   └── mlflow-artifacts/  ← MLflow│
                │       ├── runs/                  │
                │       └── models/                │
                └────────┬─────────────┬───────────┘
                         │             │
                   read  │             │ read/write
                         │             │
          ┌──────────────▼──┐    ┌─────▼──────────────┐
          │  Container:     │    │  Container:        │
          │  yolo-trainer   │───▶│  mlflow-server     │
          │                 │    │  (port 5000)       │
          │  - ultralytics  │    │  - backend: SQLite │
          │  - mlflow client│    │  - artifacts: S3   │
          └─────────────────┘    └────────────────────┘
                   │                      ▲
                   └──────────────────────┘
                      docker network
```

**Principe** : un seul bucket S3, deux préfixes logiques séparés :
- `data/` — géré par DVC en amont (datasets versionnés)
- `mlflow-artifacts/` — géré par MLflow (poids, plots, métriques)

---

## Quickstart

### 1. Initialiser le projet

```bash
make init
```

Cela crée un environnement virtuel local (`.venv`) si nécessaire et prépare le fichier `.env` sans écraser un fichier déjà présent.

### 2. Installer les dépendances Python locales

```bash
make install
```

### 3. Configurer les credentials

Éditer `.env` avec vos credentials OVH S3 si besoin.

### 4. Démarrer le serveur MLflow

```bash
make up
```

UI accessible sur **http://localhost:5000**

### 5. Lancer un entraînement

```bash
make train
```

### 6. Lancer la démo (3 runs comparatifs)

```bash
make demo
```

Cela lance 3 entraînements avec des configurations différentes :
| Run | Modèle | Epochs | LR | Optimizer |
|-----|--------|--------|----|-----------|
| 1 | YOLOv8n | 10 | 0.01 | SGD |
| 2 | YOLOv8s | 10 | 0.01 | SGD |
| 3 | YOLOv8n | 10 | 0.001 | AdamW |

### 7. Comparer dans l'UI MLflow

Ouvrir http://localhost:5000 → Expérience `yolo-poc` → Sélectionner les 3 runs → Compare.

---

## Structure du projet

```
mlflow_yolo_poc/
├── docker/
│   ├── docker-compose.yml      # Orchestration des 2 services
│   ├── Dockerfile.mlflow       # Image serveur MLflow
│   └── Dockerfile.trainer      # Image entraînement YOLO
├── src/
│   ├── config.py               # Configuration centralisée
│   ├── data_loader.py          # Téléchargement dataset depuis S3
│   └── train.py                # Entraînement + logging MLflow
├── configs/
│   ├── yolo_config.yaml        # Hyperparamètres YOLO
│   └── dataset.yaml            # Format dataset YOLO
├── .env.example                # Template credentials
├── requirements.txt            # Dépendances Python
├── run_demo.sh                 # Script démo 3 runs
└── README.md
```

---

## Ce qui est loggé dans MLflow

| Catégorie | Données |
|-----------|---------|
| **Paramètres** | model, epochs, batch, imgsz, lr0, optimizer |
| **Métriques** | mAP50, mAP50-95, precision, recall, box/cls/dfl_loss |
| **Métriques/epoch** | Courbes mAP et loss par epoch (step) |
| **Artifacts** | confusion_matrix.png, results.png, best.pt |
| **Tags** | dataset_version, yolo_variant, git_commit |

---

## Comparatif MLflow vs Weights & Biases

| Critère | MLflow | Weights & Biases |
|---------|--------|------------------|
| **Hébergement** | Self-hosted (on-premise) ou managed | SaaS uniquement (serveurs W&B) |
| **Coût** | Gratuit (open-source) | Gratuit limité, payant en équipe ($50+/user/mois) |
| **Licence** | Apache 2.0 (open-source) | Propriétaire |
| **RGPD / Souveraineté** | Données restent sur notre infra OVH | Données transitent par les serveurs W&B (US) |
| **Model Registry** | Intégré nativement | Intégré, mais dépend du SaaS |
| **Intégration S3 (OVH)** | Native via boto3, endpoint configurable | Possible mais pas le workflow par défaut |
| **Backend store** | SQLite, MySQL, PostgreSQL | Base propriétaire W&B |
| **Compatibilité DVC** | Complémentaire (data=DVC, expériences=MLflow) | Overlap avec DVC, risque de doublon |
| **Compatibilité Docker** | Conteneurisable facilement | Client Python uniquement, serveur SaaS |
| **GitHub Actions** | Intégration via CLI `mlflow` | Intégration via CLI `wandb` |
| **UI de comparaison** | Fonctionnelle, sobre | Plus riche visuellement |
| **Courbe d'apprentissage** | Simple, API Python standard | Simple aussi, plus "magique" |

### Verdict pour notre stack

MLflow est le choix logique pour notre équipe :

1. **Souveraineté des données** — Tout reste sur OVH, aucune donnée ne quitte notre infra. Argument RGPD décisif.
2. **Coût** — Zéro licence, pas de facturation par utilisateur.
3. **Complémentarité avec DVC** — DVC gère le versioning des datasets, MLflow gère le tracking des expériences. Pas de chevauchement.
4. **Un seul bucket S3** — Données et artifacts cohabitent dans le même bucket OVH avec des préfixes séparés.
5. **Docker-native** — Le serveur MLflow tourne dans un conteneur, s'intègre parfaitement dans notre stack Docker Compose existante.

W&B reste supérieur sur l'UI et les features collaboratives avancées, mais ces avantages ne justifient pas la perte de contrôle sur les données ni le coût récurrent.

---

## Configuration avancée

### Utiliser un GPU

Dans `docker/Dockerfile.trainer`, remplacer l'image de base :

```dockerfile
# CPU
FROM python:3.11-slim

# GPU (NVIDIA)
FROM ultralytics/ultralytics:latest
```

Et ajouter dans `docker-compose.yml` sous le service `yolo-trainer` :

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### Passer en PostgreSQL (production)

Pour un déploiement plus robuste, remplacer SQLite par PostgreSQL :

```yaml
# Ajouter dans docker-compose.yml
postgres:
  image: postgres:15
  environment:
    POSTGRES_DB: mlflow
    POSTGRES_USER: mlflow
    POSTGRES_PASSWORD: mlflow
  volumes:
    - pg_data:/var/lib/postgresql/data
```

Et modifier la commande MLflow :
```
--backend-store-uri postgresql://mlflow:mlflow@postgres:5432/mlflow
```

---

## Arrêter les services

```bash
docker compose -f docker/docker-compose.yml down

# Avec suppression des volumes (perte de la base MLflow)
docker compose -f docker/docker-compose.yml down -v
```
