# Guide complet du POC MLflow + YOLOv8
### Pour les débutants en MLOps

---

## Avant de commencer : le vocabulaire essentiel

| Terme | Explication simple | Analogie |
|-------|--------------------|----------|
| **MLflow** | Un "carnet de bord" pour tes entraînements — il enregistre tous les paramètres, scores et fichiers de chaque expérience | Un cahier de labo où tu notes chaque essai |
| **YOLOv8** | Un modèle de détection d'objets — il repère et encadre des objets dans des images en temps réel | Un gardien de sécurité qui regarde une caméra et dit "il y a une personne là, un chien là" |
| **Docker / Conteneur** | Une "boîte" isolée qui contient un programme et tout ce dont il a besoin, sans dépendre de la machine hôte | Un appartement meublé — tu arrives, tout est déjà là, rien ne manque |
| **docker-compose** | Un outil pour lancer plusieurs conteneurs ensemble et les faire communiquer | Un chef d'orchestre qui démarre tous les musiciens en même temps |
| **OVH S3** | Un service de stockage de fichiers dans le cloud (compatible API Amazon S3) | Un disque dur sur internet, accessible depuis n'importe où |
| **DVC** | Un outil de versioning pour les données — comme Git, mais pour des fichiers volumineux (vidéos, images) | Git pour les datasets |
| **mAP50** | Mean Average Precision à seuil IoU 0.5 — le score principal de détection d'objets. Plus c'est haut (max 1.0), mieux c'est | La note sur 100 de ton modèle |
| **epoch** | Un passage complet sur tout le dataset d'entraînement | Une lecture complète d'un livre |
| **hyperparamètre** | Un réglage de l'entraînement (ex: vitesse d'apprentissage, taille des batches) que tu fixes avant de commencer | Les réglages du four avant de cuire un gâteau |

---

## Qu'est-ce qu'on a construit ?

Un **pipeline d'entraînement automatisé** pour un modèle de détection d'objets YOLOv8, avec :
- **Tracking complet** de chaque expérience via MLflow
- **Architecture conteneurisée** (2 Docker containers qui se parlent)
- **Données** lues depuis le bucket OVH S3 partagé avec DVC
- **UI de comparaison** pour voir quel réglage donne les meilleurs résultats

### Pourquoi MLflow plutôt que W&B ?

Sans MLflow, voici ce qui arrive :
```
Run 1 : lr=0.01, mAP=0.72  ← noté sur un Post-it
Run 2 : lr=0.001, mAP=0.81 ← perdu dans un terminal
Run 3 : lr=0.01 + AdamW ... ← tu ne sais plus les paramètres exacts
```

Avec MLflow :
```
http://localhost:5000 → tableau comparatif, graphes, téléchargement du meilleur modèle
```

---

## L'architecture expliquée simplement

```
                ┌─────────────────────────────────┐
                │   OVH S3 Bucket (stage-mlops)   │
                │                                 │
                │   data/          ← images YOLO  │
                │   ├── images/train/              │
                │   ├── images/val/                │
                │   └── labels/                   │
                └────────────┬────────────────────┘
                             │ lit les données
                             │
          ┌──────────────────▼──┐     ┌──────────────────────┐
          │  yolo-trainer       │────▶│  mlflow-server       │
          │                     │     │                      │
          │  1. télécharge data │     │  - reçoit les logs   │
          │  2. entraîne YOLO   │     │  - stocke métriques  │
          │  3. envoie résultats│     │  - stocke artifacts  │
          └─────────────────────┘     │  - UI sur port 5000  │
                                      └──────────────────────┘
                  │                            │
                  └────── réseau Docker ───────┘
                           (mlops-net)
```

**Analogie :** Le `yolo-trainer` est un chef cuisinier qui prépare des recettes (entraîne le modèle). À chaque essai, il note les ingrédients et les résultats dans un cahier (MLflow). Le cahier est géré par un serveur dédié (`mlflow-server`) dans une autre pièce. Les deux se parlent via un réseau Docker privé.

---

## Chaque fichier expliqué

### Structure du projet

```
mlflow_yolo_poc/
├── docker/                  ← tout ce qui concerne les conteneurs
│   ├── docker-compose.yml   ← "la recette" pour lancer les 2 conteneurs
│   ├── Dockerfile.mlflow    ← comment construire l'image du serveur MLflow
│   └── Dockerfile.trainer   ← comment construire l'image d'entraînement
│
├── src/                     ← le code Python
│   ├── config.py            ← lit les variables d'environnement
│   ├── data_loader.py       ← télécharge les données depuis S3
│   └── train.py             ← lance l'entraînement et log dans MLflow
│
├── configs/                 ← les réglages (sans toucher au code)
│   ├── yolo_config.yaml     ← hyperparamètres YOLO (epochs, lr, batch...)
│   └── dataset.yaml         ← description du dataset (classes, chemins)
│
├── data_local/              ← données téléchargées localement (ignoré par git)
│   ├── images/train/        ← 4 images d'entraînement (COCO8)
│   ├── images/val/          ← 4 images de validation
│   ├── labels/              ← annotations au format YOLO (.txt)
│   └── runs/detect/         ← résultats de l'entraînement
│       ├── results.csv      ← métriques par epoch
│       ├── results.png      ← graphes des courbes
│       ├── confusion_matrix.png
│       └── weights/
│           ├── best.pt      ← meilleur modèle (6.5 MB)
│           └── last.pt      ← modèle à la dernière epoch
│
├── .env                     ← credentials (NE PAS COMMITER !)
├── .env.example             ← template vide à copier
├── requirements.txt         ← dépendances Python
├── run_demo.sh              ← script de démo (3 runs en séquence)
├── README.md                ← documentation principale
└── GUIDE.md                 ← ce fichier
```

---

### `docker/docker-compose.yml` — L'orchestrateur

Ce fichier dit à Docker : "lance ces 2 services ensemble, connecte-les sur le même réseau".

**Points clés :**
- `mlflow-server` : démarre automatiquement, expose le port 5000
- `yolo-trainer` : ne démarre **pas** automatiquement (`profiles: ["train"]`), on le lance à la demande
- `depends_on` : le trainer attend que le serveur MLflow soit `healthy` avant de démarrer
- `mlflow_data` : un volume Docker persistant (la base SQLite MLflow survit aux redémarrages)
- `mlops-net` : réseau privé → le trainer peut contacter `http://mlflow-server:5000` par son nom

### `docker/Dockerfile.mlflow` — L'image du serveur

Construit l'image Docker pour le serveur MLflow :
1. Part de `python:3.11-slim` (Python minimal)
2. Installe `mlflow` et `boto3`
3. Configure `~/.aws/config` pour forcer SigV4 (requis par OVH S3)
4. Expose le port 5000

### `docker/Dockerfile.trainer` — L'image d'entraînement

Construit l'image Docker pour l'entraînement :
1. Part de `ultralytics/ultralytics:latest` — PyTorch + YOLO **déjà inclus** (évite le téléchargement de 900 MB de CUDA)
2. Ajoute `mlflow`, `boto3`, `pyyaml`
3. Configure `~/.aws/config` pour OVH S3

### `src/config.py` — La configuration centralisée

Lit toutes les variables d'environnement du `.env` et les expose comme des constantes Python.

```python
OVH_ENDPOINT  = "https://s3.gra.io.cloud.ovh.net"
BUCKET_NAME   = "stage-mlops"
DATA_PREFIX   = "data/"
MLFLOW_TRACKING_URI = "http://mlflow-server:5000"
```

Fournit aussi deux fonctions utilitaires : `load_yolo_config()` et `load_dataset_config()`.

### `src/data_loader.py` — Le téléchargeur de données

Se connecte à OVH S3 via `boto3` et télécharge tout ce qui se trouve sous `data/` dans le bucket vers `data_local/`.

**Optimisation importante :** si `data_local/` existe déjà et contient des fichiers, il ne re-télécharge pas (évite les téléchargements répétés à chaque run).

### `src/train.py` — Le cœur du pipeline

C'est le script principal. Voici ce qu'il fait dans l'ordre :

```python
# 1. Désactiver le callback MLflow intégré à ultralytics
#    (évite le double-logging)
SETTINGS.update({"mlflow": False})

# 2. Charger les configs
yolo_config = load_yolo_config()   # epochs, batch, lr, optimizer...
dataset_config = load_dataset_config()  # classes, chemins

# 3. Télécharger les données si nécessaire
data_dir = download_dataset()
dataset_yaml = prepare_dataset_yaml(data_dir, dataset_config)

# 4. Ouvrir un run MLflow
mlflow.set_tracking_uri("http://mlflow-server:5000")
mlflow.set_experiment("yolo-poc")
with mlflow.start_run(run_name="yolov8n-ep10-bs16"):

    # 5. Logger les paramètres AVANT l'entraînement
    mlflow.log_params({"epochs": 10, "batch": 16, "lr0": 0.01, ...})
    mlflow.set_tags({"yolo_variant": "n", "git_commit": "abc1234"})

    # 6. Entraîner le modèle
    model = YOLO("yolov8n.pt")
    results = model.train(data=dataset_yaml, epochs=10, ...)

    # 7. Logger les métriques finales
    mlflow.log_metrics({"mAP50": 0.889, "precision": 0.778, ...})

    # 8. Logger les artifacts (fichiers)
    mlflow.log_artifact("confusion_matrix.png", "plots")
    mlflow.log_artifact("results.png", "plots")
    mlflow.log_artifact("weights/best.pt", "model")
```

### `configs/yolo_config.yaml` — Les hyperparamètres

```yaml
model: yolov8n.pt   # n=nano (le plus rapide), s=small, m=medium
epochs: 10          # nombre de passages sur le dataset
batch: 16           # images traitées en parallèle
imgsz: 640          # taille de redimensionnement des images
lr0: 0.01           # vitesse d'apprentissage initiale
optimizer: SGD      # algorithme d'optimisation
```

Ces valeurs peuvent être surchargées via les variables d'env `YOLO_MODEL`, `YOLO_EPOCHS`, etc. (utilisé par `run_demo.sh`).

### `configs/dataset.yaml` — La description du dataset

```yaml
train: images/train   # dossier des images d'entraînement
val: images/val       # dossier des images de validation
names:                # classes détectées
  0: person
  1: bicycle
  # ... 80 classes COCO standard
```

### `.env` — Les credentials (secret !)

```bash
OVH_ACCESS_KEY=20bd66...   # clé d'accès S3
OVH_SECRET_KEY=0bc1fb...   # clé secrète S3
OVH_ENDPOINT=https://s3.gra.io.cloud.ovh.net
BUCKET_NAME=stage-mlops
DATA_PREFIX=data/
```

> ⚠️ Ce fichier contient des secrets. Il est dans `.gitignore` et ne doit **jamais** être committé.

---

## Ce qui se passe quand tu lances un entraînement

```
$ docker compose run --rm yolo-trainer
```

```
Étape 1 — Docker démarre le conteneur yolo-trainer
         └→ vérifie que mlflow-server est healthy

Étape 2 — Python démarre train.py
         └→ charge yolo_config.yaml
         └→ charge dataset.yaml

Étape 3 — Vérification des données locales
         └→ si data_local/ existe : rien à faire (déjà téléchargé)
         └→ sinon : boto3 se connecte à S3 et télécharge data/

Étape 4 — Ouverture d'un run MLflow
         └→ mlflow.set_tracking_uri("http://mlflow-server:5000")
         └→ mlflow.start_run(run_name="yolov8n-ep10-bs16")
         └→ log des paramètres (epochs, batch, lr, optimizer...)

Étape 5 — Entraînement YOLOv8
         ┌─────────────────────────────────────────┐
         │  Epoch 1/10:                            │
         │    box_loss=1.43  cls_loss=3.69  ...    │
         │    mAP50=0.77  precision=0.84  ...      │
         │                                         │
         │  Epoch 2/10:                            │
         │    box_loss=1.07  cls_loss=2.93  ...    │
         │    mAP50=0.88  precision=0.85  ...      │
         │  ...                                    │
         └─────────────────────────────────────────┘
         └→ meilleur modèle sauvé dans weights/best.pt

Étape 6 — Log des métriques finales dans MLflow
         └→ mAP50=0.889, mAP50-95=0.646, precision=0.778...

Étape 7 — Log des métriques par epoch (pour les graphes)
         └→ lecture du fichier results.csv
         └→ mlflow.log_metrics({"epoch_mAP50": 0.77}, step=1)
         └→ mlflow.log_metrics({"epoch_mAP50": 0.88}, step=2)...

Étape 8 — Log des artifacts (fichiers)
         └→ confusion_matrix.png → artifacts/plots/
         └→ results.png         → artifacts/plots/
         └→ best.pt             → artifacts/model/

Résultat : "Run terminé avec succès"
           → http://mlflow-server:5000/#/experiments/1/runs/...
```

---

## Ce qui est loggé dans MLflow et pourquoi

### Paramètres (enregistrés une fois, avant l'entraînement)

| Paramètre | Valeur exemple | Pourquoi c'est important |
|-----------|---------------|--------------------------|
| `model` | `yolov8n.pt` | Permet de savoir quelle architecture a été utilisée |
| `epochs` | `10` | Nombre de passages sur le dataset |
| `batch` | `16` | Impacte la stabilité et la vitesse |
| `imgsz` | `640` | Résolution des images — impacte la précision |
| `lr0` | `0.01` | Learning rate — réglage le plus critique |
| `optimizer` | `SGD` | Algorithme d'optimisation |

### Tags (métadonnées)

| Tag | Valeur exemple | Utilité |
|-----|---------------|---------|
| `dataset_version` | `v1.0` | Traçabilité des données utilisées |
| `yolo_variant` | `n` | Filtrage rapide dans l'UI (nano/small/medium) |
| `git_commit` | `abc1234` | Reproductibilité du code exact |

### Métriques finales

| Métrique | Signification | Objectif |
|----------|--------------|---------|
| `mAP50` | Précision moyenne à IoU≥0.5 | Maximiser (max=1.0) |
| `mAP50-95` | Précision moyenne à IoU 0.5→0.95 | Maximiser |
| `precision` | Taux de vraies détections parmi toutes les détections | Maximiser |
| `recall` | Taux d'objets détectés parmi tous les objets présents | Maximiser |
| `box_loss` | Erreur de localisation des boîtes | Minimiser |
| `cls_loss` | Erreur de classification | Minimiser |
| `dfl_loss` | Distribution Focal Loss | Minimiser |

### Artifacts (fichiers)

| Fichier | Contenu | Utilité |
|---------|---------|---------|
| `confusion_matrix.png` | Matrice de confusion par classe | Voir quelles classes sont confondues |
| `results.png` | Courbes de loss et métriques par epoch | Voir si le modèle converge bien |
| `best.pt` | Poids du meilleur modèle (6.5 MB) | Fichier à déployer en production |

---

## Guide d'utilisation de l'UI MLflow

### 1. Accéder à l'interface

```
http://localhost:5000
```

### 2. Naviguer dans les expériences

```
Page d'accueil MLflow
    │
    ├── Experiments (panneau gauche)
    │   └── yolo-poc              ← notre expérience
    │       ├── Run 1 : yolov8n-ep10-bs16
    │       ├── Run 2 : yolov8s-ep10-bs16
    │       └── Run 3 : yolov8n-ep10-bs16-adamw
    │
    └── Models (onglet du haut)
        └── Model Registry (si des modèles ont été enregistrés)
```

### 3. Comparer des runs

1. Cocher les cases des runs à comparer
2. Cliquer sur **"Compare"** en haut du tableau
3. Voir les graphes côte à côte : mAP50 par epoch, loss...

### 4. Télécharger un modèle

1. Ouvrir un run
2. Onglet **"Artifacts"**
3. `model/best.pt` → bouton **Download**

---

## Les problèmes rencontrés et leurs solutions

> Cette section est importante pour comprendre pourquoi certains choix ont été faits.

### Problème 1 : `ReadTimeoutError` lors du `docker build`

**Ce qui s'est passé :** Le `pip install ultralytics` dans le Dockerfile essayait de télécharger PyTorch avec CUDA (~900 MB). La connexion a coupé après 17 minutes.

**Tentative 1 :** Forcer le torch CPU-only (`--index-url https://download.pytorch.org/whl/cpu`) → timeout encore sur pytorch.org

**Solution finale :** Utiliser `ultralytics/ultralytics:latest` comme image de base — PyTorch est déjà inclus. On n'ajoute que `mlflow` + `boto3` (~50 MB au lieu de 900 MB).

```dockerfile
# Avant (timeout)
FROM python:3.11-slim
RUN pip install torch ultralytics  ← 900MB = timeout

# Après (OK)
FROM ultralytics/ultralytics:latest  ← PyTorch déjà là
RUN pip install mlflow boto3         ← seulement 50MB
```

---

### Problème 2 : `SignatureDoesNotMatch` lors de l'upload S3

**Ce qui s'est passé :** Quand MLflow essayait d'uploader les artifacts (confusion_matrix.png, best.pt) vers OVH S3, l'erreur `SignatureDoesNotMatch` apparaissait.

**Cause :** OVH S3 exige impérativement la signature AWS v4 (`SigV4`) et le mode `path-style` (l'URL doit contenir le nom du bucket en chemin, pas en sous-domaine). boto3 n'utilise pas ces options par défaut.

**Solution :** Ajouter un fichier `~/.aws/config` dans les deux Dockerfiles :

```bash
printf '[default]\ns3 =\n    signature_version = s3v4\n    addressing_style = path\n' \
    > /root/.aws/config
```

---

### Problème 3 : Double logging MLflow

**Ce qui s'est passé :** ultralytics a son propre callback MLflow intégré. Quand `mlflow` est installé et qu'un tracking URI est défini, ultralytics log automatiquement les métriques — en plus de notre code dans `train.py`.

**Résultat :** les métriques étaient loggées deux fois, et le callback natif d'ultralytics crashait sur nos credentials S3.

**Solution :** Désactiver le callback natif en tête de `train.py` :

```python
from ultralytics.utils import SETTINGS
SETTINGS.update({"mlflow": False})  # On gère le logging nous-mêmes
```

---

### Problème 4 : Credentials S3 en lecture seule

**Ce qui s'est passé :** Même après avoir corrigé la signature S3, les uploads échouaient.

**Cause :** Les credentials OVH dans `.env` proviennent du projet `stage_mlops` (DVC). Ces clés ont uniquement les droits de **lecture** (`s3:GetObject`, `s3:ListBucket`), pas d'écriture (`s3:PutObject`). C'est intentionnel — DVC n'a besoin que de lire le bucket.

**Solution pour le POC :** Stocker les artifacts MLflow **localement** dans le volume Docker (`/mlflow/artifacts`) plutôt que sur S3. Toutes les métriques et paramètres sont toujours loggés dans MLflow, seuls les fichiers volumineux sont en local.

**Pour la production :** Créer un compte OVH avec droits `s3:PutObject` sur le préfixe `mlflow-artifacts/`.

---

### Problème 5 : L'ancien URI S3 persistait dans la base

**Ce qui s'est passé :** Après avoir changé le `--default-artifact-root` du serveur MLflow de S3 vers `/mlflow/artifacts`, les runs continuaient d'essayer d'uploader vers l'ancien URI S3.

**Cause :** L'expérience `yolo-poc` avait été créée avec l'URI S3 comme artifact root, et cette valeur est stockée dans la base SQLite. Changer la config du serveur ne modifie pas les expériences existantes.

**Solution :** Purger complètement le volume pour repartir propre :

```bash
docker compose down -v   # ← supprime le volume mlflow_data
docker compose up -d mlflow-server  # recrée tout proprement
```

---

## Résultats obtenus (run de validation)

Dataset : COCO8 (8 images — 4 train, 4 val)
Modèle : YOLOv8n (nano, le plus léger)
3 epochs, batch=4, CPU uniquement

| Classe | Images | mAP50 |
|--------|--------|-------|
| **all** | 4 | **0.889** |
| person | 3 | 0.523 |
| dog | 1 | 0.995 |
| horse | 1 | 0.995 |
| elephant | 1 | 0.828 |
| umbrella | 1 | 0.995 |
| potted plant | 1 | 0.995 |

> Note : scores élevés car dataset très petit (8 images). Sur un vrai dataset (50-100 clips), les scores seront plus représentatifs.

---

## Pour aller plus loin

### Passer en GPU

Dans `Dockerfile.trainer`, remplacer :
```dockerfile
FROM ultralytics/ultralytics:latest  # inclut CUDA automatiquement
```

Dans `docker-compose.yml`, ajouter sous `yolo-trainer` :
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### Passer à PostgreSQL (production)

Ajouter un 3ème service dans `docker-compose.yml` :
```yaml
postgres:
  image: postgres:15
  environment:
    POSTGRES_DB: mlflow
    POSTGRES_USER: mlflow
    POSTGRES_PASSWORD: mlflow
  volumes:
    - pg_data:/var/lib/postgresql/data
```

Et modifier la commande du serveur :
```
--backend-store-uri postgresql://mlflow:mlflow@postgres:5432/mlflow
```

### Activer le Model Registry

Après un entraînement réussi, enregistrer le modèle :
```python
mlflow.register_model(
    f"runs:/{run_id}/model/best.pt",
    "yolov8-detection"
)
```

Puis promouvoir en production depuis l'UI MLflow : Models → yolov8-detection → Stage → Production.

---

## Comparatif MLflow vs Weights & Biases

| Critère | MLflow | Weights & Biases |
|---------|--------|------------------|
| **Hébergement** | Self-hosted (nos serveurs OVH) | SaaS uniquement (serveurs W&B aux USA) |
| **Coût** | Gratuit (open-source) | $50+/user/mois en équipe |
| **Licence** | Apache 2.0 | Propriétaire |
| **RGPD** | Données sur OVH (FR/EU) | Données exportées vers les USA |
| **Model Registry** | Intégré | Intégré |
| **Intégration S3 OVH** | Native via boto3 | Possible mais non standard |
| **Backend store** | SQLite, MySQL, PostgreSQL | Base W&B propriétaire |
| **Compatibilité DVC** | Complémentaire parfait | Overlap, risque de doublon |
| **Conteneurisable** | Oui, nativement | Client Python uniquement |
| **UI** | Sobre, fonctionnelle | Plus riche visuellement |

**Conclusion :** Pour notre stack (OVH, DVC, Docker, RGPD), MLflow est le choix évident. W&B n'apporte rien que MLflow ne peut pas faire, et coûte la conformité RGPD + un abonnement mensuel.
