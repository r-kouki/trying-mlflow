"""
Téléchargement du dataset depuis le bucket OVH S3.
Récupère les images et annotations depuis le préfixe data/ du bucket
et les stocke localement pour l'entraînement YOLO.
"""

import os
from pathlib import Path

import boto3

from config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    BUCKET_NAME,
    DATA_PREFIX,
    LOCAL_DATA_DIR,
    OVH_ENDPOINT,
)


def get_s3_client():
    """Crée un client boto3 configuré pour OVH S3."""
    return boto3.client(
        "s3",
        endpoint_url=OVH_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


def download_dataset(force: bool = False) -> Path:
    """
    Télécharge le dataset depuis S3 vers le répertoire local.

    Args:
        force: Si True, re-télécharge même si les données existent déjà.

    Returns:
        Chemin local du dataset téléchargé.
    """
    # Vérifier si les données existent déjà
    if LOCAL_DATA_DIR.exists() and any(LOCAL_DATA_DIR.iterdir()) and not force:
        print(f"[data_loader] Données déjà présentes dans {LOCAL_DATA_DIR}, skip.")
        return LOCAL_DATA_DIR

    print(f"[data_loader] Téléchargement depuis s3://{BUCKET_NAME}/{DATA_PREFIX} ...")
    s3 = get_s3_client()

    # Lister tous les objets sous le préfixe data/
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=DATA_PREFIX)

    count = 0
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Ignorer les "dossiers" (clés se terminant par /)
            if key.endswith("/"):
                continue

            # Chemin local : on enlève le préfixe data/ pour garder la structure
            relative = key[len(DATA_PREFIX):]
            local_path = LOCAL_DATA_DIR / relative
            local_path.parent.mkdir(parents=True, exist_ok=True)

            s3.download_file(BUCKET_NAME, key, str(local_path))
            count += 1

    print(f"[data_loader] {count} fichiers téléchargés dans {LOCAL_DATA_DIR}")
    return LOCAL_DATA_DIR


def prepare_dataset_yaml(data_dir: Path, dataset_config: dict) -> Path:
    """
    Génère un dataset.yaml temporaire avec les chemins absolus locaux.
    Nécessaire pour que YOLO trouve les images/labels.

    Args:
        data_dir: Répertoire racine des données locales.
        dataset_config: Config chargée depuis configs/dataset.yaml.

    Returns:
        Chemin du dataset.yaml généré.
    """
    import tempfile
    import yaml

    # Construire le YAML avec chemins absolus
    runtime_config = {
        "path": str(data_dir),
        "train": dataset_config.get("train", "images/train"),
        "val": dataset_config.get("val", "images/val"),
        "names": dataset_config.get("names", {0: "object"}),
    }

    # Write to temp dir to avoid permission issues when data_local/ is owned by root (Docker)
    output_path = Path(tempfile.gettempdir()) / "yolo_dataset.yaml"
    with open(output_path, "w") as f:
        yaml.dump(runtime_config, f, default_flow_style=False)

    print(f"[data_loader] dataset.yaml généré : {output_path}")
    return output_path


if __name__ == "__main__":
    # Test standalone
    download_dataset()
