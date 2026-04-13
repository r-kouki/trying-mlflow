#!/usr/bin/env bash
set -e

# Register ZenML stack components — all commands are idempotent (|| true)
zenml init || true

zenml artifact-store register ovh_s3_store \
    --flavor=s3 \
    --path="s3://${BUCKET_NAME}/zenml" \
    --client_kwargs="{\"endpoint_url\":\"${OVH_ENDPOINT}\",\"region_name\":\"gra\"}" \
    --key="${OVH_ACCESS_KEY}" \
    --secret="${OVH_SECRET_KEY}" 2>/dev/null || true

zenml experiment-tracker register mlflow_tracker \
    --flavor=mlflow \
    --tracking_uri="${MLFLOW_TRACKING_URI}" 2>/dev/null || true

zenml stack register yolo-ovh-stack \
    -o default \
    -a ovh_s3_store \
    -e mlflow_tracker 2>/dev/null || true

zenml stack set yolo-ovh-stack

exec python /app/src/pipeline.py
