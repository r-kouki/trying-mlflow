#!/usr/bin/env bash
set -e

ZENML_URL="http://localhost:8080"

echo "[entrypoint] Connecting to ZenML server at ${ZENML_URL}..."

# --- Activate server on first boot (creates the admin user) ---
IS_ACTIVE=$(curl -sf "${ZENML_URL}/api/v1/info" \
  | python3 -c "import sys,json; print(json.load(sys.stdin).get('active', False))" 2>/dev/null || echo "False")

if [ "$IS_ACTIVE" = "False" ]; then
  echo "[entrypoint] ZenML server not yet activated — activating with admin/admin..."
  curl -sf -X PUT "${ZENML_URL}/api/v1/activate" \
    -H "Content-Type: application/json" \
    -d '{"admin_username":"admin","admin_password":"admin"}' > /dev/null
  echo "[entrypoint] Server activated."
fi

# --- Login and get bearer token ---
TOKEN=$(curl -sf -X POST "${ZENML_URL}/api/v1/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])") || {
    echo "[entrypoint] Login failed — running with local store."
    exec python /app/src/pipeline.py
}

# --- Create service account (idempotent) ---
HTTP_CODE=$(curl -s -o /tmp/sa.json -w "%{http_code}" \
  -X POST "${ZENML_URL}/api/v1/service_accounts" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"name":"pipeline-runner","active":true}')

if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "201" ]; then
  SA_ID=$(python3 -c "import json; print(json.load(open('/tmp/sa.json'))['id'])")
elif [ "$HTTP_CODE" = "409" ]; then
  SA_ID=$(curl -sf "${ZENML_URL}/api/v1/service_accounts?name=pipeline-runner&size=1" \
    -H "Authorization: Bearer ${TOKEN}" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['items'][0]['id'])")
else
  echo "[entrypoint] Could not create service account (HTTP ${HTTP_CODE}) — running with local store."
  exec python /app/src/pipeline.py
fi

# --- Create a fresh API key for this run ---
API_KEY=$(curl -sf -X POST "${ZENML_URL}/api/v1/service_accounts/${SA_ID}/api_keys" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d "{\"name\":\"key-$(date +%s)\"}" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['body']['key'])") || {
    echo "[entrypoint] Could not create API key — running with local store."
    exec python /app/src/pipeline.py
}

echo "[entrypoint] Connecting ZenML client to server..."
export ZENML_STORE_URL="${ZENML_URL}"
export ZENML_STORE_API_KEY="${API_KEY}"

exec python /app/src/pipeline.py
