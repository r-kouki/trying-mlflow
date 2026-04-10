PYTHON ?= python3
VENV_DIR ?= .venv
ENV_FILE ?= .env
COMPOSE_FILE ?= docker/docker-compose.yml

.PHONY: init venv install up train demo down logs status

venv:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		$(PYTHON) -m venv $(VENV_DIR); \
		echo "Created $(VENV_DIR)"; \
	else \
		echo "$(VENV_DIR) already exists"; \
	fi

install: venv
	@$(VENV_DIR)/bin/pip install --upgrade pip
	@$(VENV_DIR)/bin/pip install -r requirements.txt

init: venv
	@if [ ! -f "$(ENV_FILE)" ]; then \
		cp .env.example $(ENV_FILE); \
		echo "Created $(ENV_FILE) from .env.example"; \
	else \
		echo "$(ENV_FILE) already exists"; \
	fi
	@echo "Local bootstrap complete."
	@echo "Next steps:"
	@echo "  1. Review $(ENV_FILE)"
	@echo "  2. Run 'make install' for local Python deps"
	@echo "  3. Run 'make up' to start MLflow"

up:
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) up -d mlflow-server

train:
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) run --rm yolo-trainer

demo:
	./run_demo.sh

down:
	docker compose -f $(COMPOSE_FILE) down

logs:
	docker compose -f $(COMPOSE_FILE) logs -f mlflow-server

status:
	docker compose -f $(COMPOSE_FILE) ps
