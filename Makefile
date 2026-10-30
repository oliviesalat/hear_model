# HeAR: локальный inference-сервер. Подробности: docs/server.md
# Пути считаются от папки Makefile, запускать можно откуда угодно: make -C <repo> serve

ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

-include $(ROOT)/.env

IMAGE     ?= hear-serving
PORT      ?= 8080
MODEL_DIR ?= models/hear
PY        ?= .venv/bin/python
# tensorflow/serving публикуется только под amd64, на Apple Silicon идёт через эмуляцию
PLATFORM  ?= linux/amd64

# Относительные пути из .env/командной строки — от корня репо
MODEL_PATH := $(abspath $(if $(filter /%,$(MODEL_DIR)),$(MODEL_DIR),$(ROOT)/$(MODEL_DIR)))
PY_PATH    := $(if $(findstring /,$(PY)),$(if $(filter /%,$(PY)),$(PY),$(ROOT)/$(PY)),$(PY))

export HF_TOKEN

.PHONY: download-model build serve serve-local

download-model:
	$(PY_PATH) -c "from huggingface_hub import snapshot_download; snapshot_download('google/hear', local_dir='$(MODEL_PATH)')"

build:
	docker build --platform $(PLATFORM) -t $(IMAGE) -f $(ROOT)/hear/python/serving/Dockerfile $(ROOT)/hear

# Повторяет hear/python/serving/entrypoint.sh, но берёт модель из локальной папки вместо GCS.
# -w /: пакеты serving/ и data_processing/ лежат в корне образа.
serve: build
	@test -f $(MODEL_PATH)/saved_model.pb || { echo "Нет модели в $(MODEL_PATH), запусти: make download-model"; exit 1; }
	docker run --rm --init --platform $(PLATFORM) -w / \
		-p $(PORT):8080 \
		-e AIP_HTTP_PORT=8080 \
		-e AIP_HEALTH_ROUTE=/health \
		-e AIP_PREDICT_ROUTE=/predict \
		-e MODEL_REST_PORT=8600 \
		-v $(MODEL_PATH):/model/default/1:ro \
		--entrypoint /bin/bash $(IMAGE) -c '\
			/usr/bin/tensorflow_model_server --port=8500 --rest_api_port=8600 \
				--model_name=default --model_base_path=/model/default \
				--xla_cpu_compilation_enabled=true & \
			/server-env/bin/python3.12 -m serving.server_gunicorn --alsologtostderr --verbosity=1 & \
			wait -n'

# Без Docker и TF Serving: модель грузится in-process (serving/server_local.py).
# Работает нативно на macOS arm64. Нужен $(PY) с root requirements.txt.
serve-local:
	@test -f $(MODEL_PATH)/saved_model.pb || { echo "Нет модели в $(MODEL_PATH), запусти: make download-model"; exit 1; }
	cd $(ROOT)/hear/python && \
		AIP_HTTP_PORT=$(PORT) AIP_HEALTH_ROUTE=/health AIP_PREDICT_ROUTE=/predict MODEL_PATH=$(MODEL_PATH) \
		$(PY_PATH) -m serving.server_local --alsologtostderr --verbosity=1
