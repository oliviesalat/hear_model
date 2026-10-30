# HeAR: local inference server. Details: docs/server.md
# Paths are relative to the Makefile folder, so it runs from anywhere: make -C <repo> serve

ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

-include $(ROOT)/.env

IMAGE     ?= hear-serving
TORCH_IMAGE ?= hear-torch
PORT      ?= 8080
MODEL_DIR ?= models/hear
# PyTorch model for serve-torch: HF id (downloaded into the HF cache) or a local folder
TORCH_MODEL ?= google/hear-pytorch
# auto = cuda, then mps, then cpu
DEVICE    ?= auto
PY        ?= .venv/bin/python
# tensorflow/serving is published for amd64 only and needs AVX (Rosetta emulation lacks it)
PLATFORM  ?= linux/amd64

# Relative paths from .env or the command line are resolved from the repo root
MODEL_PATH := $(abspath $(if $(filter /%,$(MODEL_DIR)),$(MODEL_DIR),$(ROOT)/$(MODEL_DIR)))
PY_PATH    := $(if $(findstring /,$(PY)),$(if $(filter /%,$(PY)),$(PY),$(ROOT)/$(PY)),$(PY))

export HF_TOKEN

.PHONY: download-model build serve serve-local serve-torch test-torch build-torch serve-torch-docker

download-model:
	$(PY_PATH) -c "from huggingface_hub import snapshot_download; snapshot_download('google/hear', local_dir='$(MODEL_PATH)')"

build:
	docker build --platform $(PLATFORM) -t $(IMAGE) -f $(ROOT)/hear/python/serving/Dockerfile $(ROOT)/hear

# Mirrors hear/python/serving/entrypoint.sh, but takes the model from a local folder instead of GCS.
# -w /: the serving/ and data_processing/ packages live in the image root.
serve: build
	@test -f $(MODEL_PATH)/saved_model.pb || { echo "No model in $(MODEL_PATH), run: make download-model"; exit 1; }
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

# No Docker, no TF Serving: the model runs in-process (serving/server_local.py).
# Runs natively on macOS arm64. Needs $(PY) with the root requirements.txt.
serve-local:
	@test -f $(MODEL_PATH)/saved_model.pb || { echo "No model in $(MODEL_PATH), run: make download-model"; exit 1; }
	cd $(ROOT)/hear/python && \
		AIP_HTTP_PORT=$(PORT) AIP_HEALTH_ROUTE=/health AIP_PREDICT_ROUTE=/predict MODEL_PATH=$(MODEL_PATH) \
		$(PY_PATH) -m serving.server_local --alsologtostderr --verbosity=1

# FastAPI + PyTorch (server/), same /health and /predict contract. Swagger: http://localhost:$(PORT)/docs
serve-torch:
	cd $(ROOT) && HEAR_MODEL=$(TORCH_MODEL) HEAR_DEVICE=$(DEVICE) \
		$(PY_PATH) -m uvicorn server.app:app --host 0.0.0.0 --port $(PORT)

test-torch:
	cd $(ROOT) && $(PY_PATH) -m unittest server.test_app

# PyTorch server in Docker: native host architecture (arm64 on a Mac), CPU only (no MPS in Docker).
build-torch:
	docker build -t $(TORCH_IMAGE) -f $(ROOT)/server/Dockerfile $(ROOT)

# A local TORCH_MODEL folder is mounted at /model; an HF id is downloaded into the hear-hf-cache volume.
TORCH_MODEL_DIR := $(firstword $(wildcard $(abspath $(if $(filter /%,$(TORCH_MODEL)),$(TORCH_MODEL),$(ROOT)/$(TORCH_MODEL)))/config.json))
serve-torch-docker: build-torch
	docker run --rm --init -p $(PORT):8080 -e HF_TOKEN \
		$(if $(TORCH_MODEL_DIR),-v $(dir $(TORCH_MODEL_DIR)):/model:ro -e HEAR_MODEL=/model,-v hear-hf-cache:/cache/huggingface -e HEAR_MODEL=$(TORCH_MODEL)) \
		$(TORCH_IMAGE)
