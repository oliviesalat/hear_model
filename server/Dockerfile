# HeAR embeddings API on PyTorch (server/). Build context: repo root.
#   docker build -f server/Dockerfile -t hear-torch .
# Builds for the host architecture (arm64 on Apple Silicon, no emulation).
# Model is not baked in: mount a local copy at /model or let it download into the HF cache.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    HEAR_MODEL=google/hear-pytorch \
    HEAR_DEVICE=cpu

WORKDIR /app

COPY server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# server/ imports preprocessing from the vendored upstream package.
COPY hear/python/data_processing/__init__.py hear/python/data_processing/audio_utils.py hear/python/data_processing/
COPY server/ server/

# Pre-create HF_HOME: a named volume mounted there inherits this ownership.
RUN useradd --create-home app && mkdir -p /cache/huggingface && chown -R app /cache
USER app

EXPOSE 8080
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8080"]
