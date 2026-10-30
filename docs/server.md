# HeAR inference server

The server takes a 2-second audio clip and returns its embedding (a vector from the HeAR model).

There are two implementations with the same HTTP API:

| | TF server (upstream) | PyTorch server |
|---|---|---|
| Code | `hear/python/serving/` | `server/` |
| Model | [google/hear](https://huggingface.co/google/hear) (TF SavedModel) | [google/hear-pytorch](https://huggingface.co/google/hear-pytorch) |
| Runs | Docker on x86_64 with AVX; `make serve-local` natively | natively (macOS MPS, Linux CUDA/CPU) or Docker on any arch |
| Use for | parity with the Vertex AI deployment | local development |

Both return the same embeddings (see [PyTorch server](#pytorch-server-server)).

## TF server architecture

Two processes run inside one Docker container:

```
client ──HTTP──▶ gunicorn + Flask (:8080) ──gRPC──▶ TensorFlow Serving (:8500) ──▶ HeAR SavedModel
                 serving/server_gunicorn.py          REST :8600 — health check only
```

| Port | What listens | Exposed |
|------|--------------|---------|
| `8080` (`AIP_HTTP_PORT`) | HTTP API (gunicorn, 3 workers, 120 s timeout) | yes, this is the API |
| `8500` | TF Serving gRPC; the frontend calls the model through it (60 s timeout) | no |
| `8600` (`MODEL_REST_PORT`) | TF Serving REST; the frontend checks model health through it | no |

Change the host port with `PORT`: `make serve PORT=9000`.

## Running the TF server

### Requirements

- Docker on an x86_64 machine with AVX (`grep -o -w 'avx\|avx2' /proc/cpuinfo | sort -u`).
  The `tensorflow/serving` image exists only for `amd64`, and `tensorflow_model_server` and
  `tensorflow` 2.18 are built with AVX.
- Apple Silicon: Rosetta emulation does not expose AVX to the container, so `make build` succeeds
  but the server crashes with `Illegal instruction`. Turning off
  *Use Rosetta for x86_64/amd64 emulation* in Docker Desktop switches to QEMU, which has AVX
  (untested, and very slow). On a Mac, use `make serve-local` or the [PyTorch server](#pytorch-server-server) instead.
- Access to the model on Hugging Face: [google/hear](https://huggingface.co/google/hear) is gated —
  accept the terms on the model page and create a token.
- A Python environment with dependencies: `pip install -r requirements.txt` (needs `huggingface_hub`).

### Steps

Run commands from the repo root (the folder with the `Makefile`). From elsewhere: `make -C <path to repo> serve`.
All Makefile paths are relative to its folder, so the current directory does not matter.

```bash
# 0. Put HF_TOKEN into .env (the rest can stay as is)

# 1. Download the model into models/hear (once)
make download-model

# 2. Build the image and start the server
make serve
```

Layout after the download:

```
hear_model/
├── .env               # local settings (in .gitignore)
├── Makefile
├── docs/server.md
├── models/hear/       # MODEL_DIR (in .gitignore)
│   ├── saved_model.pb
│   └── variables/
└── hear/              # build context for docker build
    └── python/serving/Dockerfile
```

`make serve` builds the image (`make build`) and starts the container. The first build is slow (apt, pip, gcloud).
Stop it with `Ctrl+C`.

The server is ready when health returns `ok`:

```bash
curl http://localhost:8080/health
```

### Without Docker: `make serve-local`

Same server, but the model runs in-process (`hear/python/serving/server_local.py`) instead of in TF Serving.
Works natively on macOS arm64. Needs `.venv` with the root `requirements.txt` and the downloaded model.
Runs 1 gunicorn worker (each worker holds its own copy of the model, ~1.6 GB).

### Settings

Settings come from `.env` in the repo root. A command-line argument (`make serve PORT=9000`) overrides `.env`.
`.env` is read by make, not by the shell: no quotes, no spaces around `=`.
Relative paths (`MODEL_DIR`, `TORCH_MODEL`, `PY`) are resolved from the repo root.

| Variable | Default | Purpose |
|----------|---------|---------|
| `HF_TOKEN` | empty | Hugging Face token for model downloads. If empty, the token from `hf auth login` is used. A value in `.env` overrides `HF_TOKEN` from the shell |
| `PORT` | `8080` | API port on the host |
| `MODEL_DIR` | `models/hear` | folder with the SavedModel (`saved_model.pb`, `variables/`) |
| `IMAGE` | `hear-serving` | TF server Docker image name |
| `PLATFORM` | `linux/amd64` | TF server image platform |
| `PY` | `.venv/bin/python` | Python for `download-model`, `serve-local`, `serve-torch` |
| `TORCH_MODEL` | `google/hear-pytorch` | PyTorch model: HF id or local folder |
| `DEVICE` | `auto` | PyTorch device: `auto` (cuda → mps → cpu), `cpu`, `mps`, `cuda` |
| `TORCH_IMAGE` | `hear-torch` | PyTorch server Docker image name |

### Difference from production

In production (Vertex AI) the entrypoint is `hear/python/serving/entrypoint.sh`: it copies the model from GCS
(`AIP_STORAGE_URI`) with `gcloud`, and Vertex sets `AIP_HTTP_PORT`, `AIP_HEALTH_ROUTE`, `AIP_PREDICT_ROUTE` itself.
`make serve` does the same, but mounts the model from a local folder and sets the variables itself.

## PyTorch server (`server/`)

FastAPI server on [google/hear-pytorch](https://huggingface.co/google/hear-pytorch). Runs natively
on macOS (MPS) and Linux (CUDA/CPU), without TF Serving or AVX. Same `/health` and `/predict` contract as [below](#endpoints).

```bash
pip install -r requirements.txt
make serve-torch                 # downloads the model into the HF cache on first run (needs HF_TOKEN)
make serve-torch DEVICE=cpu      # auto (default) = cuda → mps → cpu
make test-torch                  # API unit tests, no model needed
```

Swagger UI: `http://localhost:8080/docs`.

In Docker (`server/Dockerfile`: CPU build of torch, native host architecture, arm64 on a Mac with no emulation):

```bash
make serve-torch-docker                                  # model is downloaded into the hear-hf-cache volume
make serve-torch-docker TORCH_MODEL=models/hear-pytorch  # local folder is mounted at /model
```

There is no GPU in the container: Docker on a Mac does not pass MPS through, so Docker is slower than native `make serve-torch`.

Differences from the TF server:
- `gcs_uri` is not supported: an instance with it gets an error.
- A malformed request body (not JSON, no `instances` list) returns `400`, not `500`.
- All valid instances run as one batch (32 per forward pass) instead of one by one.

Embeddings match the TF SavedModel: cosine similarity 1.0 on noise and speech-like signals, on CPU and MPS.
On degenerate synthetic clips (a pure sine, complete silence) it is about 0.97.
`server/` imports preprocessing from `hear/python/data_processing/audio_utils.py`
(patched locally, see [the upstream bug report](issues/upstream-torch-preprocess-batch-normalization.md)).

## Endpoints

Paths are set by environment variables; locally they are:

| Method | Path | Variable |
|--------|------|----------|
| `GET` | `/health` | `AIP_HEALTH_ROUTE` |
| `POST` | `/predict` | `AIP_PREDICT_ROUTE` |

### `GET /health`

TF server: checks that TF Serving has loaded the model (`GET localhost:8600/v1/models/default`).
PyTorch server and `serve-local`: the model loads before the server starts listening, so it always returns `ok`.

| Code | Body | Meaning |
|------|------|---------|
| `200` | `ok` | ready for requests |
| `503` | `not ok` | TF server only: the model is still loading or TF Serving is down |

### `POST /predict`

**This is the embedding endpoint.** HeAR is an encoder model: its "prediction" is the embedding; there is no other endpoint.
The name `predict` comes from the Vertex AI contract: Vertex sends requests to the route in `AIP_PREDICT_ROUTE`
in the form `{"instances": [...]}` → `{"predictions": [...]}`. Locally you can change the path, e.g.
`-e AIP_PREDICT_ROUTE=/embed` in the Makefile; on Vertex it is set by Vertex anyway.

`Content-Type: application/json`. The body is an object with an `instances` array, one audio clip per item:

```json
{
  "instances": [
    {"input_array": [0.0, 0.01, ...]},
    {"input_bytes": "UklGRiQ..."},
    {"gcs_uri": "gs://bucket/path/clip.wav", "bearer_token": "ya29..."}
  ]
}
```

Each instance has exactly **one** of these fields:

| Field | Type | Meaning |
|-------|------|---------|
| `input_array` | array of **32000** floats | 2 seconds of mono audio, 16 kHz, values in `[-1, 1]` |
| `input_bytes` | string (base64) | bytes of a WAV file |
| `gcs_uri` | string `gs://.../*.wav` | WAV in Google Cloud Storage (TF server only) |
| `bearer_token` | string | optional, only with `gcs_uri`: OAuth token for bucket access |

Any other key (including `extensions` from `vertex_schemata/instance.yaml`) → instance error.

WAV handling (`input_bytes`, `gcs_uri`):
- integer samples are scaled to float `[-1, 1]` (`int16`/`int32`; the PyTorch server also handles `uint8`);
- stereo is mixed down to mono (mean over channels);
- sample rate ≠ 16 kHz → resampled to 16 kHz;
- after that the length must be exactly 32000 samples (2 s), otherwise an error. Split long recordings into 2 s clips on the client.

#### Response

`200`, one `predictions` item per instance, in the same order:

```json
{
  "predictions": [
    {"embedding": [0.12, -0.03, ...]},
    {"error": {"description": "Failed to get prediction for instance. Reason: Audio array must have 32000 samples. Got 16000 samples."}}
  ]
}
```

- `embedding` is a float vector (512 values for HeAR).
- An error in one instance does not fail the whole request: its item has `error.description`, and the status is still `200`.

Whole-request errors:

| Code | Body | When |
|------|------|------|
| `400` | `{"error": "No JSON body."}` | TF server: body is not JSON |
| `400` | ``{"error": "Body must be JSON with an `instances` list."}`` | PyTorch server: body is not JSON or has no `instances` list |
| `500` | `{"error": "Internal server error."}` | TF server: no `instances` key; both: any other unhandled exception |

### Examples

curl, silence (32000 zeros):

```bash
python3 -c 'import json; print(json.dumps({"instances": [{"input_array": [0.0]*32000}]}))' \
  | curl -s -X POST http://localhost:8080/predict -H 'Content-Type: application/json' -d @- \
  | head -c 300
```

Python, WAV file:

```python
import base64
import requests

with open("clip.wav", "rb") as f:
    payload = {"instances": [{"input_bytes": base64.b64encode(f.read()).decode()}]}

resp = requests.post("http://localhost:8080/predict", json=payload, timeout=120)
resp.raise_for_status()
prediction = resp.json()["predictions"][0]
print(prediction.get("error") or len(prediction["embedding"]))
```

## Troubleshooting

- **`/health` returns `503` for a long time** — TF Serving is loading the model; under amd64 emulation this can take a minute or two. Check the container logs.
- **`Illegal instruction`** in the TF server container — the CPU (or Rosetta emulation) has no AVX; see [Requirements](#requirements).
- **`No model in .../models/hear`** — `make download-model` was not run, or `MODEL_DIR` points elsewhere. The folder must contain `saved_model.pb`.
- **`401`/`403` on download** — the model license is not accepted on Hugging Face, or `HF_TOKEN` in `.env` is empty or wrong.
- **`ModuleNotFoundError: huggingface_hub`** (or `torch`, `fastapi`) — run `pip install -r requirements.txt` in `.venv`.
- **Port in use** — `make serve PORT=9000`.
