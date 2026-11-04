# AGENTS.md

Local inference server for Google's HeAR (Health Acoustic Representations) audio model: 2 s clip in, 512-d embedding out.

## Layout

- `hear/`: vendored copy of the open-source [Google-Health/hear](https://github.com/Google-Health/hear) repo (upstream TF serving, data processing, notebooks). Treat it as third-party code: change it only when necessary, and record every change in the "Local patches" table in `README.md`.
- `server/`: our own FastAPI server built on top, running the PyTorch model (`google/hear-pytorch`). Same `/predict` API as the upstream TF server, plus a drag-and-drop demo page (`/`) backed by `/demo/predict`, which reproduces the upstream notebook pipeline.
- `docs/server.md`: how to run both servers and how they compare. `docs/datasets.md`: candidate datasets for heads. `docs/issues/`: bug reports for upstream.
- `Makefile`: all run/build/test entry points.
