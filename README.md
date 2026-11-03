# hear_model

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Vendored code

`hear/` is a vendored copy of https://github.com/Google-Health/hear
at commit `36bbb90879545a2f1106668048b1a12a61ff2a0d` (2025-07-03), with the local patches below.

### Local patches

| File | Change | Commit |
|------|--------|--------|
| `hear/python/serving/Dockerfile` | build on `python:3.12-slim`, copy only `tensorflow_model_server` from `tensorflow/serving` (deadsnakes dropped Ubuntu 20.04, so the original image no longer builds) | `55cdec3` |
| `hear/python/data_processing/audio_utils.py` | normalize audio per clip instead of per batch, out of place ([bug report](docs/issues/upstream-torch-preprocess-batch-normalization.md)) | `f989212` |
| `hear/python/serving/serving_framework/tensorflow/inline_model_runner.py` (+ test) | pass named inputs to the TF signature as kwargs; every in-process `/predict` failed without it | `4731aba` |
| `hear/python/serving/server_local.py` | new: TF server with the model in-process, for `make serve-local` | `c17f4aa` |

To update upstream: replace the directory with a newer checkout, bump the commit above,
then re-apply each patch (`git show <commit> -- hear/`) unless upstream has fixed it.
Keep this table in sync when changing anything under `hear/`.

## Roadmap

### Disease classification head

HeAR only produces 512-d embeddings; there are no released heads for disease prediction
(the closest upstream example is the few-shot cough classifier in
`hear/notebooks/train_data_efficient_classifier.ipynb`). Plan: train our own head on top of
the embeddings.

- Start with three sklearn heads compared on the same split: `LogisticRegression`
  (linear probe), `SVC(kernel="rbf", probability=True)` and `MLPClassifier`
  (one hidden layer); move to anything heavier only if all plateau.
  Put a `StandardScaler` in front (required for SVM). SVM has no native probabilities
  (`probability=True` adds internal CV calibration, slower to train) and kernel SVM
  scales poorly past ~10k clips.
- Labels are per patient, so split train/test by patient to avoid leakage.
- A recording yields one embedding per 2 s clip; aggregate clip probabilities (mean or max)
  into a per-recording prediction.
- Candidate public datasets: Coswara, COUGHVID (COVID-19), CODA (TB, on request).

For sound *events* (cough, snore, sneeze, …) use the pretrained event detector from the
`google/hear` HF repo instead (see `hear/notebooks/hear_event_detector_demo.ipynb`).
