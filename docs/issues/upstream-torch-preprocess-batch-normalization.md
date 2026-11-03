---
name: Bug Report
about: Create a bug report to help us improve
title: 'Bug: PyTorch preprocess_audio normalizes over the whole batch, so clips in a batch change each other''s embeddings'
labels: 'bug', 'needs triage'
assignees: ''
---

## Describe the overall issue and situation

`preprocess_audio` in `python/data_processing/audio_utils.py` (the preprocessing
for [google/hear-pytorch](https://huggingface.co/google/hear-pytorch)) scales the
audio to `[-1, 1]` using the min/max of the **whole batch** instead of each clip:

```python
# python/data_processing/audio_utils.py, _mel_pcen
x -= torch.min(x)
x = x / (torch.max(x) + 1e-8)
x = (x * 2) - 1
```

`torch.min(x)` / `torch.max(x)` without `dim` reduce over all elements of the
`[batch, samples]` tensor. So a clip's embedding depends on the other clips in
the same batch: a quiet clip batched with a loud one gets a different embedding
than when embedded alone. The TensorFlow SavedModel
([google/hear](https://huggingface.co/google/hear)) does not have this behavior:
its embeddings are batch-independent.

I found it while building a batched embedding server on `google/hear-pytorch`
and validating it against the TF SavedModel.

A secondary issue in the same lines: `x = x.float()` returns the same tensor for
float32 input, so `x -= torch.min(x)` modifies the caller's input tensor in place.

## Expected behavior

- The embedding of a clip does not depend on the other clips in the batch.
- PyTorch embeddings match the TF SavedModel for batched input as well (they
  already match with cosine similarity 1.000000 when clips are processed one at a time).
- `preprocess_audio` does not modify its input.

## Actual behavior

Same quiet clip, embedded alone vs. in a batch with one loud clip:

| | alone vs. in batch (cosine) | in batch vs. TF (cosine) |
|---|---|---|
| TF SavedModel `google/hear` | 1.000000 | — |
| PyTorch `google/hear-pytorch` + `preprocess_audio` | **0.918** | **0.918** |
| PyTorch with per-clip normalization (fix below) | 1.000000 | 1.000000 |

With other batch compositions I saw cosine similarity down to 0.58 (a 440 Hz tone
at amplitude 0.5 batched with a full-range clip).

Also, after `preprocess_audio(audio)` the `audio` tensor passed in by the caller has been changed.

The example in the model card does not show the problem because all
`torch.rand((4, 32000))` clips have the same range.

## Steps to reproduce the issue

1. `pip install torch transformers==4.50.3 tensorflow==2.18.0 huggingface_hub`, clone this repo,
   accept the terms for `google/hear` and `google/hear-pytorch` on Hugging Face.
2. From the directory containing the `hear` checkout, run:

```python
import importlib

import numpy as np
import tensorflow as tf
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel

audio_utils = importlib.import_module("hear.python.data_processing.audio_utils")

rng = np.random.default_rng(0)
quiet = (0.05 * rng.standard_normal(32000)).astype(np.float32)
loud = rng.uniform(-1, 1, 32000).astype(np.float32)
cos = lambda a, b: float((a * b).sum() / np.linalg.norm(a) / np.linalg.norm(b))

model = AutoModel.from_pretrained("google/hear-pytorch").eval()
with torch.inference_mode():
    alone = (
        model(audio_utils.preprocess_audio(torch.from_numpy(quiet[None])))
        .pooler_output[0]
        .numpy()
    )
    batched = (
        model(audio_utils.preprocess_audio(torch.from_numpy(np.stack([quiet, loud]))))
        .pooler_output[0]
        .numpy()
    )
print("torch: alone vs in batch", cos(alone, batched))  # 0.918

tf_model = tf.saved_model.load(snapshot_download("google/hear"))
serve = tf_model.signatures["serving_default"]
tf_alone = serve(x=tf.constant(quiet[None]))["output_0"].numpy()[0]
tf_batched = serve(x=tf.constant(np.stack([quiet, loud])))["output_0"].numpy()[0]
print("tf: alone vs in batch", cos(tf_alone, tf_batched))  # 1.0
print("torch in batch vs tf", cos(batched, tf_alone))  # 0.918

audio = torch.rand((2, 32000)) * 0.1
before = audio.clone()
audio_utils.preprocess_audio(audio)
print("input modified:", not torch.equal(audio, before))  # True
```

3. Compare the printed values with the comments.

## Any additional content

Suggested fix (per-clip reduction, out of place), which brings batched PyTorch
embeddings to cosine similarity 1.000000 with TF on CPU and MPS:

```python
x = x - torch.amin(x, dim=-1, keepdim=True)
x = x / (torch.amax(x, dim=-1, keepdim=True) + 1e-8)
x = (x * 2) - 1
```

Environment: macOS 15.7 (Apple M1), Python 3.12.6, torch 2.12.1, transformers 4.50.3,
tensorflow 2.18.0; repo at commit 36bbb90.
