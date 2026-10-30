"""HeAR PyTorch model: batch of audio clips -> embeddings."""

import threading

import numpy as np
import torch
from transformers import AutoModel

from hear.python.data_processing.audio_utils import preprocess_audio


def pick_device(name: str) -> torch.device:
    """Resolves a device name; "auto" picks cuda, then mps, then cpu."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Embedder:
    """Runs google/hear-pytorch on raw clips (preprocessing included)."""

    def __init__(self, model_path: str, device: torch.device, max_batch: int = 32):
        """Loads the model.

        Args:
            model_path: Hugging Face model id or local directory.
            device: Device to run the model on.
            max_batch: Clips per forward pass; bounds peak memory on large requests.
        """
        self.device = device
        self._model = AutoModel.from_pretrained(model_path).eval().to(device)
        self._max_batch = max_batch
        # Requests run in a threadpool; parallel forwards on one model only compete
        # for the same device, so serialize them.
        self._lock = threading.Lock()

    def embed(self, clips: np.ndarray) -> np.ndarray:
        """Computes embeddings.

        Args:
            clips: float32 array of shape (n, 32000), n >= 1.

        Returns:
            float32 array of shape (n, 512).
        """
        chunks = []
        with self._lock, torch.inference_mode():
            for start in range(0, len(clips), self._max_batch):
                audio = torch.from_numpy(clips[start : start + self._max_batch]).to(self.device)
                chunks.append(self._model(preprocess_audio(audio)).pooler_output.cpu().numpy())
        return np.concatenate(chunks)
