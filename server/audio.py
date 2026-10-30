"""Decodes request instances into HeAR input clips: 2 s of 16 kHz mono audio."""

import base64
import binascii
import io
from typing import Any

import numpy as np
from scipy import signal
from scipy.io import wavfile

SAMPLE_RATE = 16000
CLIP_SAMPLES = 2 * SAMPLE_RATE


class InstanceError(ValueError):
    """A client-caused problem with one instance; the message is shown to the client."""


def instance_to_clip(instance: dict[str, Any]) -> np.ndarray:
    """Turns one request instance into an audio clip.

    Args:
        instance: Request instance with exactly one of `input_array`
            (32000 floats in [-1, 1]) or `input_bytes` (base64-encoded WAV).

    Returns:
        A float32 array of shape (CLIP_SAMPLES,).

    Raises:
        InstanceError: The instance is malformed or the audio is not 2 s long.
    """
    if "gcs_uri" in instance:
        raise InstanceError("gcs_uri is not supported, send input_bytes or input_array.")
    if set(instance) not in ({"input_array"}, {"input_bytes"}):
        raise InstanceError(
            "Request has incorrectly formatted audio input keys. "
            "Must specify exactly one of `input_bytes` or `input_array`."
        )
    if "input_bytes" in instance:
        return _decode_wav(instance["input_bytes"])
    return _parse_array(instance["input_array"])


def _parse_array(values: Any) -> np.ndarray:
    try:
        clip = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise InstanceError("input_array must be a list of numbers.") from error
    if clip.ndim != 1:
        raise InstanceError(f"Audio array must have 1 dimension. Got {clip.ndim} dimensions.")
    return _check_length(clip)


def _decode_wav(wav_base64: Any) -> np.ndarray:
    try:
        sample_rate, audio = wavfile.read(io.BytesIO(base64.b64decode(wav_base64, validate=True)))
    except (binascii.Error, TypeError, ValueError) as error:
        raise InstanceError("input_bytes must be a base64-encoded WAV file.") from error

    if audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128) / 128
    elif np.issubdtype(audio.dtype, np.signedinteger):
        audio = audio.astype(np.float32) / -np.iinfo(audio.dtype).min
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        audio = signal.resample(audio, int(len(audio) * SAMPLE_RATE / sample_rate))
    return _check_length(audio.astype(np.float32))


def _check_length(clip: np.ndarray) -> np.ndarray:
    if clip.shape[0] != CLIP_SAMPLES:
        raise InstanceError(
            f"Audio array must have {CLIP_SAMPLES} samples (2s sampled at 16kHz). "
            f"Got {clip.shape[0]} samples."
        )
    return clip
