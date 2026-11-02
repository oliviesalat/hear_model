"""API tests with a fake embedder (no model download). Run: python -m unittest server.test_app"""

import base64
import io
import unittest

import numpy as np
from fastapi.testclient import TestClient
from scipy.io import wavfile

from server.app import app


class FakeEmbedder:
    def __init__(self) -> None:
        self.batches: list[np.ndarray] = []

    def embed(self, clips: np.ndarray) -> np.ndarray:
        self.batches.append(clips)
        # First value marks which clip produced the embedding.
        return np.repeat(clips[:, :1], 512, axis=1)


def wav_base64(audio: np.ndarray, sample_rate: int) -> str:
    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, audio)
    return base64.b64encode(buffer.getvalue()).decode()


class PredictTest(unittest.TestCase):
    def setUp(self) -> None:
        self.embedder = FakeEmbedder()
        app.state.embedder = self.embedder
        # No context manager: lifespan (real model loading) is skipped.
        self.client = TestClient(app, raise_server_exceptions=False)

    def predict(self, body: object) -> list[dict]:
        response = self.client.post("/predict", json=body)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["predictions"]

    def test_health(self) -> None:
        response = self.client.get("/health")
        self.assertEqual((response.status_code, response.text), (200, "ok"))

    def test_demo_page(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("/predict", response.text)

    def test_bad_instances_keep_order_and_good_ones_share_one_batch(self) -> None:
        predictions = self.predict({"instances": [
            {"input_array": [0.25] * 32000},
            {"input_array": [0.0] * 16000},
            {"input_array": [0.0] * 32000, "input_bytes": "x"},
            {"gcs_uri": "gs://bucket/clip.wav"},
            {"input_array": [[0.0] * 32000]},
            {"input_bytes": "not base64!"},
            {"input_array": [0.5] * 32000},
        ]})
        self.assertEqual(predictions[0]["embedding"][0], 0.25)
        self.assertEqual(predictions[6]["embedding"][0], 0.5)
        self.assertEqual(len(predictions[0]["embedding"]), 512)
        for index, reason in [(1, "Got 16000 samples"), (2, "exactly one of"), (3, "gcs_uri"),
                              (4, "1 dimension"), (5, "base64-encoded WAV")]:
            self.assertIn(reason, predictions[index]["error"]["description"])
        self.assertEqual([len(batch) for batch in self.embedder.batches], [2])

    def test_wav_int16_stereo_is_scaled_mixed_and_resampled(self) -> None:
        stereo = np.full((16000, 2), [8192, 24576], dtype=np.int16)  # 2 s at 8 kHz
        predictions = self.predict({"instances": [{"input_bytes": wav_base64(stereo, 8000)}]})
        self.assertAlmostEqual(predictions[0]["embedding"][0], 0.5, places=3)
        self.assertEqual(self.embedder.batches[0].shape, (1, 32000))

    def test_demo_short_stereo_wav_is_resampled_then_zero_padded(self) -> None:
        stereo = np.full((8000, 2), [8192, 24576], dtype=np.int16)  # 1 s at 8 kHz
        buffer = io.BytesIO()
        wavfile.write(buffer, 8000, stereo)
        response = self.client.post("/demo/predict", content=buffer.getvalue(),
                                    headers={"Content-Type": "audio/wav"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual((response.json()["sample_rate"], response.json()["channels"]), (8000, 2))
        clip = self.embedder.batches[0][0]
        self.assertEqual(clip.shape, (32000,))
        self.assertAlmostEqual(clip[100], 16384, delta=1)  # mono mean, int scale kept like the notebook
        self.assertEqual(clip[16000:].tolist(), [0.0] * 16000)

    def test_demo_rejects_non_wav(self) -> None:
        response = self.client.post("/demo/predict", content=b"not a wav", headers={"Content-Type": "audio/wav"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.embedder.batches, [])

    def test_all_instances_bad_skips_model(self) -> None:
        self.predict({"instances": [{"input_array": [0.0]}]})
        self.assertEqual(self.embedder.batches, [])

    def test_malformed_body_is_400(self) -> None:
        for body in ({"x": 1}, {"instances": "nope"}):
            response = self.client.post("/predict", json=body)
            self.assertEqual(response.status_code, 400)
        response = self.client.post("/predict", content=b"not json", headers={"Content-Type": "application/json"})
        self.assertEqual(response.status_code, 400)

    def test_model_failure_is_500_json(self) -> None:
        self.embedder.embed = lambda clips: 1 / 0
        response = self.client.post("/predict", json={"instances": [{"input_array": [0.0] * 32000}]})
        self.assertEqual((response.status_code, response.json()), (500, {"error": "Internal server error."}))


if __name__ == "__main__":
    unittest.main()
