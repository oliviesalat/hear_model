"""Local dev server: same HTTP API as server_gunicorn, model runs in-process.

Uses InlineModelRunner instead of TF Serving, so it runs natively on any
platform with a tensorflow wheel (incl. macOS arm64), without Docker.

Required env: AIP_HTTP_PORT, AIP_HEALTH_ROUTE, AIP_PREDICT_ROUTE, MODEL_PATH
(directory with saved_model.pb).
"""

from collections.abc import Sequence
import os

from absl import app
from absl import logging

from serving.serving_framework import inline_prediction_executor
from serving.serving_framework import server_gunicorn
from serving.serving_framework.tensorflow import inline_model_runner
from serving import predictor


def _load_model_runner() -> inline_model_runner.InlineModelRunner:
  # Imported here: the runner is built post-fork in the gunicorn worker.
  import tensorflow as tf  # pylint: disable=g-import-not-at-top

  return inline_model_runner.InlineModelRunner(
      tf.saved_model.load(os.environ['MODEL_PATH'])
  )


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  options = {
      'bind': f'0.0.0.0:{int(os.environ["AIP_HTTP_PORT"])}',
      # Each worker holds its own model copy in memory; one is enough for dev.
      'workers': 1,
      'timeout': 120,
  }
  logging.info('Launching local gunicorn application.')
  server_gunicorn.PredictionApplication(
      inline_prediction_executor.InlinePredictionExecutor(
          predictor.Predictor().predict, _load_model_runner
      ),
      # No separate model server: the model loads before the worker serves.
      health_check=None,
      options=options,
  ).run()


if __name__ == '__main__':
  app.run(main)
