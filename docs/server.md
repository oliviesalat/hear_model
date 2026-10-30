# HeAR inference-сервер

Сервер принимает 2-секундный аудиоклип и возвращает его embedding (вектор из модели HeAR).
Код: `hear/python/serving/`.

## Как устроено

Внутри одного Docker-контейнера работают два процесса:

```
клиент ──HTTP──▶ gunicorn + Flask (:8080) ──gRPC──▶ TensorFlow Serving (:8500) ──▶ SavedModel HeAR
                 serving/server_gunicorn.py          REST :8600 — только для health check
```

| Порт | Что слушает | Наружу |
|------|-------------|--------|
| `8080` (`AIP_HTTP_PORT`) | HTTP API (gunicorn, 3 workers, timeout 120s) | да, это API |
| `8500` | TF Serving gRPC, через него frontend вызывает модель (timeout 60s) | нет |
| `8600` (`MODEL_REST_PORT`) | TF Serving REST, frontend проверяет по нему здоровье модели | нет |

Порт снаружи меняется через `PORT` в Makefile: `make serve PORT=9000`.

## Локальный запуск

TF Serving под macOS нативно не ставится, поэтому сервер запускается только в Docker.

### Что нужно

- Docker (Docker Desktop / OrbStack). Образ `tensorflow/serving` есть только под `amd64`,
  на Apple Silicon он работает через эмуляцию: медленнее, но работает.
  В Docker Desktop включи *Use Rosetta for x86_64/amd64 emulation*.
- Доступ к модели на Hugging Face: модель [google/hear](https://huggingface.co/google/hear) gated —
  прими условия на странице модели и создай token.
- Python-окружение с зависимостями: `pip install -r requirements.txt` (нужен `huggingface_hub`).

### Шаги

Команды запускаются из корня репо (папка с `Makefile`). Из другой папки: `make -C <путь к репо> serve`.
Все пути в Makefile считаются от его папки, поэтому текущая директория на них не влияет.

```bash
# 0. Вписать HF_TOKEN в .env (остальное можно не трогать)

# 1. Скачать модель в models/hear (один раз)
make download-model

# 2. Собрать образ и запустить сервер
make serve
```

Структура после скачивания:

```
hear_model/
├── .env               # локальные настройки (в .gitignore)
├── Makefile
├── docs/server.md
├── models/hear/       # MODEL_DIR (в .gitignore)
│   ├── saved_model.pb
│   └── variables/
└── hear/              # build context для docker build
    └── python/serving/Dockerfile
```

`make serve` сам собирает образ (`make build`) и стартует контейнер. Первая сборка долгая (apt, pip, gcloud).
Остановить: `Ctrl+C`.

Сервер готов, когда health отвечает `ok`:

```bash
curl http://localhost:8080/health
```

Настройки берутся из `.env` в корне репо. Аргумент командной строки (`make serve PORT=9000`) приоритетнее `.env`.
`.env` читает make, а не shell: кавычки не нужны, вокруг `=` пробелов быть не должно.
Относительные пути (`MODEL_DIR`, `PY`) считаются от корня репо.

| Переменная | По умолчанию | Зачем |
|------------|--------------|-------|
| `HF_TOKEN` | пусто | token Hugging Face для `download-model`. Если пусто, берётся токен из `hf auth login`. Значение в `.env` перекрывает `HF_TOKEN` из shell |
| `PORT` | `8080` | порт API на хосте |
| `MODEL_DIR` | `models/hear` | папка с SavedModel (`saved_model.pb`, `variables/`) |
| `IMAGE` | `hear-serving` | имя Docker-образа |
| `PLATFORM` | `linux/amd64` | платформа образа |
| `PY` | `.venv/bin/python` | Python для `download-model` |

### Отличие от продового запуска

В проде (Vertex AI) entrypoint — `hear/python/serving/entrypoint.sh`: он копирует модель из GCS
(`AIP_STORAGE_URI`) через `gcloud`, а Vertex сам задаёт `AIP_HTTP_PORT`, `AIP_HEALTH_ROUTE`, `AIP_PREDICT_ROUTE`.
`make serve` делает то же самое, но монтирует модель из локальной папки и задаёт переменные сам.

## Эндпоинты

Пути задаются переменными окружения; локально они такие:

| Метод | Путь | Переменная |
|-------|------|-----------|
| `GET` | `/health` | `AIP_HEALTH_ROUTE` |
| `POST` | `/predict` | `AIP_PREDICT_ROUTE` |

### `GET /health`

Проверяет, что TF Serving загрузил модель (`GET localhost:8600/v1/models/default`).

| Код | Тело | Значит |
|-----|------|--------|
| `200` | `ok` | можно слать запросы |
| `503` | `not ok` | модель ещё грузится или TF Serving упал |

### `POST /predict`

`Content-Type: application/json`. Тело — объект с массивом `instances`, по одному аудиоклипу на элемент:

```json
{
  "instances": [
    {"input_array": [0.0, 0.01, ...]},
    {"input_bytes": "UklGRiQ..."},
    {"gcs_uri": "gs://bucket/path/clip.wav", "bearer_token": "ya29..."}
  ]
}
```

В каждом instance ровно **одно** из полей:

| Поле | Тип | Что это |
|------|-----|---------|
| `input_array` | массив из **32000** float | 2 секунды моно-аудио, 16 kHz, значения в `[-1, 1]` |
| `input_bytes` | string (base64) | байты WAV-файла |
| `gcs_uri` | string `gs://.../*.wav` | WAV в Google Cloud Storage |
| `bearer_token` | string | опционально, только вместе с `gcs_uri`: OAuth token для доступа к bucket |

Любые другие ключи (в т.ч. `extensions` из `vertex_schemata/instance.yaml`) → ошибка instance.

Обработка WAV (`input_bytes`, `gcs_uri`):
- `int16`/`int32` нормализуются в float `[-1, 1]`;
- стерео сводится в моно (среднее по каналам);
- sample rate ≠ 16 kHz → resample до 16 kHz;
- после этого длина должна быть ровно 32000 сэмплов (2 с), иначе ошибка. Длинные записи режь на клипы по 2 с на клиенте.

#### Ответ

`200`, по одному элементу `predictions` на каждый instance, в том же порядке:

```json
{
  "predictions": [
    {"embedding": [0.12, -0.03, ...]},
    {"error": {"description": "Failed to get prediction for instance. Reason: Audio array must have 32000 samples. Got 16000 samples."}}
  ]
}
```

- `embedding` — вектор float (у HeAR — 512 значений).
- Ошибка одного instance не валит весь запрос: в его элементе будет `error.description`, код ответа всё равно `200`.

Коды ошибок всего запроса:

| Код | Тело | Когда |
|-----|------|-------|
| `400` | `{"error": "No JSON body."}` | тело не JSON |
| `500` | `{"error": "Internal server error."}` | нет ключа `instances` или иное необработанное исключение |

### Примеры

curl, тишина (32000 нулей):

```bash
python3 -c 'import json; print(json.dumps({"instances": [{"input_array": [0.0]*32000}]}))' \
  | curl -s -X POST http://localhost:8080/predict -H 'Content-Type: application/json' -d @- \
  | head -c 300
```

Python, WAV-файл:

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

- **`/health` долго отвечает `503`** — TF Serving грузит модель; под эмуляцией amd64 это может занять минуту-другую. Смотри логи контейнера.
- **`Нет модели в .../models/hear`** — не выполнен `make download-model`, или `MODEL_DIR` указывает не туда. В папке должен лежать `saved_model.pb`.
- **`401`/`403` при `download-model`** — не принята лицензия модели на Hugging Face или пустой/неверный `HF_TOKEN` в `.env`.
- **`ModuleNotFoundError: huggingface_hub`** — `pip install -r requirements.txt` в `.venv`.
- **Порт занят** — `make serve PORT=9000`.
