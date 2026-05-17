# Распознавание карт и покерных комбинаций

Система детектирует игральные карты на фото или в потоке с веб-камеры, определяет масть и номинал каждой карты (52 класса) и классифицирует покерную комбинацию (Royal Flush, Full House и т.д.).

**Датасет:** [Playing Cards Object Detection (Roboflow)](https://universe.roboflow.com/augmented-startups/playing-cards-ow27d)

**Модель:** YOLO11s с дообучением через PyTorch Lightning

---

## Настройка окружения

**Требования:** Python 3.11, [uv](https://docs.astral.sh/uv/)

```bash
# Установить uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Клонировать и установить зависимости
git clone https://github.com/<your-username>/poker-card-detection
cd poker-card-detection
uv sync

# Установить pre-commit хуки
uv run pre-commit install

# Получить бесплатный API ключ на roboflow.com
export ROBOFLOW_API_KEY=your_key_here
```

---

## Обучение

```bash
# 1. Скачать датасет (загружается с Roboflow, версионируется через DVC)
uv run poker-cards download

# 2. Запустить MLflow UI
mlflow ui --port 5001

# 3. Запустить обучение
uv run poker-cards train

# Переопределить гиперпараметры через Hydra:
uv run poker-cards train '["training.epochs=10", "training.batch_size=32"]'
```

---

## Использование Triton

```bash
# Экспортировать обученную модель в ONNX
uv run poker-cards export
# → models/best.onnx

# Подготовить репозиторий моделей для Triton
uv run poker-cards setup-triton
# → models/triton/poker_card_detection/config.pbtxt
# → models/triton/poker_card_detection/1/model.onnx
```

---

## Инференс

**Одно изображение (CLI):**

```bash
uv run poker-cards infer --source path/to/image.jpg
```

**FastAPI сервер — локальная модель (дефолт):**

```bash
uv run poker-cards serve
# → http://localhost:8090
```

**FastAPI сервер — через Triton:**

```bash
# Сначала запустить Triton (требует Docker + NVIDIA GPU):
docker run --gpus all -p 8000:8000 -p 8001:8001 \
  -v $(pwd)/models/triton:/models \
  nvcr.io/nvidia/tritonserver:24.07-py3 \
  tritonserver --model-repository=/models

# Затем запустить FastAPI с Triton-бэкендом:
uv run poker-cards serve --use-triton true
```

Проверить активный бэкенд:

```bash
curl http://localhost:8090/health
# {"status": "ok", "backend": "local"}
# {"status": "ok", "backend": "triton"}
```

Пример запроса к API:

```bash
curl -X POST http://localhost:8090/detect \
  -F "file=@poker_table.jpg"
```

Пример ответа:

```json
{
  "detections": [
    { "class": "AH", "confidence": 0.96, "bbox": [0.32, 0.45, 0.12, 0.18] },
    { "class": "KH", "confidence": 0.94, "bbox": [0.55, 0.44, 0.11, 0.17] }
  ],
  "poker_hand": "One Pair",
  "cards_found": 2
}
```

**Streamlit UI** (веб-камера + загрузка фото):

```bash
# FastAPI должен быть запущен
uv run streamlit run poker_card_detection/serving/ui.py
# → http://localhost:8501
```
