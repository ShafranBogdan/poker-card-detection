# Распознавание карт и покерных комбинаций

Система детектирует игральные карты на фото или в потоке с веб-камеры, определяет масть и номинал каждой карты (52 класса) и классифицирует покерную комбинацию (Royal Flush, Full House и т.д.).

**Датасет:** [Playing Cards Object Detection (Roboflow)](https://universe.roboflow.com/augmented-startups/playing-cards-ow27d)
— 10 100 изображений, 52 класса, аннотации в формате YOLO, ~1–2 ГБ.

**Модель:** YOLO11s с дообучением через PyTorch Lightning.
Ожидаемый mAP@0.5 ≥ 0.90 после обучения.

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

# Получить бесплатный API ключ на roboflow.com и задать переменную окружения
export ROBOFLOW_API_KEY=your_key_here
```

---

## Обучение

```bash
# 1. Скачать датасет (загружается с Roboflow, версионируется через DVC)
uv run poker-cards download

# 2. Запустить MLflow UI (в отдельном терминале)
mlflow ui --port 8080

# 3. Запустить обучение (YOLO11s, 100 эпох, логи в MLflow на 127.0.0.1:8080)
uv run poker-cards train

# Переопределить гиперпараметры через синтаксис Hydra:
uv run poker-cards train '["training.epochs=10", "training.batch_size=32"]'
```

Результаты экспериментов — в MLflow UI по адресу `http://127.0.0.1:8080`.

---

## Подготовка к продакшену

```bash
# Экспортировать обученную модель в ONNX
uv run poker-cards export
# → models/best.onnx

# (Опционально) Конвертировать в TensorRT из ONNX:
./scripts/export_tensorrt.sh
```

Для запуска инференса достаточно одного артефакта: `models/best.pt` (или `models/best.onnx`).

---

## Инференс

**Одно изображение:**

```bash
uv run poker-cards infer --source path/to/image.jpg
```

**FastAPI сервер** (принимает `POST /detect` с файлом изображения):

```bash
uv run poker-cards serve
# → http://localhost:8090/detect
```

Пример запроса:

```bash
curl -X POST http://localhost:8090/detect \
  -F "file=@poker_table.jpg"
```

Пример ответа:

```json
{
  "detections": [
    { "class": "Ah", "confidence": 0.96, "bbox": [0.32, 0.45, 0.12, 0.18] }
  ],
  "poker_hand": "Royal Flush",
  "cards_found": 5
}
```

**Streamlit UI** (веб-камера + загрузка фото):

```bash
uv run streamlit run poker_card_detection/serving/ui.py
# → http://localhost:8501
```

**Triton Inference Server:**

```bash
docker run --gpus all -p 8000:8000 -p 8001:8001 \
  -v $(pwd)/models/triton:/models \
  nvcr.io/nvidia/tritonserver:24.07-py3 \
  tritonserver --model-repository=/models
```
