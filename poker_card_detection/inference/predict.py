from pathlib import Path
from typing import Union

import numpy as np
from ultralytics import YOLO

from poker_card_detection.inference.poker_hand import classify_hand


def load_model(model_path: Union[str, Path]) -> YOLO:
    return YOLO(str(model_path))


def predict_image(
    model: YOLO, image_path: Union[str, Path], conf_threshold: float = 0.5
) -> dict:
    results = model(str(image_path), conf=conf_threshold, verbose=False)[0]
    return _parse_results(results)


def predict_from_array(
    model: YOLO, image: np.ndarray, conf_threshold: float = 0.5
) -> dict:
    results = model(image, conf=conf_threshold, verbose=False)[0]
    return _parse_results(results)


def _parse_results(results) -> dict:
    detections = []
    labels = []

    for box in results.boxes:
        class_id = int(box.cls[0])
        label = results.names[class_id]
        confidence = float(box.conf[0])
        bbox = box.xywhn[0].tolist()

        detections.append(
            {"class": label, "confidence": round(confidence, 4), "bbox": bbox}
        )
        labels.append(label)

    return {
        "detections": detections,
        "poker_hand": classify_hand(labels),
        "cards_found": len(detections),
    }
