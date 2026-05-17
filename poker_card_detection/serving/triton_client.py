from pathlib import Path

import cv2
import numpy as np
import torch
import tritonclient.http as httpclient
from ultralytics.utils.ops import non_max_suppression

from poker_card_detection.inference.poker_hand import classify_hand
from poker_card_detection.inference.predict import _deduplicate_by_class

INPUT_NAME = "images"
OUTPUT_NAME = "output0"


class TritonClient:
    def __init__(
        self,
        host: str,
        http_port: int,
        model_name: str,
        model_version: str,
        class_names: dict,
        image_size: int = 640,
    ):
        self.client = httpclient.InferenceServerClient(url=f"{host}:{http_port}")
        self.model_name = model_name
        self.model_version = str(model_version)
        self.class_names = class_names
        self.image_size = image_size

    def is_ready(self) -> bool:
        try:
            return self.client.is_model_ready(self.model_name, self.model_version)
        except Exception:
            return False

    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
    ) -> dict:
        tensor = self._preprocess(image)

        infer_input = httpclient.InferInput(INPUT_NAME, tensor.shape, "FP32")
        infer_input.set_data_from_numpy(tensor)

        infer_output = httpclient.InferRequestedOutput(OUTPUT_NAME)
        response = self.client.infer(
            self.model_name,
            [infer_input],
            outputs=[infer_output],
            model_version=self.model_version,
        )

        raw_output = torch.from_numpy(response.as_numpy(OUTPUT_NAME))
        nms_results = non_max_suppression(
            raw_output, conf_thres=conf_threshold, iou_thres=iou_threshold
        )[0]

        detections = self._parse_nms_output(nms_results)
        detections = _deduplicate_by_class(detections)
        labels = [det["class"] for det in detections]

        return {
            "detections": detections,
            "poker_hand": classify_hand(labels),
            "cards_found": len(detections),
        }

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (self.image_size, self.image_size))
        tensor = image_resized.astype(np.float32) / 255.0
        return np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]

    def _parse_nms_output(self, detections: torch.Tensor) -> list[dict]:
        results = []
        for det in detections.tolist():
            x1, y1, x2, y2, conf, cls_id = det
            cx = (x1 + x2) / 2 / self.image_size
            cy = (y1 + y2) / 2 / self.image_size
            w = (x2 - x1) / self.image_size
            h = (y2 - y1) / self.image_size
            results.append(
                {
                    "class": self.class_names[int(cls_id)],
                    "confidence": round(conf, 4),
                    "bbox": [cx, cy, w, h],
                }
            )
        return results


def build_triton_client(cfg) -> TritonClient:
    from ultralytics.data.utils import check_det_dataset

    data_info = check_det_dataset(str(Path(cfg.data.yaml_path).resolve()))
    return TritonClient(
        host=cfg.serving.triton_host,
        http_port=cfg.serving.triton_http_port,
        model_name=cfg.serving.triton_model_name,
        model_version=cfg.serving.triton_model_version,
        class_names=data_info["names"],
        image_size=cfg.data.image_size,
    )
