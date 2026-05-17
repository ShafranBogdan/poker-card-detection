from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from hydra import compose, initialize_config_dir

from poker_card_detection.inference.predict import load_model, predict_from_array

app = FastAPI(title="Poker Card Detection API", version="0.1.0")

_local_model = None
_triton_client = None
_inference_cfg = None
_use_triton = False


def _load_config():
    config_dir = str((Path(__file__).parent.parent.parent / "configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name="config")


def initialize(use_triton: bool = False):
    global _local_model, _triton_client, _inference_cfg, _use_triton

    cfg = _load_config()
    _inference_cfg = cfg.inference
    _use_triton = use_triton

    if use_triton:
        from poker_card_detection.serving.triton_client import build_triton_client

        _triton_client = build_triton_client(cfg)
        if not _triton_client.is_ready():
            raise RuntimeError(
                f"Triton model '{cfg.serving.triton_model_name}' is not ready. "
                "Run 'poker-cards setup-triton' and start Triton server first."
            )
    else:
        model_path = Path("models/best.pt")
        if not model_path.exists():
            raise RuntimeError("Model not found. Run 'poker-cards train' first.")
        _local_model = load_model(model_path)


@app.get("/health")
async def health():
    return {"status": "ok", "backend": "triton" if _use_triton else "local"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    np_array = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    if _use_triton:
        if _triton_client is None:
            raise HTTPException(
                status_code=503, detail="Triton client not initialized."
            )
        result = _triton_client.predict(
            image,
            conf_threshold=_inference_cfg.conf_threshold,
            iou_threshold=_inference_cfg.iou_threshold,
        )
    else:
        if _local_model is None:
            raise HTTPException(status_code=503, detail="Local model not initialized.")
        result = predict_from_array(
            _local_model,
            image,
            conf_threshold=_inference_cfg.conf_threshold,
            iou_threshold=_inference_cfg.iou_threshold,
        )

    return JSONResponse(content=result)
