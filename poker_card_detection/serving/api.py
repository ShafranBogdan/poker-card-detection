from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from hydra import compose, initialize_config_dir

from poker_card_detection.inference.predict import load_model, predict_from_array

app = FastAPI(title="Poker Card Detection API", version="0.1.0")

_model = None
_inference_cfg = None


def _load_inference_config():
    config_dir = str((Path(__file__).parent.parent.parent / "configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config")
    return cfg.inference


def _get_model():
    global _model, _inference_cfg
    if _model is None:
        model_path = Path("models/best.pt")
        if not model_path.exists():
            raise HTTPException(
                status_code=503,
                detail="Model not found. Run 'poker-cards train' first.",
            )
        _model = load_model(model_path)
        _inference_cfg = _load_inference_config()
    return _model, _inference_cfg


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    np_array = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    model, inference_cfg = _get_model()
    result = predict_from_array(
        model,
        image,
        conf_threshold=inference_cfg.conf_threshold,
        iou_threshold=inference_cfg.iou_threshold,
    )
    return JSONResponse(content=result)
