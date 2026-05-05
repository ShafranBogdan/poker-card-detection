from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from poker_card_detection.inference.predict import load_model, predict_from_array

app = FastAPI(title="Poker Card Detection API", version="0.1.0")

_model = None


def _get_model():
    global _model
    if _model is None:
        model_path = Path("models/best.pt")
        if not model_path.exists():
            raise HTTPException(
                status_code=503,
                detail="Model not found. Run 'poker-cards train' first.",
            )
        _model = load_model(model_path)
    return _model


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

    model = _get_model()
    result = predict_from_array(model, image)
    return JSONResponse(content=result)
