from pathlib import Path

import fire
from hydra import compose, initialize_config_dir


def _load_cfg(overrides=None):
    config_dir = str((Path(__file__).parent.parent / "configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name="config", overrides=overrides or [])


def download(overrides=None):
    cfg = _load_cfg(overrides)
    from poker_card_detection.data.download import download_data

    download_data(cfg)


def train(overrides=None):
    cfg = _load_cfg(overrides)
    from poker_card_detection.training.train import run_training

    run_training(cfg)


def export(overrides=None):
    cfg = _load_cfg(overrides)
    from poker_card_detection.training.train import export_to_onnx

    export_to_onnx(cfg)


def setup_triton(overrides=None):
    cfg = _load_cfg(overrides)
    from poker_card_detection.serving.triton_setup import prepare_triton_repository

    prepare_triton_repository(cfg)


def infer(source: str, model_path: str = "models/best.pt", overrides=None):
    import json

    cfg = _load_cfg(overrides)
    from poker_card_detection.inference.predict import load_model, predict_image

    model = load_model(model_path)
    result = predict_image(
        model,
        source,
        conf_threshold=cfg.inference.conf_threshold,
        iou_threshold=cfg.inference.iou_threshold,
    )
    print(json.dumps(result, indent=2))


def serve(use_triton: bool = False, overrides=None):
    import uvicorn

    cfg = _load_cfg(overrides)

    from poker_card_detection.serving.api import initialize

    initialize(use_triton=use_triton)

    uvicorn.run(
        "poker_card_detection.serving.api:app",
        host=cfg.serving.api_host,
        port=cfg.serving.api_port,
        reload=False,
    )


def main():
    fire.Fire(
        {
            "download": download,
            "train": train,
            "export": export,
            "setup-triton": setup_triton,
            "infer": infer,
            "serve": serve,
        }
    )


if __name__ == "__main__":
    main()
