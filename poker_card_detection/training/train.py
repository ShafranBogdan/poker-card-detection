from pathlib import Path

import mlflow
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from poker_card_detection.data.dataset import CardDataModule
from poker_card_detection.models.yolo_lightning import YOLOLightningModule


def run_training(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)

    _pull_data(cfg)

    data_module = CardDataModule(cfg)
    model = YOLOLightningModule(cfg)

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name,
        tracking_uri=cfg.mlflow.tracking_uri,
        tags={"git_commit": _get_git_commit()},
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="models/checkpoints",
        filename="best-{epoch:02d}-{val/total_loss:.4f}",
        monitor="val/total_loss",
        mode="min",
        save_top_k=1,
    )
    early_stopping = EarlyStopping(
        monitor="val/total_loss",
        patience=cfg.training.patience,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        logger=mlflow_logger,
        callbacks=[checkpoint_callback, early_stopping],
        precision=cfg.training.precision,
        log_every_n_steps=10,
        default_root_dir="models",
    )

    trainer.fit(model, datamodule=data_module)
    mlflow_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    trainer.test(model, datamodule=data_module)

    best_ckpt = checkpoint_callback.best_model_path
    _compute_and_log_map(best_ckpt, cfg, mlflow_logger.run_id)
    _save_final_weights(best_ckpt, cfg)


def export_to_onnx(cfg: DictConfig) -> Path:
    from ultralytics import YOLO

    model_path = Path("models/best.pt")
    yolo = YOLO(str(model_path))
    yolo.export(format="onnx", imgsz=cfg.data.image_size, simplify=True, dynamic=False)
    return model_path.with_suffix(".onnx")


def _pull_data(cfg: DictConfig) -> None:
    data_yaml = Path(cfg.data.yaml_path)
    if data_yaml.exists():
        return

    try:
        import dvc.api

        dvc.api.params_show()
    except Exception:
        from poker_card_detection.data.download import download_data

        download_data(cfg)


def _get_git_commit() -> str:
    try:
        import git

        repo = git.Repo(search_parent_directories=True)
        return repo.head.commit.hexsha[:8]
    except Exception:
        return "unknown"


def _compute_and_log_map(checkpoint_path: str, cfg: DictConfig, run_id: str) -> None:
    from ultralytics import YOLO

    best_model = YOLOLightningModule.load_from_checkpoint(checkpoint_path, cfg=cfg)

    yolo = YOLO(cfg.model.weights)
    yolo.model = best_model.model

    results = yolo.val(
        data=cfg.data.yaml_path, imgsz=cfg.data.image_size, verbose=False
    )

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(
            {
                "val/mAP50": float(results.box.map50),
                "val/mAP50-95": float(results.box.map),
            }
        )


def _save_final_weights(checkpoint_path: str, cfg: DictConfig) -> None:
    import torch
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset

    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)

    best_lightning = YOLOLightningModule.load_from_checkpoint(checkpoint_path, cfg=cfg)

    data_info = check_det_dataset(str(Path(cfg.data.yaml_path).resolve()))

    yolo = YOLO(cfg.model.weights)
    yolo.model = best_lightning.model
    yolo.model.names = data_info["names"]
    yolo.model.nc = len(data_info["names"])

    torch.save(
        {"model": yolo.model, "names": data_info["names"]}, output_dir / "best.pt"
    )
