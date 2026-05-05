from pathlib import Path
from types import SimpleNamespace

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from ultralytics.data.utils import check_det_dataset
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.torch_utils import intersect_dicts


def build_model(cfg: DictConfig) -> DetectionModel:
    from ultralytics.utils.downloads import attempt_download_asset

    data_info = check_det_dataset(str(Path(cfg.data.yaml_path).resolve()))
    num_classes = len(data_info["names"])

    arch = cfg.model.weights.replace(".pt", ".yaml")
    model = DetectionModel(arch, nc=num_classes, verbose=False)
    model.names = data_info["names"]
    model.nc = num_classes

    weights_path = attempt_download_asset(cfg.model.weights)
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    pretrained_state = ckpt["model"].float().state_dict() if "model" in ckpt else ckpt
    current_state = model.state_dict()
    matched = intersect_dicts(pretrained_state, current_state, exclude=["Detect"])
    model.load_state_dict(matched, strict=False)

    return model


class YOLOLightningModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self.cfg = cfg

        self.model = build_model(cfg)
        self.model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)

        for param in self.model.parameters():
            param.requires_grad_(True)

        self.criterion = v8DetectionLoss(self.model)

    def on_fit_start(self):
        self.model.train()
        self._sync_criterion_to_device()

    def on_validation_epoch_start(self):
        self._sync_criterion_to_device()

    def on_test_epoch_start(self):
        self._sync_criterion_to_device()

    def _sync_criterion_to_device(self):
        if hasattr(self.criterion, "proj"):
            self.criterion.proj = self.criterion.proj.to(self.device)
        if hasattr(self.criterion, "device"):
            self.criterion.device = self.device

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def _shared_step(self, batch: dict, prefix: str) -> torch.Tensor:
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        batch["img"] = batch["img"].float() / 255.0
        predictions = self.model(batch["img"])
        loss_scaled, loss_components = self.criterion(predictions, batch)

        loss = loss_scaled.sum()

        self.log(
            f"{prefix}/box_loss", loss_components[0], prog_bar=True, sync_dist=True
        )
        self.log(
            f"{prefix}/cls_loss", loss_components[1], prog_bar=True, sync_dist=True
        )
        self.log(
            f"{prefix}/dfl_loss", loss_components[2], prog_bar=True, sync_dist=True
        )
        self.log(f"{prefix}/total_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.training.epochs,
            eta_min=self.cfg.training.lr * 0.01,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
