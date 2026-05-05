from pathlib import Path

import pytorch_lightning as pl
from omegaconf import DictConfig
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset


class CardDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.data_yaml = str(Path(cfg.data.yaml_path).resolve())
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

    def _make_args(self, augment: bool = True):
        aug = self.cfg.data.augment
        args = get_cfg()
        args.task = "detect"
        args.data = self.data_yaml
        args.imgsz = self.cfg.data.image_size
        args.batch = self.cfg.training.batch_size
        args.workers = self.cfg.data.workers
        args.augment = augment
        args.rect = False
        args.cache = False
        args.single_cls = False
        args.classes = None
        args.fraction = 1.0
        args.degrees = aug.degrees
        args.translate = aug.translate
        args.scale = aug.scale
        args.fliplr = aug.fliplr
        args.mosaic = float(aug.mosaic) if augment else 0.0
        args.hsv_h = aug.hsv_h
        args.hsv_s = aug.hsv_s
        args.hsv_v = aug.hsv_v
        return args

    def setup(self, stage=None):
        data_info = check_det_dataset(self.data_yaml)

        if stage in (None, "fit"):
            self._train_dataset = build_yolo_dataset(
                self._make_args(augment=True),
                data_info["train"],
                self.cfg.training.batch_size,
                data_info,
                mode="train",
            )
            self._val_dataset = build_yolo_dataset(
                self._make_args(augment=False),
                data_info["val"],
                self.cfg.training.batch_size,
                data_info,
                mode="val",
            )

        if stage in (None, "test"):
            test_path = data_info.get("test", data_info["val"])
            self._test_dataset = build_yolo_dataset(
                self._make_args(augment=False),
                test_path,
                self.cfg.training.batch_size,
                data_info,
                mode="val",
            )

    def train_dataloader(self):
        return build_dataloader(
            self._train_dataset,
            self.cfg.training.batch_size,
            self.cfg.data.workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return build_dataloader(
            self._val_dataset,
            self.cfg.training.batch_size,
            self.cfg.data.workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return build_dataloader(
            self._test_dataset,
            self.cfg.training.batch_size,
            self.cfg.data.workers,
            shuffle=False,
        )
