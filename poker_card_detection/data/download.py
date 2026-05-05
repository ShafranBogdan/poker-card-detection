import os
import shutil
from pathlib import Path

from omegaconf import DictConfig


def download_data(cfg: DictConfig) -> Path:
    import roboflow

    raw_dir = Path(cfg.data.raw_dir)

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise EnvironmentError("ROBOFLOW_API_KEY environment variable is not set.")

    rf = roboflow.Roboflow(api_key=api_key)
    project = rf.workspace(cfg.data.workspace).project(cfg.data.project)

    already_downloaded = any(raw_dir.rglob("data.yaml")) if raw_dir.exists() else False

    if not already_downloaded:
        raw_dir.mkdir(parents=True, exist_ok=True)
        dataset = project.version(cfg.data.version).download(
            "yolov8",
            location=str(raw_dir),
            overwrite=True,
        )
        dataset_dir = Path(dataset.location)
    else:
        dataset_dir = raw_dir

    yaml_dst = Path(cfg.data.yaml_path)
    if not yaml_dst.exists():
        found = list(raw_dir.rglob("data.yaml"))
        if not found:
            raise FileNotFoundError(f"data.yaml not found anywhere under {raw_dir}")
        shutil.copy(found[0], yaml_dst)

    return dataset_dir
