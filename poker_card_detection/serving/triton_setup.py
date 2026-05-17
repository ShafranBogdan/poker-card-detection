import shutil
from pathlib import Path

from omegaconf import DictConfig


def prepare_triton_repository(cfg: DictConfig) -> Path:
    onnx_path = Path("models/best.onnx")
    if not onnx_path.exists():
        raise FileNotFoundError(
            "models/best.onnx not found. Run 'poker-cards export' first."
        )

    triton_dir = Path("models/triton") / cfg.serving.triton_model_name
    model_dir = triton_dir / str(cfg.serving.triton_model_version)
    model_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(onnx_path, model_dir / "model.onnx")

    config_pbtxt = _generate_config(cfg)
    (triton_dir / "config.pbtxt").write_text(config_pbtxt)

    print(f"Triton repository prepared at: {triton_dir}")
    print("Start Triton with:")
    print(
        f"  docker run --gpus all -p {cfg.serving.triton_http_port}:8000 "
        f"-p {cfg.serving.triton_grpc_port}:8001 "
        f"-v $(pwd)/models/triton:/models "
        f"nvcr.io/nvidia/tritonserver:24.07-py3 "
        f"tritonserver --model-repository=/models"
    )
    return triton_dir


def _generate_config(cfg: DictConfig) -> str:
    size = cfg.data.image_size
    return f"""name: "{cfg.serving.triton_model_name}"
backend: "onnxruntime"
max_batch_size: 0

input [
  {{
    name: "images"
    data_type: TYPE_FP32
    dims: [1, 3, {size}, {size}]
  }}
]

output [
  {{
    name: "output0"
    data_type: TYPE_FP32
    dims: [-1, -1]
  }}
]

instance_group [
  {{
    kind: KIND_GPU
    count: 1
  }}
]
"""
