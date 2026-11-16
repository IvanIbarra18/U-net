"""Hardware / software environment of an experiment.

Every low-precision result depends on the exact stack it ran on: the same
format can be native on one GPU and emulated (or unavailable) on another,
and TensorRT engines are only valid for the GPU and TensorRT version that
built them. environment_info() collects that stack so it can be stored next
to each result.

Usage (from the repository root):
    python -m quantization.env_report
"""

import importlib
import json
import platform
import sys

import torch


def _version(module_name: str):
    try:
        return getattr(importlib.import_module(module_name), "__version__", "unknown")
    except Exception:  # not installed, or fails to load (missing DLLs, ...)
        return None


def _driver_version():
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            return pynvml.nvmlSystemGetDriverVersion()
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None


def environment_info(device_index: int = 0) -> dict:
    """Flat dict of versions and GPU capabilities (JSON/CSV friendly)."""
    info = {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "torch_arch_list": " ".join(torch.cuda.get_arch_list()) if torch.cuda.is_available() else None,
        "tensorrt": _version("tensorrt"),
        "onnx": _version("onnx"),
        "onnxruntime": _version("onnxruntime"),
        "driver": _driver_version(),
        "gpu": None,
        "compute_capability": None,
        "gpu_memory_mib": None,
        "torch_supports_gpu": None,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device_index)
        info["gpu"] = props.name
        info["compute_capability"] = f"{props.major}.{props.minor}"
        info["gpu_memory_mib"] = props.total_memory // 2 ** 20
        # A PyTorch build without kernels for this GPU fails at the first op.
        sm = f"sm_{props.major}{props.minor}"
        info["torch_supports_gpu"] = sm in torch.cuda.get_arch_list()
    return info


if __name__ == "__main__":
    print(json.dumps(environment_info(), indent=2))
