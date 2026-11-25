"""The hardware and software environment of an experiment.

Every low-precision result depends on the exact hardware and software it ran
on. The same number format can run natively on one GPU, but be emulated or
not available at all on another GPU. Also, a TensorRT engine only works with
the GPU and the TensorRT version that built it. environment_info() collects
all this information, so we can save it next to each result.

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
    except Exception:  # not installed, or it fails to load (for example, missing DLLs)
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
    """Return a flat dict of versions and GPU features (easy to save as JSON or CSV)."""
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
        # If this PyTorch build has no kernels for this GPU, it fails at the first operation.
        # Kernels built for an older GPU of the same generation also run (for example,
        # sm_86 kernels run on an sm_89 L40S), so any sm_<major><minor or lower> counts.
        built = [arch for arch in torch.cuda.get_arch_list() if arch.startswith("sm_")]
        info["torch_supports_gpu"] = any(
            arch[3:-1] == str(props.major) and int(arch[-1]) <= props.minor for arch in built
            if arch[3:].isdigit()
        )
    return info


if __name__ == "__main__":
    print(json.dumps(environment_info(), indent=2))
