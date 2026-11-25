"""Build, inspect and run TensorRT engines from the exported ONNX U-Net.

Pipeline:  checkpoint --export_onnx.py--> ONNX --build_engine()--> engine
           test.py --backend tensorrt runs the engine through TensorRTModel.

The ONNX graph decides the precision, not TensorRT. We create the network as
STRONGLY_TYPED. This means every tensor keeps the dtype written in the graph
(FP32, or INT8, FP8 and others through explicit Q/DQ nodes), and TensorRT is
not allowed to choose a lower precision by itself. (TensorRT 11 removed the
FP16 and INT8 builder flags completely. In TensorRT 10, strong typing is
optional and you must turn it on.)

TF32: like cuDNN, TensorRT turns it on by default for FP32 convolutions.
build_engine() turns it off unless tf32=True, so an "FP32" engine really is FP32.

We do not guess what really ran. We read it back from the built engine with
the engine inspector: layer types, input and output dtypes, and kernel
"tactic" names. For example, sm80_xmma_fprop_implicit_gemm_f32f32_f32f32_f32
means FP32 inputs, FP32 sums and FP32 output. We save this as JSON next to
the engine and summarize it in the results, so we never claim a precision
without proof.

An engine only works on the GPU model and TensorRT version that built it.
So we cache one engine for each (ONNX hash, GPU, TensorRT, profile, TF32),
and we build them again on the cluster.
"""

import collections
import json
import re
import time
from datetime import datetime
from pathlib import Path

import torch

from quantization.env_report import environment_info
from quantization.export_onnx import sha256

# Layer types that do the convolution work. The inspector names them in two ways,
# depending on how TensorRT compiled the graph (verified with TensorRT 11.4 / 11.5):
#   Myelin (RTX 5080):        "correlation", "deconv"
#   classic builder (L40S):   "CaskConvolution", "CaskDeconvolutionV2", "CaskGemmDeconvolution"
# layer_kind() maps both to the same two names, so the reports look the same on every GPU.
def layer_kind(layer_type) -> str | None:
    """Return "correlation" (a convolution), "deconv", "gemm" or None (not a compute layer)."""
    name = (layer_type or "").lower()
    if "deconv" in name:
        return "deconv"
    if "conv" in name or name == "correlation":
        return "correlation"
    if "gemm" in name or "matrix_multiply" in name:
        return "gemm"
    return None


def _trt():
    import tensorrt  # imported only when needed, so runs that use only PyTorch do not need it

    return tensorrt


def engine_path_for(onnx_path: Path, profile: dict, tf32: bool) -> Path:
    """Return the cache file name. There is one engine for each ONNX file,
    GPU, TensorRT version, shape profile and TF32 setting."""
    trt = _trt()
    gpu = re.sub(r"[^A-Za-z0-9]+", "-", torch.cuda.get_device_name()).strip("-")
    (min_h, min_w), _, (max_h, max_w) = profile["min"], profile["opt"], profile["max"]
    name = (
        f"{onnx_path.stem}__{gpu}_trt{trt.__version__}"
        f"_{min_h}x{min_w}-{max_h}x{max_w}{'_tf32' if tf32 else ''}.engine"
    )
    return onnx_path.parent / "engines" / name


def build_engine(onnx_path: Path, engine_path: Path, profile: dict, tf32: bool = False, n_channels: int = 3) -> Path:
    """Read the ONNX file into a strongly typed network and build an engine.

    The engine is for batch size 1 and image sizes (H, W) from profile["min"]
    to profile["max"]. Its kernels are tuned for profile["opt"]."""
    trt = _trt()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT could not parse {onnx_path}:\n{errors}")

    config = builder.create_builder_config()
    if tf32:
        config.set_flag(trt.BuilderFlag.TF32)
    else:
        config.clear_flag(trt.BuilderFlag.TF32)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED  # keep tactic names for the inspector

    input_name = network.get_input(0).name
    shapes = [(1, n_channels, *profile[key]) for key in ("min", "opt", "max")]
    optimization_profile = builder.create_optimization_profile()
    optimization_profile.set_shape(input_name, *shapes)
    config.add_optimization_profile(optimization_profile)

    print(f"Building TensorRT engine for {onnx_path.name} (shapes {shapes[0][2:]}..{shapes[2][2:]}, "
          f"TF32 {'on' if tf32 else 'off'}); this can take a few minutes...")
    start = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed (see the log above).")
    build_seconds = time.perf_counter() - start

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))
    engine = trt.Runtime(logger).deserialize_cuda_engine(serialized)
    layers = inspect_layers(engine)
    engine_path.with_suffix(".layers.json").write_text(json.dumps(layers, indent=1), encoding="utf-8")
    engine_path.with_suffix(".json").write_text(json.dumps({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "onnx": str(onnx_path),
        "onnx_sha256": sha256(onnx_path),
        "profile": profile,
        "tf32": tf32,
        "strongly_typed": True,
        "build_seconds": round(build_seconds, 1),
        "engine_bytes": engine_path.stat().st_size,
        "layer_summary": summarize_layers(layers),
        "environment": environment_info(),
    }, indent=2), encoding="utf-8")
    print(f"Engine built in {build_seconds:.0f} s: {engine_path}")
    return engine_path


def get_engine(onnx_path: Path, profile: dict, tf32: bool = False, n_channels: int = 3, rebuild: bool = False) -> Path:
    """Return the path to a cached engine for these settings.

    We build the engine if it does not exist, if `rebuild` is True, or if the
    ONNX file changed after the engine was built."""
    engine_path = engine_path_for(onnx_path, profile, tf32)
    metadata_path = engine_path.with_suffix(".json")
    if not rebuild and engine_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("onnx_sha256") == sha256(onnx_path) and metadata.get("profile") == profile:
            return engine_path
        print("The ONNX file or the profile changed since the cached engine was built; rebuilding.")
    return build_engine(onnx_path, engine_path, profile, tf32=tf32, n_channels=n_channels)


def inspect_layers(engine) -> list:
    """Return information for each layer of a built engine: name, type,
    tactic and tensor dtypes. (The engine must be built with DETAILED
    profiling verbosity.)"""
    trt = _trt()
    info = json.loads(engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
    layers = []
    for layer in info["Layers"]:
        if not isinstance(layer, dict):  # verbosity was too low, so we only have names
            layers.append({"name": layer})
            continue
        layers.append({
            "name": layer.get("Name"),
            "type": layer.get("LayerType"),
            "tactic": layer.get("TacticName", ""),
            "inputs": [tensor.get("Datatype") for tensor in layer.get("Inputs", [])],
            "outputs": [tensor.get("Datatype") for tensor in layer.get("Outputs", [])],
        })
    return layers


# Kernel names include their dtypes. For example, ..._i8i8_i8i32_f32_... means:
# activation in i8 and output in i8, weights in i8 and sums (accumulator) in
# i32, and the final step (epilogue) in f32.
_DT = r"i8|i4|f16|bf16|tf32|f32|e4m3|e5m2|e2m1|fp8|fp4"
_KERNEL_DTYPES = re.compile(rf"_(?P<input>{_DT})(?P<output>{_DT})_(?P<weight>{_DT})_?(?P<accumulator>i32|f32|f16)_")


def kernel_dtypes(tactic: str) -> dict | None:
    """Read {input, output, weight, accumulator} from a tactic name.

    Return None if the name does not match."""
    match = _KERNEL_DTYPES.search(tactic or "")
    return match.groupdict() if match else None


def summarize_layers(layers: list) -> dict:
    """Answer the question: in what precision did the convolutions really run?

    compute_layers  conv and deconv layers counted by (input dtype -> output
                    dtype), as the engine inspector reports them.
    weights,        conv and deconv layers counted by weight dtype and by
    accumulators    accumulator dtype, read from the kernel names
                    ("unparsed" if we cannot read them).
    tf32_tactics    the number of kernels whose name says TF32 (0 when TF32
                    is off).
    """
    compute = [layer for layer in layers if layer_kind(layer.get("type"))]
    by_dtype = collections.Counter(
        f"{layer_kind(layer['type'])}:{(layer['inputs'] or ['?'])[0]}->{(layer['outputs'] or ['?'])[0]}"
        for layer in compute
    )
    parsed = [kernel_dtypes(layer.get("tactic", "")) for layer in compute]
    return {
        "layers": len(layers),
        "layer_types": dict(collections.Counter(layer.get("type") for layer in layers)),
        "compute_layers": dict(by_dtype),
        "weights": dict(collections.Counter(p["weight"] if p else "unparsed" for p in parsed)),
        "accumulators": dict(collections.Counter(p["accumulator"] if p else "unparsed" for p in parsed)),
        "tf32_tactics": sum("tf32" in layer.get("tactic", "").lower() for layer in compute),
    }


class TensorRTModel:
    """You can call this like a PyTorch model: FP32 CUDA images [1, C, H, W] -> logits.

    The execution context uses USER_MANAGED device memory that comes from a
    PyTorch tensor. So torch.cuda.max_memory_allocated() includes the memory
    that TensorRT uses for activations. TensorRT allocates the engine weights
    itself, so we report them separately (as the engine size).
    """

    def __init__(self, engine_path: Path):
        trt = _trt()
        self.engine_path = Path(engine_path)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(self.engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Could not load {engine_path} (built for another GPU or TensorRT version?)")
        self.context = self.engine.create_execution_context(trt.ExecutionContextAllocationStrategy.USER_MANAGED)
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_name = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.output_name = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        self.stream = torch.cuda.Stream()
        self.workspace = None  # it grows when needed and never gets smaller
        self.engine_bytes = self.engine_path.stat().st_size
        self.metadata = json.loads(self.engine_path.with_suffix(".json").read_text(encoding="utf-8"))

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        if images.dtype != torch.float32 or not images.is_cuda:
            raise TypeError("TensorRTModel expects FP32 CUDA images.")
        images = images.contiguous()
        if not self.context.set_input_shape(self.input_name, tuple(images.shape)):
            raise ValueError(f"Input shape {tuple(images.shape)} is outside the engine profile {self.metadata['profile']}.")
        needed = self.context.update_device_memory_size_for_shapes()
        if self.workspace is None or self.workspace.numel() < needed:
            self.workspace = None  # free the old buffer before we allocate a new one
            self.workspace = torch.empty(max(needed, 1), dtype=torch.uint8, device=images.device)
        self.context.set_device_memory(self.workspace.data_ptr(), self.workspace.numel())

        logits = torch.empty(tuple(self.context.get_tensor_shape(self.output_name)), dtype=torch.float32, device=images.device)
        self.context.set_tensor_address(self.input_name, images.data_ptr())
        self.context.set_tensor_address(self.output_name, logits.data_ptr())
        # The engine must wait for the work that made `images`, and later work
        # must wait for the engine. We do this without syncing the whole device.
        current = torch.cuda.current_stream(images.device)
        self.stream.wait_stream(current)
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed.")
        current.wait_stream(self.stream)
        for tensor in (images, logits, self.workspace):
            tensor.record_stream(self.stream)  # not reused while the engine may still read it
        return logits

    def layer_summary(self) -> dict:
        # We compute this again from the saved per-layer file. This way, engines
        # cached before we added a summary field still report that field.
        layers_path = self.engine_path.with_suffix(".layers.json")
        return summarize_layers(json.loads(layers_path.read_text(encoding="utf-8")))


def profile_from_sizes(sizes) -> dict:
    """Return an optimization profile that covers every (H, W) in `sizes`.

    The kernels are tuned for the most common size."""
    sizes = list(sizes)
    heights = [h for h, _ in sizes]
    widths = [w for _, w in sizes]
    opt = collections.Counter(sizes).most_common(1)[0][0]
    return {"min": [min(heights), min(widths)], "opt": list(opt), "max": [max(heights), max(widths)]}
