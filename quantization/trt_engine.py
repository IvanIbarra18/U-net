"""Build, inspect and run TensorRT engines from the exported ONNX U-Net.

Pipeline:  checkpoint --export_onnx.py--> ONNX --build_engine()--> engine
           test.py --backend tensorrt runs the engine through TensorRTModel.

Precision is decided by the ONNX graph, not by TensorRT. The network is
created STRONGLY_TYPED: every tensor keeps the dtype written in the graph
(FP32 now; INT8/FP8/... later through explicit Q/DQ nodes), and TensorRT may
not pick a lower precision on its own. (TensorRT 11 removed the FP16/INT8
builder flags entirely; in TensorRT 10 strong typing is opt-in.)

TF32: TensorRT enables it by default for FP32 convolutions, like cuDNN.
build_engine() disables it unless tf32=True, so an "FP32" engine is FP32.

What actually ran is read back from the built engine with the engine
inspector (layer types, input/output dtypes and kernel "tactic" names, e.g.
sm80_xmma_fprop_implicit_gemm_f32f32_f32f32_f32 = FP32 inputs, FP32
accumulation, FP32 output). It is stored next to the engine as JSON and
summarized in the results, so a precision is never claimed without evidence.

Engines are specific to the GPU model and TensorRT version that built them,
so they are cached per (ONNX hash, GPU, TensorRT, profile, TF32) and rebuilt
on the cluster.
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

# Layer types that do the convolution work (names used by the inspector).
COMPUTE_LAYER_TYPES = {"correlation", "convolution", "deconv", "deconvolution", "gemm", "matrix_multiply"}


def _trt():
    import tensorrt  # imported lazily: PyTorch-only runs must not need it

    return tensorrt


def engine_path_for(onnx_path: Path, profile: dict, tf32: bool) -> Path:
    """Cache file name: one engine per ONNX file, GPU, TensorRT version,
    shape profile and TF32 setting."""
    trt = _trt()
    gpu = re.sub(r"[^A-Za-z0-9]+", "-", torch.cuda.get_device_name()).strip("-")
    (min_h, min_w), _, (max_h, max_w) = profile["min"], profile["opt"], profile["max"]
    name = (
        f"{onnx_path.stem}__{gpu}_trt{trt.__version__}"
        f"_{min_h}x{min_w}-{max_h}x{max_w}{'_tf32' if tf32 else ''}.engine"
    )
    return onnx_path.parent / "engines" / name


def build_engine(onnx_path: Path, engine_path: Path, profile: dict, tf32: bool = False, n_channels: int = 3) -> Path:
    """Parse the ONNX file into a strongly typed network and build an engine
    for batch 1 and image sizes profile["min"] .. profile["max"] (H, W),
    with kernels tuned for profile["opt"]."""
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
    """Path to a cached engine for this configuration, building it if it is
    missing, if `rebuild`, or if the ONNX file changed since it was built."""
    engine_path = engine_path_for(onnx_path, profile, tf32)
    metadata_path = engine_path.with_suffix(".json")
    if not rebuild and engine_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("onnx_sha256") == sha256(onnx_path) and metadata.get("profile") == profile:
            return engine_path
        print("The ONNX file or the profile changed since the cached engine was built; rebuilding.")
    return build_engine(onnx_path, engine_path, profile, tf32=tf32, n_channels=n_channels)


def inspect_layers(engine) -> list:
    """Per-layer information of a built engine (needs DETAILED profiling
    verbosity at build time): name, type, tactic and tensor dtypes."""
    trt = _trt()
    info = json.loads(engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
    layers = []
    for layer in info["Layers"]:
        if not isinstance(layer, dict):  # verbosity too low: only names
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


# Kernel names spell out their dtypes, e.g. ..._i8i8_i8i32_f32_... =
# activation in i8, out i8 | weights i8, accumulator i32 | epilogue f32.
_DT = r"i8|i4|f16|bf16|tf32|f32|e4m3|e5m2|e2m1|fp8|fp4"
_KERNEL_DTYPES = re.compile(rf"_(?P<input>{_DT})(?P<output>{_DT})_(?P<weight>{_DT})_?(?P<accumulator>i32|f32|f16)_")


def kernel_dtypes(tactic: str) -> dict | None:
    """{input, output, weight, accumulator} parsed from a tactic name, or None."""
    match = _KERNEL_DTYPES.search(tactic or "")
    return match.groupdict() if match else None


def summarize_layers(layers: list) -> dict:
    """What precision did the convolutions actually run in?

    compute_layers  conv/deconv layers by (input dtype -> output dtype)
                    as reported by the engine inspector;
    weights /       conv/deconv layers by weight dtype and accumulator dtype
    accumulators    parsed from the kernel names ("unparsed" if unknown);
    tf32_tactics    kernels whose name says TF32 (0 when TF32 is off).
    """
    compute = [layer for layer in layers if layer.get("type") in COMPUTE_LAYER_TYPES]
    by_dtype = collections.Counter(
        f"{layer['type']}:{(layer['inputs'] or ['?'])[0]}->{(layer['outputs'] or ['?'])[0]}" for layer in compute
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
    """Callable like a PyTorch model: FP32 CUDA images [1, C, H, W] -> logits.

    The execution context uses USER_MANAGED device memory taken from a
    PyTorch tensor, so torch.cuda.max_memory_allocated() includes TensorRT's
    activation memory. The engine weights themselves are allocated by
    TensorRT and are reported separately (engine size).
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
        self.workspace = None  # grown on demand, never shrunk
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
            self.workspace = None  # free the old buffer before allocating
            self.workspace = torch.empty(max(needed, 1), dtype=torch.uint8, device=images.device)
        self.context.set_device_memory(self.workspace.data_ptr(), self.workspace.numel())

        logits = torch.empty(tuple(self.context.get_tensor_shape(self.output_name)), dtype=torch.float32, device=images.device)
        self.context.set_tensor_address(self.input_name, images.data_ptr())
        self.context.set_tensor_address(self.output_name, logits.data_ptr())
        # Order the engine after the work that produced `images`, and later
        # work after the engine, without a full device synchronization.
        current = torch.cuda.current_stream(images.device)
        self.stream.wait_stream(current)
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed.")
        current.wait_stream(self.stream)
        for tensor in (images, logits, self.workspace):
            tensor.record_stream(self.stream)  # not reused while the engine may still read them
        return logits

    def layer_summary(self) -> dict:
        # Recomputed from the saved per-layer file, so engines cached before
        # a summary field was added still report it.
        layers_path = self.engine_path.with_suffix(".layers.json")
        return summarize_layers(json.loads(layers_path.read_text(encoding="utf-8")))


def profile_from_sizes(sizes) -> dict:
    """Optimization profile covering every (H, W) in `sizes`; kernels are
    tuned for the most frequent size."""
    sizes = list(sizes)
    heights = [h for h, _ in sizes]
    widths = [w for _, w in sizes]
    opt = collections.Counter(sizes).most_common(1)[0][0]
    return {"min": [min(heights), min(widths)], "opt": list(opt), "max": [max(heights), max(widths)]}
