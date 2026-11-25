#!/bin/bash
# Full Carvana experiment list on the local RTX 5080 (no SLURM needed).
#
# It runs the same steps as the cluster job, only for Carvana: training,
# baselines, calibration, ONNX export, simulated quantization and TensorRT.
# Every step gets its own log and its own GPU monitor (nvidia-smi, one row per
# second), like the cluster jobs. A failed step is recorded and the script
# moves on to the next one.
#
# Run it from Git Bash in the repository root:
#     bash scripts/run_carvana_rtx5080.sh
#
# Before you start: close other programs that use the GPU (the latencies are
# measured), and turn off Windows sleep, or the run stops when the PC sleeps.

set -uo pipefail

cd "$(dirname "$0")/.."                # repository root
PY="venv/Scripts/python.exe"           # the venv has the CUDA 12.8 PyTorch build that supports the RTX 5080
export PYTHONIOENCODING=utf-8          # torch.onnx prints emoji; the Windows console cannot show them otherwise

NAME="Carvana-Baseline-5080"           # experiment name, so it never mixes with the cluster checkpoints
CKPT="checkpoints/best_${NAME}.pth"
CALIB="calibration/best_${NAME}_n128.json"
RESULTS="test_results/rtx5080_carvana"
JOB_OUTPUT="$PWD/outputs/Carvana_RTX5080_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$JOB_OUTPUT"

echo "================================================"
echo "U-Net quantization experiments: Carvana on RTX 5080"
echo "Start:   $(date)"
echo "Output:  $JOB_OUTPUT"
echo "Results: $RESULTS"
echo "================================================"
"$PY" -m quantization.env_report | tee "$JOB_OUTPUT/environment.json"
nvidia-smi | tee "$JOB_OUTPUT/nvidia-smi.txt"

MONITOR_PID=""
FAILED_STEPS=()

stop_monitor() {
    if [ -n "$MONITOR_PID" ]; then
        kill "$MONITOR_PID" 2>/dev/null || true
        wait "$MONITOR_PID" 2>/dev/null || true
        MONITOR_PID=""
    fi
}
trap stop_monitor EXIT INT TERM        # also stop the monitor if you press Ctrl+C

# run_step NAME COMMAND...  runs one command with its own log and GPU monitor.
run_step() {
    local name="$1"
    shift
    local log_file="$JOB_OUTPUT/${name}.log"
    local metrics_file="$JOB_OUTPUT/${name}_gpu_metrics.csv"

    echo ""
    echo "================================================"
    echo "Starting: $name    ($(date))"
    echo "Command:  $*"
    echo "================================================"

    nvidia-smi \
        --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.gr,clocks.mem \
        --format=csv --loop=1 > "$metrics_file" &
    MONITOR_PID=$!

    "$@" 2>&1 | tee "$log_file"
    local exit_code=${PIPESTATUS[0]}

    stop_monitor
    echo "Finished: $name    exit code $exit_code    ($(date))"
    if [ "$exit_code" -ne 0 ]; then
        FAILED_STEPS+=("$name")
    fi
}

TEST=("$PY" test.py --dataset Carvana --checkpoint "$CKPT" --output "$RESULTS")
CAL=(--calib-file "$CALIB")
FIRST=(--fp-layers enc1.block.0)

# ---------------- 0. Unit tests ----------------
run_step unit_tests "$PY" -m unittest discover -s tests -t .

# ---------------- 1. Training (true FP32: TF32 is off by default) ----------------
run_step train "$PY" train.py --dataset Carvana --epochs 100 --batch-size 5 --experiment-name "$NAME"

# ---------------- 2. Baselines (PyTorch, validation split) ----------------
run_step base_fp32      "${TEST[@]}"
run_step base_tf32      "${TEST[@]}" --tf32 on
run_step base_cast_fp16 "${TEST[@]}" --cast FP16
run_step base_cast_bf16 "${TEST[@]}" --cast BF16

# ---------------- 3. Calibration and ONNX export ----------------
run_step calibrate       "$PY" -m quantization.calibrate --dataset Carvana --checkpoint "$CKPT" --samples 128
run_step export_onnx     "$PY" -m quantization.export_onnx --dataset Carvana --checkpoint "$CKPT"
run_step export_onnx_d2s "$PY" -m quantization.export_onnx --dataset Carvana --checkpoint "$CKPT" --upsample depth2space

# ---------------- 4. Simulation: INT8 and the calibration sweep ----------------
run_step sim_int8_w8a32 "${TEST[@]}" --quant INT8
for method in max percentile-99.9 percentile-99.99 percentile-99.999 entropy-skip-zeros; do
    run_step "sim_int8_w8a8_${method}" "${TEST[@]}" --quant INT8 --act-calib "$method" "${CAL[@]}"
done

# ---------------- 5. Simulation: other formats (first conv in FP32) ----------------
# Weight-only, plus W4A8 (INT8 activations are simulated in place, no extra memory).
# FP8/FP4 activations are not simulated here: they do not fit in 16 GB at full resolution.
run_step sim_fp8_w8a32   "${TEST[@]}" --quant FP8 "${FIRST[@]}"
run_step sim_fp6_w6a32   "${TEST[@]}" --quant FP6 "${FIRST[@]}"
run_step sim_int4_w4a32  "${TEST[@]}" --quant INT4 "${FIRST[@]}"
run_step sim_int4_b32    "${TEST[@]}" --quant INT4 --block-size 32 "${FIRST[@]}"
run_step sim_int4_w4a8   "${TEST[@]}" --quant INT4 --act-format INT8 --act-calib percentile-99.99 "${CAL[@]}" "${FIRST[@]}"
run_step sim_fp4_w4a32   "${TEST[@]}" --quant FP4 "${FIRST[@]}"
run_step sim_nvfp4_w4a32 "${TEST[@]}" --quant NVFP4 "${FIRST[@]}"
run_step sim_mxfp4_w4a32 "${TEST[@]}" --quant MXFP4 "${FIRST[@]}"

# ---------------- 6. TensorRT (real) ----------------
run_step trt_fp32          "${TEST[@]}" --backend tensorrt
run_step trt_int8_w8a8     "${TEST[@]}" --backend tensorrt --quant INT8 --act-calib percentile-99.99 "${CAL[@]}"
run_step trt_int8_w8a8_fp1 "${TEST[@]}" --backend tensorrt --quant INT8 --act-calib percentile-99.99 "${CAL[@]}" "${FIRST[@]}"
run_step trt_fp8_w8a8      "${TEST[@]}" --backend tensorrt --quant FP8 --act-calib percentile-99.99 "${CAL[@]}" "${FIRST[@]}"
run_step trt_int4_w4a32    "${TEST[@]}" --backend tensorrt --quant INT4 "${FIRST[@]}"
run_step trt_fp4_w4a32     "${TEST[@]}" --backend tensorrt --quant FP4 "${FIRST[@]}"

echo ""
echo "================================================"
echo "ALL STEPS FINISHED: $(date)"
echo "Logs and GPU metrics: $JOB_OUTPUT"
echo "Results (summary.csv, per_sample/): $RESULTS"
if [ "${#FAILED_STEPS[@]}" -ne 0 ]; then
    echo "WARNING: these steps failed: ${FAILED_STEPS[*]}"
    exit 1
fi
echo "Every step finished without errors."
echo "================================================"
