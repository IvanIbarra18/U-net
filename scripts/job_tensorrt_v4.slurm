#!/bin/bash
#SBATCH --job-name=unet_precision_v4
#SBATCH --partition=gpul40q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=precision-v4-%j.out
#SBATCH --error=precision-v4-%j.err
#
# Precision evaluation of the three FP32 baselines (Carvana, ISBI, ISIC) on the L40S.
#
# For each dataset:
#   1. ONNX export (standard graph + depth2space graph for FP8).
#   2. PyTorch runs: FP32/TF32/FP16/BF16 baselines and fake quantization
#      (simulated INT8, FP8, FP6, INT4, FP4, NVFP4, MXFP4).
#   3. TensorRT runs: real FP32, INT8, FP8, INT4 and FP4 engines.
# Every quantized configuration runs twice: with the first convolution
# (enc1.block.0) kept in FP32, and with every layer quantized.
#
# It does NOT train or calibrate: it reuses the existing checkpoints and
# calibration files, and stops if any of them is missing.
#
# Needs PyTorch >= 2.11 in the environment (PyTorch 2.5.1 cannot export the
# U-Net to ONNX with dynamic height and width).
#
# Submit from the repository root, after git pull:
#     sbatch scripts/job_tensorrt_v4.slurm

set -Eeuo pipefail

source ~/myEnvs/unet_uv/bin/activate
cd "$SLURM_SUBMIT_DIR"

JOB_OUTPUT="$PWD/outputs/PrecisionEvaluation_v4_${SLURM_JOB_ID}"
RESULTS="$JOB_OUTPUT/results"
mkdir -p "$JOB_OUTPUT" "$RESULTS"

DATASETS=(Carvana ISBI ISIC)   # ISIC last: its image sizes have not been tested with TensorRT yet

echo "================================================"
echo "U-Net Precision Evaluation V4"
echo "Job: $SLURM_JOB_ID    Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "Output: $JOB_OUTPUT"
echo "Commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "================================================"

# ============================================================
# 1. PREFLIGHT: environment, checkpoints, calibration files
# ============================================================

python -m quantization.env_report | tee "$JOB_OUTPUT/environment.json"
python - <<'PY'
import torch, tensorrt as trt
major, minor = (int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
assert (major, minor) >= (2, 11), f"PyTorch {torch.__version__}: the ONNX export needs >= 2.11"
assert torch.cuda.is_available(), "CUDA is unavailable"
assert trt.Builder(trt.Logger(trt.Logger.WARNING)) is not None, "TensorRT Builder failed"
print("PyTorch:", torch.__version__, "| GPU:", torch.cuda.get_device_name(0), "| TensorRT:", trt.__version__)
PY
nvidia-smi | tee "$JOB_OUTPUT/nvidia-smi.txt"

for dataset in "${DATASETS[@]}"; do
    for file in "checkpoints/best_${dataset}-Baseline.pth" "calibration/best_${dataset}-Baseline_n128.json"; do
        if [[ ! -s "$file" ]]; then
            echo "ERROR: missing $file (this job does not train or calibrate)."
            exit 1
        fi
        ls -lh "$file"
    done
done

# ============================================================
# 2. EXPERIMENT RUNNER: one log and one GPU metrics CSV per step.
# A failed step is recorded and the job moves on.
# ============================================================

FAILED=()
TOTAL=0

run_experiment() {
    local name="$1"
    shift
    TOTAL=$((TOTAL + 1))
    local metrics_file="$JOB_OUTPUT/${name}_gpu_metrics.csv"
    local log_file="$JOB_OUTPUT/${name}.log"

    echo ""
    echo "================================================"
    echo "Starting: $name    ($(date))"
    echo "Command:  $*"
    echo "================================================"

    nvidia-smi \
        --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.gr,clocks.mem \
        --format=csv --loop=1 > "$metrics_file" &
    local monitor_pid=$!

    set +e
    "$@" 2>&1 | tee "$log_file"
    local exit_code=${PIPESTATUS[0]}
    set -e

    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true

    echo "Finished: $name    exit code $exit_code    ($(date))"
    if [[ "$exit_code" -ne 0 ]]; then
        FAILED+=("$name")
    fi
    return 0
}

# ============================================================
# 3. UNIT TESTS (mandatory: nothing else runs if they fail)
# ============================================================

run_experiment unit_tests python -m unittest discover -s tests -t .
if [[ "${#FAILED[@]}" -ne 0 ]]; then
    echo "ERROR: unit tests failed. No experiments were started."
    exit 1
fi

# ============================================================
# 4. ONE DATASET: export, fake quantization, TensorRT
# ============================================================

evaluate_dataset() {
    local dataset="$1"
    local ckpt="checkpoints/best_${dataset}-Baseline.pth"
    local calib=(--calib-file "calibration/best_${dataset}-Baseline_n128.json")
    local first=(--fp-layers enc1.block.0)
    local sim=(python test.py --dataset "$dataset" --checkpoint "$ckpt" --output "$RESULTS/$dataset/pytorch")
    local trt=(python test.py --dataset "$dataset" --checkpoint "$ckpt" --output "$RESULTS/$dataset/tensorrt" --backend tensorrt)
    local p="$dataset"

    echo ""
    echo "################ DATASET: $dataset ################"

    # ---- 4.1 ONNX export: standard graph, and depth2space graph for FP8 ----
    run_experiment "${p}_onnx"     python -m quantization.export_onnx --dataset "$dataset" --checkpoint "$ckpt"
    run_experiment "${p}_onnx_d2s" python -m quantization.export_onnx --dataset "$dataset" --checkpoint "$ckpt" --upsample depth2space

    # ---- 4.2 PyTorch baselines ----
    run_experiment "${p}_base_fp32"      "${sim[@]}"
    run_experiment "${p}_base_tf32"      "${sim[@]}" --tf32 on
    run_experiment "${p}_base_cast_fp16" "${sim[@]}" --cast FP16
    run_experiment "${p}_base_cast_bf16" "${sim[@]}" --cast BF16

    # ---- 4.3 Fake quantization: INT8 ----
    run_experiment "${p}_sim_int8_w8a32_fp1" "${sim[@]}" --quant INT8 "${first[@]}"
    run_experiment "${p}_sim_int8_w8a32_all" "${sim[@]}" --quant INT8
    # Calibration sweep (every layer quantized), plus the TensorRT setting with the first conv in FP32.
    for method in max percentile-99.9 percentile-99.99 percentile-99.999 entropy-skip-zeros; do
        run_experiment "${p}_sim_int8_w8a8_${method}_all" "${sim[@]}" --quant INT8 --act-calib "$method" "${calib[@]}"
    done
    run_experiment "${p}_sim_int8_w8a8_percentile-99.99_fp1" "${sim[@]}" --quant INT8 --act-calib percentile-99.99 "${calib[@]}" "${first[@]}"

    # ---- 4.4 Fake quantization: other formats (weights only, plus W4A8) ----
    # FP8/FP6/FP4 activations are not simulated: their simulation needs several
    # full-size copies of each activation map.
    for format in FP8 FP6 INT4 FP4 NVFP4 MXFP4; do
        local name="${format,,}"
        run_experiment "${p}_sim_${name}_weights_fp1" "${sim[@]}" --quant "$format" "${first[@]}"
        run_experiment "${p}_sim_${name}_weights_all" "${sim[@]}" --quant "$format"
    done
    run_experiment "${p}_sim_int4_b32_fp1" "${sim[@]}" --quant INT4 --block-size 32 "${first[@]}"
    run_experiment "${p}_sim_int4_b32_all" "${sim[@]}" --quant INT4 --block-size 32
    run_experiment "${p}_sim_int4_w4a8_fp1" "${sim[@]}" --quant INT4 --act-format INT8 --act-calib percentile-99.99 "${calib[@]}" "${first[@]}"
    run_experiment "${p}_sim_int4_w4a8_all" "${sim[@]}" --quant INT4 --act-format INT8 --act-calib percentile-99.99 "${calib[@]}"

    # ---- 4.5 TensorRT (real low-precision engines) ----
    run_experiment "${p}_trt_fp32" "${trt[@]}"
    # W8A8: real INT8 / FP8 weights and activations.
    run_experiment "${p}_trt_int8_w8a8_fp1" "${trt[@]}" --quant INT8 --act-calib percentile-99.99 "${calib[@]}" "${first[@]}"
    run_experiment "${p}_trt_int8_w8a8_all" "${trt[@]}" --quant INT8 --act-calib percentile-99.99 "${calib[@]}"
    run_experiment "${p}_trt_fp8_w8a8_fp1"  "${trt[@]}" --quant FP8 --act-calib percentile-99.99 "${calib[@]}" "${first[@]}"
    run_experiment "${p}_trt_fp8_w8a8_all"  "${trt[@]}" --quant FP8 --act-calib percentile-99.99 "${calib[@]}"
    # Weight-only 4-bit: compressed storage, FP32 convolutions.
    run_experiment "${p}_trt_int4_weights_fp1" "${trt[@]}" --quant INT4 "${first[@]}"
    run_experiment "${p}_trt_int4_weights_all" "${trt[@]}" --quant INT4
    run_experiment "${p}_trt_fp4_weights_fp1"  "${trt[@]}" --quant FP4 "${first[@]}"
    run_experiment "${p}_trt_fp4_weights_all"  "${trt[@]}" --quant FP4
}

for dataset in "${DATASETS[@]}"; do
    evaluate_dataset "$dataset"
done

# ============================================================
# 5. SUMMARY
# ============================================================

echo ""
echo "================================================"
echo "PRECISION EVALUATION V4 FINISHED: $(date)"
echo "Executed steps: $TOTAL    Failed: ${#FAILED[@]}"
echo "Logs and GPU metrics: $JOB_OUTPUT"
echo "Results: $RESULTS/<dataset>/{pytorch,tensorrt}/summary.csv"
echo "================================================"
if [[ "${#FAILED[@]}" -ne 0 ]]; then
    echo "WARNING: failed steps: ${FAILED[*]}"
    exit 1
fi
echo "Every step finished without errors."
