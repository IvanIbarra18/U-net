"""Make charts from the test.py results (summary.csv and per_sample/*.csv).

Each chart compares the FP32 baseline with the other configurations: the
configurations are on the x axis, and the metric is on the y axis. The
PyTorch runs (baselines and fake quantization) and the TensorRT runs get
separate charts, because their latency and memory are not comparable.

Charts made for each dataset and backend:
    <backend>_mean_dice.png       mean Dice, with the FP32 baseline as a dashed line
    <backend>_dice_per_image.png  Dice change per image vs the FP32 baseline
                                  (box plot, so you can see the worst images)
    <backend>_latency.png         forward latency per image
    <backend>_peak_memory.png     peak GPU memory allocated
    <backend>_model_size.png      weights (PyTorch) or engine file (TensorRT) size

Usage (from the repository root):
    python scripts/plot_results.py --results test_results/rtx5080_carvana
    python scripts/plot_results.py --results test_results/rtx5080_carvana --output figures/carvana

If a configuration was run more than once, the last row in summary.csv is used.
"""

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # we only save files, so no window is needed
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# One color per format family. The colors follow a palette that stays
# readable for colorblind readers; the order never changes between charts.
FAMILY_COLORS = {
    "Baseline": "#8c8b86",
    "INT8": "#2a78d6",
    "FP8": "#eb6834",
    "FP6": "#1baf7a",
    "INT4": "#eda100",
    "FP4": "#e87ba4",  # FP4, NVFP4 and MXFP4
}
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"

CALIB_SHORT = {
    "max": "max",
    "percentile-99.9": "p99.9",
    "percentile-99.99": "p99.99",
    "percentile-99.999": "p99.999",
    "entropy": "entropy",
    "entropy-skip-zeros": "entropy (skip 0)",
    "dynamic": "dynamic",
}
BLOCK_PRESETS = ("NVFP4", "MXFP4")  # their block size is part of the name


# ---------------------------------------------------------------------------
# Reading the results
# ---------------------------------------------------------------------------

def number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def is_baseline(row):
    """True for the true FP32 run: no quantization, no cast, no autocast, no TF32."""
    return (row["quantization"] == "none" and row["model_dtype"] == "FP32"
            and row["autocast"] == "off" and row["tf32"] == "off")


def family(row):
    if row["quantization"] == "none":
        return "Baseline"
    name = row["quantization"].split("-")[0]
    return "FP4" if name in BLOCK_PRESETS else name


def block_size(row):
    """Return the block size written in the representation, or None."""
    match = re.search(r"blocks of (\d+)", row["representation"])
    return int(match.group(1)) if match else None


def label(row):
    """A short x-axis label, for example 'INT8 W8A8\\np99.99\\n1st conv FP32'."""
    quant = row["quantization"]
    if quant == "none":
        if row["model_dtype"] != "FP32":
            return f"{row['model_dtype']}\ncast"
        if row["autocast"] != "off":
            return f"{row['autocast']}\nautocast"
        return "TF32" if row["tf32"] == "on" else "FP32"

    lines = [quant.replace("-", " ")]
    size = block_size(row)
    if size and quant.split("-")[0] not in BLOCK_PRESETS:
        lines[0] += f" b{size}"
    if row["act_calibration"]:
        lines.append(CALIB_SHORT.get(row["act_calibration"], row["act_calibration"]))
    if row["fp_layers"] == "enc1.block.0":
        lines.append("1st conv FP32")
    elif row["fp_layers"]:
        lines.append(f"FP32: {row['fp_layers']}")
    return "\n".join(lines)


def per_sample_path(row, results_dir):
    """Rebuild the per_sample file name that test.py wrote for this row
    (see utilis.reporting.per_sample_filename)."""
    quant = row["quantization"]
    cast = row["model_dtype"] != "FP32"
    precision = row["model_dtype"] if cast else ("FP32" if row["autocast"] == "off" else row["autocast"])
    tensorrt = row["backend"].startswith("tensorrt")
    stem = Path(row["checkpoint"].replace("\\", "/")).stem  # the path may come from Windows or Linux

    parts = [
        f"{stem}_{row['dataset']}_{row['split']}_{row['model']}_{precision}",
        "_cast" if cast else "",
        "_trt" if tensorrt else "",
        "_d2s" if "depth2space" in row["backend"] else "",
        "_tf32" if row["tf32"] == "on" else "",
        "" if quant == "none" else f"_{quant}",
        f"_{row['act_calibration']}" if row["act_calibration"] and row["act_calibration"] != "dynamic" else "",
    ]
    if quant != "none":
        size = block_size(row)
        if size and quant.split("-")[0] not in BLOCK_PRESETS:
            scale = re.search(r"blocks of \d+, (\w+) scales", row["representation"])
            parts.append(f"_b{size}" + (f"-{scale.group(1)}" if scale else ""))
        if row["act_calibration"] == "dynamic":
            parts.append("_dyn")
        weight_format = row["representation"].split()[0]
        act = re.search(r", (\S+) [^,]*activations", row["representation"])
        if act and act.group(1) not in ("fp32", weight_format):
            parts.append(f"_a-{act.group(1).split('_')[0].upper()}")
    if row["fp_layers"]:
        parts.append(f"_fp-{row['fp_layers'].replace(',', '+')}")
    return results_dir / "per_sample" / ("".join(parts) + ".csv")


def read_per_sample(path):
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as file:
        return {r["sample_id"]: float(r["dice"]) for r in csv.DictReader(file)}


def load_runs(results_dir):
    """Return {(dataset, backend group): [run, ...]} in the order of summary.csv."""
    with (results_dir / "summary.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    groups = {}
    for row in rows:
        group = "tensorrt" if row["backend"].startswith("tensorrt") else "pytorch"
        runs = groups.setdefault((row["dataset"], group), {})
        run = {
            "label": label(row),
            "family": family(row),
            "baseline": is_baseline(row),
            "mean_dice": number(row["mean_dice"]),
            "latency": number(row["latency_ms_per_image"]),
            "memory": number(row["peak_gpu_allocated_mib"]),
            "size": number(row["engine_mib"] if group == "tensorrt" else row["weights_mib"]),
            "per_sample_path": per_sample_path(row, results_dir),
        }
        runs.pop(run["label"], None)  # a rerun replaces the older row, and moves to the end
        runs[run["label"]] = run
    return {key: list(runs.values()) for key, runs in groups.items()}


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def new_axes(n_configs, title, ylabel):
    fig, ax = plt.subplots(figsize=(max(7.0, 0.85 * n_configs + 1.5), 5.2))
    ax.set_title(title, loc="left", fontsize=12, color=TEXT, pad=26)  # room for the legend under it
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    return fig, ax


def finish(fig, ax, runs, path, note=None):
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([r["label"] for r in runs], fontsize=7.5, color=TEXT)
    ax.set_xlim(-0.6, len(runs) - 0.4)
    families = [f for f in FAMILY_COLORS if any(r["family"] == f for r in runs)]
    handles = [Patch(color=FAMILY_COLORS[f], label="FP4 / NVFP4 / MXFP4" if f == "FP4" else f) for f in families]
    handles += [line for line in ax.get_lines() if not line.get_label().startswith("_")]  # the baseline line
    if handles:
        ax.legend(handles=handles, frameon=False, fontsize=8, ncol=len(handles),
                  loc="lower left", bbox_to_anchor=(0, 1.01), borderaxespad=0, handlelength=1.2)
    if note:
        fig.text(0.01, 0.01, note, fontsize=7.5, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04 if note else 0, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  {path}")


def baseline_line(ax, value, text):
    """Draw the baseline as a dashed line. Its label goes in the legend, so it
    never covers the value labels of the bars."""
    ax.axhline(value, color=TEXT, linewidth=1, linestyle="--", zorder=1, label=text)


def bar_chart(runs, key, baseline, title, ylabel, path, unit, note=None):
    runs = [r for r in runs if r[key] is not None]
    if not runs:
        return
    fig, ax = new_axes(len(runs), title, ylabel)
    values = [r[key] for r in runs]
    ax.bar(range(len(runs)), values, width=0.7, color=[FAMILY_COLORS[r["family"]] for r in runs],
           edgecolor="white", linewidth=1, zorder=2)
    for i, value in enumerate(values):
        ax.annotate(f"{value:.0f}" if value >= 100 else f"{value:.1f}", xy=(i, value), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7, color=MUTED)
    if baseline is not None and baseline.get(key) is not None:
        baseline_line(ax, baseline[key], f"FP32 baseline {baseline[key]:.1f} {unit}")
    ax.set_ylim(0, max(values) * 1.12)
    finish(fig, ax, runs, path, note)


def dice_chart(runs, baseline, title, path):
    runs = [r for r in runs if r["mean_dice"] is not None]
    fig, ax = new_axes(len(runs), title, "Mean Dice")
    values = [r["mean_dice"] for r in runs]
    ax.scatter(range(len(runs)), values, s=60, color=[FAMILY_COLORS[r["family"]] for r in runs],
               edgecolor="white", linewidth=1.5, zorder=3)
    for i, value in enumerate(values):
        ax.annotate(f"{value:.4f}", xy=(i, value), xytext=(0, 7), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7, color=MUTED)
    if baseline is not None:
        baseline_line(ax, baseline["mean_dice"], f"FP32 baseline {baseline['mean_dice']:.4f}")
    low, high = min(values), max(values)
    pad = max(high - low, 1e-4) * 0.15
    ax.set_ylim(low - pad, high + 2 * pad)
    finish(fig, ax, runs, path, "The y axis is zoomed in: it does not start at 0.")


def per_image_chart(runs, baseline, title, path):
    if baseline is None:
        return
    reference = read_per_sample(baseline["per_sample_path"])
    if reference is None:
        print(f"  (no per-image file for the baseline: {baseline['per_sample_path'].name})")
        return
    data, kept = [], []
    for run in runs:
        if run is baseline:
            continue
        scores = read_per_sample(run["per_sample_path"])
        if scores is None:
            print(f"  (no per-image file for {run['label']!r}: {run['per_sample_path'].name})")
            continue
        common = [s for s in reference if s in scores]
        data.append([scores[s] - reference[s] for s in common])
        kept.append(run)
    if not kept:
        return

    fig, ax = new_axes(len(kept), title, "Dice change per image (config - FP32)")
    boxes = ax.boxplot(data, positions=range(len(kept)), widths=0.6, patch_artist=True,
                       medianprops={"color": TEXT, "linewidth": 1},
                       whiskerprops={"color": MUTED}, capprops={"color": MUTED},
                       flierprops={"marker": "o", "markersize": 3, "markerfacecolor": MUTED,
                                   "markeredgecolor": "none", "alpha": 0.6})
    for box, run in zip(boxes["boxes"], kept):
        box.set_facecolor(FAMILY_COLORS[run["family"]])
        box.set_edgecolor("white")
    for i, deltas in enumerate(data):
        worst = min(deltas)
        ax.annotate(f"{worst:+.3f}", xy=(i, worst), xytext=(0, -9), textcoords="offset points",
                    ha="center", va="top", fontsize=7, color=MUTED)
    ax.axhline(0, color=TEXT, linewidth=1, linestyle="--", zorder=1)
    low = min(min(d) for d in data)
    high = max(max(d) for d in data)
    pad = max(high - low, 1e-3) * 0.08
    ax.set_ylim(low - 2 * pad, high + pad)
    finish(fig, ax, kept, path, "Each box: the images of the validation split. The number under a box is the worst image.")


def make_charts(runs, dataset, group, output_dir):
    backend = "TensorRT" if group == "tensorrt" else "PyTorch"
    baseline = next((r for r in runs if r["baseline"]), None)
    if baseline is None:
        print(f"  (no FP32 baseline for {dataset} {backend}: charts have no reference line)")
    prefix = output_dir / f"{dataset}_{group}"
    what = "TensorRT engines" if group == "tensorrt" else "baselines and fake quantization"

    dice_chart(runs, baseline, f"{dataset}, {backend}: mean Dice ({what})", Path(f"{prefix}_mean_dice.png"))
    per_image_chart(runs, baseline, f"{dataset}, {backend}: Dice change per image vs FP32",
                    Path(f"{prefix}_dice_per_image.png"))
    bar_chart(runs, "latency", baseline, f"{dataset}, {backend}: forward latency per image",
              "Latency (ms / image)", Path(f"{prefix}_latency.png"), "ms")
    bar_chart(runs, "memory", baseline, f"{dataset}, {backend}: peak GPU memory allocated",
              "Peak GPU memory allocated (MiB)", Path(f"{prefix}_peak_memory.png"), "MiB")
    if group == "tensorrt":
        bar_chart(runs, "size", baseline, f"{dataset}, TensorRT: engine file size", "Engine size (MiB)",
                  Path(f"{prefix}_model_size.png"), "MiB")
    else:
        bar_chart(runs, "size", baseline, f"{dataset}, PyTorch: size of the weights in memory", "Weights (MiB)",
                  Path(f"{prefix}_model_size.png"), "MiB",
                  note="Fake quantization keeps the weights in FP32, so the size does not change. "
                       "Real storage savings are in the TensorRT charts.")


def main():
    parser = argparse.ArgumentParser(description="Make charts from test.py results.")
    parser.add_argument("--results", type=Path, required=True, help="Folder with summary.csv and per_sample/.")
    parser.add_argument("--output", type=Path, default=None, help="Default: <results>/plots")
    args = parser.parse_args()

    output_dir = args.output or args.results / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    for (dataset, group), runs in load_runs(args.results).items():
        print(f"{dataset} / {group}: {len(runs)} configurations")
        make_charts(runs, dataset, group, output_dir)


if __name__ == "__main__":
    main()
