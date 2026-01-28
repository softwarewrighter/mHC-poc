#!/usr/bin/env python3
"""Generate comparison plots for CUDA depth sweep results."""
from __future__ import annotations
import os
import json
import glob
import matplotlib.pyplot as plt


def read_metrics(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def find_cuda_runs(runs_dir: str = "runs"):
    """Find CUDA depth sweep runs and organize by depth and variant."""
    runs = {}

    # Look for cuda_sweep_* directories
    for sweep_dir in glob.glob(os.path.join(runs_dir, "cuda_sweep_*")):
        if not os.path.isdir(sweep_dir):
            continue

        # Look for subdirectories like 12l_baseline, 24l_mhc, etc.
        for run_dir in glob.glob(os.path.join(sweep_dir, "*")):
            if not os.path.isdir(run_dir):
                continue

            name = os.path.basename(run_dir)
            # Parse: 12l_baseline, 24l_hc, 48l_mhc
            parts = name.split("_")
            if len(parts) >= 2:
                depth = parts[0]  # 12l, 24l, 48l
                variant = parts[1]  # baseline, hc, mhc
                key = (depth, variant)
                mpath = os.path.join(run_dir, "metrics.jsonl")
                if os.path.exists(mpath):
                    runs[key] = read_metrics(mpath)

    return runs


def plot_loss_comparison(runs: dict, out_dir: str):
    """Plot loss curves for all variants at each depth."""
    depths = ["12l", "24l", "48l"]
    variants = ["baseline", "hc", "mhc"]
    colors = {"baseline": "gray", "hc": "red", "mhc": "blue"}
    labels = {"baseline": "Baseline", "hc": "HC (unconstrained)", "mhc": "mHC (constrained)"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("CUDA: Loss Comparison Across Depths", fontsize=14, y=1.02)

    for i, depth in enumerate(depths):
        ax = axes[i]
        for variant in variants:
            key = (depth, variant)
            if key in runs:
                data = runs[key]
                steps = [r["step"] for r in data]
                loss = [r["loss"] for r in data]
                ax.plot(steps, loss, color=colors[variant], label=labels[variant], linewidth=1.5)

        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title(f"{depth.upper()}")
        ax.legend(fontsize=9)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cuda_loss_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_gain_comparison(runs: dict, out_dir: str):
    """Plot gain proxy for HC vs mHC at each depth."""
    depths = ["12l", "24l", "48l"]
    colors = {"hc": "red", "mhc": "blue"}
    labels = {"hc": "HC (unconstrained)", "mhc": "mHC (constrained)"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("CUDA: Gain Proxy Comparison (HC vs mHC)", fontsize=14, y=1.02)

    for i, depth in enumerate(depths):
        ax = axes[i]
        for variant in ["hc", "mhc"]:
            key = (depth, variant)
            if key in runs:
                data = runs[key]
                steps = [r["step"] for r in data]
                gain = [r["gain_proxy_log10_max_entry"] for r in data]
                ax.plot(steps, gain, color=colors[variant], label=labels[variant], linewidth=1.5)

        ax.set_xlabel("Step")
        ax.set_ylabel("Gain Proxy (log10)")
        ax.set_title(f"{depth.upper()}")
        ax.legend(fontsize=9)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cuda_gain_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_depth_scaling(runs: dict, out_dir: str):
    """Plot how final gain scales with depth for HC vs mHC."""
    depths = ["12l", "24l", "48l"]
    depth_nums = [12, 24, 48]

    hc_gains = []
    mhc_gains = []

    for depth in depths:
        hc_key = (depth, "hc")
        mhc_key = (depth, "mhc")

        if hc_key in runs:
            hc_gains.append(runs[hc_key][-1]["gain_proxy_log10_max_entry"])
        else:
            hc_gains.append(None)
        if mhc_key in runs:
            mhc_gains.append(runs[mhc_key][-1]["gain_proxy_log10_max_entry"])
        else:
            mhc_gains.append(None)

    # Filter out None values
    hc_valid = [(d, g) for d, g in zip(depth_nums, hc_gains) if g is not None]
    mhc_valid = [(d, g) for d, g in zip(depth_nums, mhc_gains) if g is not None]

    fig, ax = plt.subplots(figsize=(8, 5))

    if hc_valid:
        hc_d, hc_g = zip(*hc_valid)
        ax.plot(hc_d, hc_g, 'ro-', markersize=10, linewidth=2, label="HC (unconstrained)")
        for d, g in zip(hc_d, hc_g):
            ax.annotate(f'{g:.1f}', (d, g), textcoords="offset points",
                       xytext=(0, 10), ha='center', fontsize=10, color='red')

    if mhc_valid:
        mhc_d, mhc_g = zip(*mhc_valid)
        ax.plot(mhc_d, mhc_g, 'bs-', markersize=10, linewidth=2, label="mHC (constrained)")
        for d, g in zip(mhc_d, mhc_g):
            ax.annotate(f'{g:.1f}', (d, g), textcoords="offset points",
                       xytext=(0, -15), ha='center', fontsize=10, color='blue')

    ax.set_xlabel("Number of Layers", fontsize=12)
    ax.set_ylabel("Final Gain Proxy (log10)", fontsize=12)
    ax.set_title("CUDA: Gain Scaling with Depth", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xticks(depth_nums)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cuda_depth_scaling.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_grad_norm_comparison(runs: dict, out_dir: str):
    """Plot gradient norm for all variants at each depth."""
    depths = ["12l", "24l", "48l"]
    variants = ["baseline", "hc", "mhc"]
    colors = {"baseline": "gray", "hc": "red", "mhc": "blue"}
    labels = {"baseline": "Baseline", "hc": "HC (unconstrained)", "mhc": "mHC (constrained)"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("CUDA: Gradient Norm Comparison", fontsize=14, y=1.02)

    for i, depth in enumerate(depths):
        ax = axes[i]
        for variant in variants:
            key = (depth, variant)
            if key in runs:
                data = runs[key]
                steps = [r["step"] for r in data]
                gnorm = [r["grad_norm"] for r in data]
                ax.plot(steps, gnorm, color=colors[variant], label=labels[variant], linewidth=1.5)

        ax.set_xlabel("Step")
        ax.set_ylabel("Gradient Norm")
        ax.set_title(f"{depth.upper()}")
        ax.legend(fontsize=9)
        ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cuda_grad_norm_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_48l_comparison(runs: dict, out_dir: str):
    """Side-by-side 48L comparison - the critical depth test."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("CUDA: 48-Layer Stress Test (HC vs mHC)", fontsize=14, y=1.02)

    # Loss comparison
    ax = axes[0]
    for variant, color, label in [("hc", "red", "HC"), ("mhc", "blue", "mHC")]:
        key = ("48l", variant)
        if key in runs:
            data = runs[key]
            steps = [r["step"] for r in data]
            loss = [r["loss"] for r in data]
            ax.plot(steps, loss, color=color, label=label, linewidth=2)

    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("Loss: mHC Converges, HC Fails", fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim(bottom=0)

    # Gain comparison
    ax = axes[1]
    for variant, color, label in [("hc", "red", "HC"), ("mhc", "blue", "mHC")]:
        key = ("48l", variant)
        if key in runs:
            data = runs[key]
            steps = [r["step"] for r in data]
            gain = [r["gain_proxy_log10_max_entry"] for r in data]
            ax.plot(steps, gain, color=color, label=label, linewidth=2)

    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Gain Proxy (log10)", fontsize=11)
    ax.set_title("Gain: HC Explodes, mHC Bounded", fontsize=12)
    ax.legend(fontsize=11)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cuda_48l_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    runs = find_cuda_runs("runs")
    if not runs:
        print("No CUDA depth sweep runs found in runs/cuda_sweep_*/")
        print("Run 'bash scripts/run_cuda_depth_sweep.sh' first.")
        return

    out_dir = "cuda/docs/images"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Found {len(runs)} CUDA runs: {list(runs.keys())}")

    plot_loss_comparison(runs, out_dir)
    print("  -> cuda_loss_comparison.png")

    plot_grad_norm_comparison(runs, out_dir)
    print("  -> cuda_grad_norm_comparison.png")

    plot_gain_comparison(runs, out_dir)
    print("  -> cuda_gain_comparison.png")

    plot_depth_scaling(runs, out_dir)
    print("  -> cuda_depth_scaling.png")

    plot_48l_comparison(runs, out_dir)
    print("  -> cuda_48l_comparison.png")

    print(f"\nPlots saved to {out_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
