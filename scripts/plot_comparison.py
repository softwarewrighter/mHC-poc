#!/usr/bin/env python3
"""Generate comparison plots across variants and depths."""
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


def find_runs(runs_dir: str = "runs"):
    """Find all runs and organize by depth and variant."""
    runs = {}
    for d in glob.glob(os.path.join(runs_dir, "*")):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        # Parse: 20260123_091627_baseline_12l
        parts = name.split("_")
        if len(parts) >= 4:
            variant = parts[2]  # baseline, hc, mhc
            depth = parts[3]    # 12l, 24l, 48l
            key = (depth, variant)
            mpath = os.path.join(d, "metrics.jsonl")
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
        ax.set_title(f"{depth.upper()} Loss Comparison")
        ax.legend()
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_comparison.png"), dpi=150)
    plt.close()


def plot_gain_comparison(runs: dict, out_dir: str):
    """Plot gain proxy for HC vs mHC at each depth."""
    depths = ["12l", "24l", "48l"]
    colors = {"hc": "red", "mhc": "blue"}
    labels = {"hc": "HC (unconstrained)", "mhc": "mHC (constrained)"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

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
        ax.set_title(f"{depth.upper()} Gain Proxy Comparison")
        ax.legend()
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gain_comparison.png"), dpi=150)
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
        if mhc_key in runs:
            mhc_gains.append(runs[mhc_key][-1]["gain_proxy_log10_max_entry"])

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(depth_nums, hc_gains, 'ro-', markersize=10, linewidth=2, label="HC (unconstrained)")
    ax.plot(depth_nums, mhc_gains, 'bs-', markersize=10, linewidth=2, label="mHC (constrained)")

    ax.set_xlabel("Number of Layers", fontsize=12)
    ax.set_ylabel("Final Gain Proxy (log10)", fontsize=12)
    ax.set_title("Gain Scaling with Depth: HC Explodes, mHC Stays Bounded", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xticks(depth_nums)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Add value annotations
    for i, (hc, mhc) in enumerate(zip(hc_gains, mhc_gains)):
        ax.annotate(f'{hc:.1f}', (depth_nums[i], hc), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontsize=10, color='red')
        ax.annotate(f'{mhc:.1f}', (depth_nums[i], mhc), textcoords="offset points",
                   xytext=(0, -15), ha='center', fontsize=10, color='blue')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "depth_scaling.png"), dpi=150)
    plt.close()


def plot_grad_norm_comparison(runs: dict, out_dir: str):
    """Plot gradient norm for all variants at each depth."""
    depths = ["12l", "24l", "48l"]
    variants = ["baseline", "hc", "mhc"]
    colors = {"baseline": "gray", "hc": "red", "mhc": "blue"}
    labels = {"baseline": "Baseline (1 residual)", "hc": "HC (unconstrained)", "mhc": "mHC (constrained)"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

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
        ax.set_title(f"{depth.upper()} Gradient Norm Comparison")
        ax.legend()
        ax.set_yscale('log')  # Log scale to see differences

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "grad_norm_comparison.png"), dpi=150)
    plt.close()


def plot_48l_comparison(runs: dict, out_dir: str):
    """Side-by-side 48L comparison - the money shot."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

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
    ax.set_title("48-Layer Loss: mHC Converges, HC Fails", fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim(bottom=0)

    # Gain comparison
    ax = axes[1]
    for variant, color, label in [("hc", "red", "HC (10^27 amplification!)"), ("mhc", "blue", "mHC (bounded)")]:
        key = ("48l", variant)
        if key in runs:
            data = runs[key]
            steps = [r["step"] for r in data]
            gain = [r["gain_proxy_log10_max_entry"] for r in data]
            ax.plot(steps, gain, color=color, label=label, linewidth=2)

    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Gain Proxy (log10)", fontsize=11)
    ax.set_title("48-Layer Gain: HC Amplifies, mHC Constrains", fontsize=12)
    ax.legend(fontsize=11)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "48l_comparison.png"), dpi=150)
    plt.close()


def main():
    runs = find_runs("runs")
    if not runs:
        print("No runs found in runs/")
        return

    out_dir = "docs/images"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Found {len(runs)} runs")

    plot_loss_comparison(runs, out_dir)
    print("  -> loss_comparison.png")

    plot_grad_norm_comparison(runs, out_dir)
    print("  -> grad_norm_comparison.png")

    plot_gain_comparison(runs, out_dir)
    print("  -> gain_comparison.png")

    plot_depth_scaling(runs, out_dir)
    print("  -> depth_scaling.png")

    plot_48l_comparison(runs, out_dir)
    print("  -> 48l_comparison.png")

    print("Done!")


if __name__ == "__main__":
    main()
