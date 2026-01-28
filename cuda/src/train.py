"""Training script - PyTorch/CUDA implementation.

Mirrors mlx/src/train.py for correctness comparison.

Usage:
    python -m cuda.src.train --config cuda/configs/tiny_24l_mhc.yaml --out runs/cuda_test
"""
from __future__ import annotations
import os
import json
import time
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

from .dataset import IncrementingTokenDataset
from .model_baseline import TinyLM
from .model_hc import TinyLM_HC
from .model_mhc import TinyLM_mHC
from .metrics import grad_global_norm, has_nan_or_inf, gain_proxy_from_Hres


def build_model(cfg: dict, device: torch.device) -> nn.Module:
    """Build model based on config variant."""
    variant = cfg["variant"]
    common = dict(
        vocab_size=cfg["vocab_size"],
        seq_len=cfg["seq_len"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"],
        n_layers=cfg["n_layers"],
    )

    if variant == "baseline":
        model = TinyLM(**common)
    elif variant == "hc":
        model = TinyLM_HC(**common, streams=cfg["streams"])
    elif variant == "mhc":
        model = TinyLM_mHC(
            **common, streams=cfg["streams"], sinkhorn_iters=cfg["sinkhorn_iters"]
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return model.to(device)


def get_hres_modules(model: nn.Module, cfg: dict) -> list:
    """Get H_res modules for gain proxy calculation."""
    if cfg["variant"] in ("hc", "mhc"):
        return list(model.H_res)
    return []


def train(cfg_path: str, out_dir: str):
    """Main training loop."""
    # Load config
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Setup output directory
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set seeds
    torch.manual_seed(int(cfg["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(cfg["seed"]))

    # Create dataset and model
    ds = IncrementingTokenDataset(cfg["vocab_size"], cfg["seq_len"], device)
    model = build_model(cfg, device)
    model.train()

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )

    # Loss function
    loss_fn = nn.CrossEntropyLoss()

    # Metrics file
    metrics_path = os.path.join(out_dir, "metrics.jsonl")
    fmet = open(metrics_path, "w", encoding="utf-8")

    t0 = time.time()
    nan_events = 0

    for step in range(int(cfg["steps"])):
        # Get batch
        x, y = ds.batch(int(cfg["batch_size"]))

        # Forward pass
        optimizer.zero_grad()
        logits = model(x)  # [B, T, V]

        # Compute loss
        B, T, V = logits.shape
        loss = loss_fn(logits.view(B * T, V), y.view(B * T))

        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrics
        gnorm = grad_global_norm(model)
        nan_here = has_nan_or_inf(loss) or has_nan_or_inf(logits)
        if nan_here:
            nan_events += 1

        Hres_modules = get_hres_modules(model, cfg)
        gain = gain_proxy_from_Hres(Hres_modules) if Hres_modules else 0.0

        rec = {
            "step": step,
            "loss": float(loss.item()),
            "grad_norm": float(gnorm),
            "nan_or_inf": int(nan_here),
            "nan_or_inf_total": int(nan_events),
            "gain_proxy_log10_max_entry": float(gain),
            "elapsed_sec": float(time.time() - t0),
        }
        fmet.write(json.dumps(rec) + "\n")

        if step % int(cfg["log_every"]) == 0:
            print(
                f"[{cfg['variant']}] step={step:5d} "
                f"loss={rec['loss']:.4f} gnorm={rec['grad_norm']:.3f} "
                f"nan={nan_events} gain={gain:.3f}"
            )

    fmet.close()
    print(f"Training complete. Metrics saved to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()
    train(args.config, args.out)
