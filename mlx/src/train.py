from __future__ import annotations
import os, json, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import yaml

from .dataset import IncrementingTokenDataset
from .model_baseline import TinyLM
from .model_hc import TinyLM_HC
from .model_mhc import TinyLM_mHC
from .metrics import grad_global_norm, has_nan_or_inf, gain_proxy_from_Hres

def cross_entropy_logits(logits: mx.array, targets: mx.array) -> mx.array:
    # logits: [B,T,V], targets: [B,T]
    # Flatten
    B, T, V = logits.shape
    l = logits.reshape((B*T, V))
    t = targets.reshape((B*T,))
    return mx.mean(nn.losses.cross_entropy(l, t))

def build_model(cfg: dict):
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
        return TinyLM(**common)
    if variant == "hc":
        return TinyLM_HC(**common, streams=cfg["streams"])
    if variant == "mhc":
        return TinyLM_mHC(**common, streams=cfg["streams"], sinkhorn_iters=cfg["sinkhorn_iters"])
    raise ValueError(f"unknown variant: {variant}")

def get_hres_mats(model, cfg):
    if cfg["variant"] == "hc" or cfg["variant"] == "mhc":
        return [m.matrix() for m in model.H_res]
    return []

def train(cfg_path: str, out_dir: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    mx.random.seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))

    ds = IncrementingTokenDataset(cfg["vocab_size"], cfg["seq_len"])
    model = build_model(cfg)
    model.train()

    opt = optim_from_cfg(cfg, model)

    metrics_path = os.path.join(out_dir, "metrics.jsonl")
    fmet = open(metrics_path, "w", encoding="utf-8")

    def loss_fn(m, x, y):
        logits = m(x)
        loss = cross_entropy_logits(logits, y)
        return loss, logits

    value_and_grad = mx.value_and_grad(lambda m, x, y: loss_fn(m, x, y)[0])

    t0 = time.time()
    nan_events = 0

    for step in range(int(cfg["steps"])):
        x, y = ds.batch(int(cfg["batch_size"]))
        loss, grads = value_and_grad(model, x, y)
        opt.update(model, grads)

        # force compute now
        mx.eval(loss)

        # metrics
        gnorm = grad_global_norm(grads)
        logits = model(x)
        mx.eval(logits)
        nan_here = has_nan_or_inf(loss) or has_nan_or_inf(logits)
        if nan_here:
            nan_events += 1

        Hres_list = get_hres_mats(model, cfg)
        gain = gain_proxy_from_Hres(Hres_list) if Hres_list else 0.0

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
            print(f"[{cfg['variant']}] step={step:5d} loss={rec['loss']:.4f} gnorm={rec['grad_norm']:.3f} nan={nan_events} gain={gain:.3f}")

    fmet.close()

    # plots
    from .plot import plot_run
    plot_run(out_dir)

def optim_from_cfg(cfg: dict, model):
    lr = float(cfg["lr"])
    wd = float(cfg.get("weight_decay", 0.0))
    opt = optim.AdamW(learning_rate=lr, weight_decay=wd)
    opt.init(model.parameters())
    return opt

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    train(args.config, args.out)
