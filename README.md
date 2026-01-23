# mhc-demo

A minimal, correctness-first demo repo that **implements mHC (Manifold-Constrained Hyper-Connections)** and
shows its **stability benefits** under a depth stress-test.

- Phase 1: **MLX (Apple Silicon)** reference implementation.
- Phase 2: CUDA (HF/PyTorch; Unsloth optional depending on compatibility with forward-pass changes).

> This repo is intentionally small and explicit. Speed optimizations come after correctness.

## Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) for fast dependency management

## Quick start (MLX)

### 1) Create an environment
```bash
uv venv
source .venv/bin/activate
uv pip install -r mlx/requirements.txt
```

### 2) Run a depth sweep
```bash
bash scripts/run_depth_sweep.sh
```

Outputs go to `runs/`:
- `metrics.jsonl` (one JSON per step)
- `config.json`
- `plots/*.png`

### 3) Compare runs
Open the plots:
- `loss.png`
- `grad_norm.png`
- `nan_inf_events.png`
- `gain_proxy.png`

## Repo map
- `docs/eli4-mHC.md` – simple explanation using analogies (start here)
- `docs/eli5-mHC.md` – technical explanation + how THIS repo implements mHC
- `mlx/src/` – MLX implementation (baseline / HC / mHC)
- `scripts/` – convenience scripts for running and plotting

## Notes
- The default dataset is a synthetic "incrementing token" task: sequences follow `(start + i) mod vocab`.
  This is learnable (unlike pure random tokens), trains fast, and is great for stability stress-testing.
- The most honest small-scale demo is **stability vs depth** (loss + grad spikes + NaNs), not SOTA accuracy.
