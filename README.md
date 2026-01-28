# mhc-demo

A minimal, correctness-first demo repo that **implements mHC (Manifold-Constrained Hyper-Connections)** and
shows its **stability benefits** under a depth stress-test.

Based on the paper: [arXiv:2512.24880](https://arxiv.org/abs/2512.24880)

[![mHC Proof of Concept](images/thumbnail1.png)](https://www.youtube.com/shorts/fh21_zIK2ZE)

*Click to watch the YouTube Short*

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
- `docs/history.md` – evolution from sigmoid to ResNet to HC to mHC (with paper references)
- `mlx/src/` – MLX implementation (baseline / HC / mHC)
- `scripts/` – convenience scripts for running and plotting

## Results

Depth stress test comparing baseline, HC, and mHC at 12, 24, and 48 layers:

### Gain Proxy (log10 scale - lower is more stable)

| Depth | Baseline | HC | mHC |
|-------|----------|-----|------|
| 12L | 0.0 | 7.0 | -0.5 |
| 24L | 0.0 | 14.4 | -0.6 |
| 48L | 0.0 | **27.3** | **-0.6** |

HC's amplification explodes with depth (10^7 to 10^27), while mHC stays bounded at ~10^-0.6.

### Final Loss (lower is better)

| Depth | Baseline | HC | mHC |
|-------|----------|-----|------|
| 12L | 1.78 | 0.0001 | 0.002 |
| 24L | 1.86 | 0.0001 | 0.0002 |
| 48L | 3.79 | **5.54** | **0.0002** |

At 48 layers, HC becomes unstable (loss 5.54), while mHC converges perfectly (0.0002).

### Key Insight: Gain Scaling with Depth

![Depth Scaling](docs/images/depth_scaling.png)

**What this shows:** As we add more layers, HC's residual amplification grows exponentially (red line: 10^7 → 10^14 → 10^27), while mHC's doubly-stochastic constraint keeps amplification bounded near zero (blue line stays flat). This is the core benefit of mHC.

### 48-Layer Head-to-Head Comparison

![48L Comparison](docs/images/48l_comparison.png)

**Left (Loss):** mHC (blue) converges rapidly to near-zero loss, while HC (red) is stuck at ~5.5 and never learns. **Right (Gain):** HC has 10^27 amplification that prevents learning; mHC stays bounded near zero.

### Loss Across All Depths

![Loss Comparison](docs/images/loss_comparison.png)

**What this shows:** At 12 layers (left), all variants converge. At 24 layers (middle), baseline struggles but HC/mHC still work. At 48 layers (right), only mHC (blue) converges - baseline (gray) oscillates and HC (red) flatlines at high loss.

### Gradient Norm Across All Depths

![Gradient Norm Comparison](docs/images/grad_norm_comparison.png)

**What this shows:** Gradient norms reveal training health. At 48 layers (right), HC (red) shows erratic gradient behavior while mHC (blue) maintains stable gradients throughout training. The baseline (gray) shows gradients that decay but remain stable - the benefit of single-stream residuals without the expressiveness of multi-stream.

## Notes
- The default dataset is a synthetic "incrementing token" task: sequences follow `(start + i) mod vocab`.
  This is learnable (unlike pure random tokens), trains fast, and is great for stability stress-testing.
- The most honest small-scale demo is **stability vs depth** (loss + grad spikes + NaNs), not SOTA accuracy.
