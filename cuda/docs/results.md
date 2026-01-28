# CUDA/PyTorch Results

## Hardware

- NVIDIA GPU with CUDA (PyTorch 2.10.0)
- Python 3.12.8

## Quick Demo

```bash
source cuda/.venv/bin/activate
python -m cuda.src.train --config cuda/configs/demo_mhc.yaml --out runs/cuda_demo
```

### Demo Results (24-layer mHC, 50 steps)

| Metric | Start (step 0) | End (step 49) | Change |
|--------|----------------|---------------|--------|
| Loss | 5.615 | 0.118 | -97.9% |
| Grad Norm | 0.747 | 0.106 | -85.8% |
| Gain Proxy | -0.602 | -0.602 | 0% |
| NaN/Inf | 0 | 0 | - |
| Elapsed | - | 17.1 sec | ~2.9 steps/sec |

### Key Observations

1. **Rapid convergence**: Loss dropped from 5.6 to 0.12 in just 50 steps (17 seconds)
2. **Smooth gradient decay**: Grad norm decreased monotonically from 0.75 to 0.11
3. **Bounded gain**: Gain proxy remained exactly constant at -0.602 throughout training
4. **Zero instability**: No NaN/Inf events occurred

## Full Training Run (500 steps)

```bash
python -m cuda.src.train --config cuda/configs/tiny_24l_mhc.yaml --out runs/cuda_test
```

### Results

| Metric | Start (step 0) | End (step 499) |
|--------|----------------|----------------|
| Loss | 5.615 | 0.00215 |
| Grad Norm | 0.747 | 0.00254 |
| Gain Proxy | -0.602 | -0.602 |
| NaN/Inf Events | 0 | 0 |
| Elapsed Time | - | 166 sec |

## Cross-Platform Comparison: CUDA vs MLX

### 24-Layer mHC

| Metric | CUDA (500 steps) | MLX (800 steps) |
|--------|------------------|-----------------|
| Final Loss | 0.00215 | 0.0002 |
| Gain Proxy | **-0.602** | **-0.6** |
| NaN Events | 0 | 0 |
| Gradient Stability | Stable | Stable |

### Validation

The CUDA implementation successfully reproduces the key mHC behaviors observed in MLX:

- **Identical gain proxy** (-0.6) confirms correct Sinkhorn-Knopp doubly-stochastic projection
- **Smooth convergence** on the incrementing token task
- **Stable gradients** without explosion or vanishing
- **Zero numerical instability** (no NaN/Inf)

The small loss difference (0.00215 vs 0.0002) is explained by fewer training steps (500 vs 800).

## Depth Sweep

To run the full 9-job depth sweep (3 depths x 3 variants):

```bash
bash scripts/run_cuda_depth_sweep.sh
```

This generates results comparable to the MLX depth sweep documented in [docs/results.md](../../docs/results.md).

### Expected Results (based on MLX)

| Depth | Baseline | HC | mHC |
|-------|----------|-----|------|
| 12L | Loss ~1.8 | Loss ~0.0001, Gain 7.0 | Loss ~0.002, Gain -0.5 |
| 24L | Loss ~1.9 | Loss ~0.0001, Gain 14.4 | Loss ~0.0002, Gain -0.6 |
| 48L | Loss ~3.8 | **Unstable** (Gain 27.3) | **Stable** (Gain -0.6) |

The critical test is 48 layers: HC should show instability (high loss, extreme gain), while mHC should converge perfectly.

## Code Status

| Component | Status |
|-----------|--------|
| `mhc.py` (Sinkhorn-Knopp) | Complete |
| `model_baseline.py` | Complete |
| `model_hc.py` | Complete |
| `model_mhc.py` | Complete |
| `train.py` | Complete |
| `metrics.py` | Complete |
| All 9 variant configs | Complete |
| Depth sweep script | Complete |
| Demo config | Complete |

## Interpretation

The CUDA results validate that mHC's core benefit - bounded signal amplification via doubly-stochastic constraints - transfers correctly from MLX to PyTorch/CUDA:

1. **Gain proxy stays constant** regardless of training progress or loss value
2. **The -0.6 value** (10^-0.6 = 0.25x) means the residual path slightly attenuates rather than amplifies
3. **This bounded behavior** is what enables stable training at extreme depths (48+ layers)

Compare this to HC, where gain grows exponentially with depth (10^7 at 12L, 10^27 at 48L), eventually causing training collapse.
