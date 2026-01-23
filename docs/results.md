# Results

## Hardware

- Apple Silicon (M-series) with MLX 0.30.3
- Python 3.10.18

## Command

```bash
bash scripts/run_depth_sweep.sh
```

Runs 9 training jobs: 3 depths (12, 24, 48 layers) x 3 variants (baseline, HC, mHC), each for 800 steps.

## Summary

### Gain Proxy (log10 scale)

The gain proxy measures cumulative signal amplification through the residual path. Lower/bounded values indicate stability.

| Depth | Baseline | HC | mHC |
|-------|----------|-----|------|
| 12L | 0.0 | 7.0 | -0.5 |
| 24L | 0.0 | 14.4 | -0.6 |
| 48L | 0.0 | **27.3** | **-0.6** |

**Key observation**: HC's gain grows exponentially with depth (10^7 at 12L, 10^14 at 24L, 10^27 at 48L), while mHC stays constant at ~10^-0.6 regardless of depth. This demonstrates the doubly-stochastic constraint successfully bounds amplification.

### Final Loss

| Depth | Baseline | HC | mHC |
|-------|----------|-----|------|
| 12L | 1.78 | 0.0001 | 0.002 |
| 24L | 1.86 | 0.0001 | 0.0002 |
| 48L | 3.79 | **5.54** | **0.0002** |

**Key observation**: At 48 layers:
- Baseline struggles (loss 3.79) - standard residuals can't effectively train very deep networks on this task
- HC becomes unstable (loss 5.54) - the unconstrained amplification prevents convergence
- mHC converges perfectly (loss 0.0002) - the constraint maintains trainability at any depth

### NaN/Inf Events

All runs completed with zero NaN/Inf events. This is due to:
1. Conservative learning rate (1e-4)
2. Simple synthetic dataset
3. 800 steps may not be enough for HC to fully diverge

However, the gain proxy and final loss clearly show HC's instability even without explicit NaN events.

## Plots

Key comparison plots are in `docs/images/`:
- `hc_48l_gain.png` - HC 48L gain proxy (exponential growth)
- `mhc_48l_gain.png` - mHC 48L gain proxy (bounded)
- `hc_48l_loss.png` - HC 48L training loss (fails to converge)
- `mhc_48l_loss.png` - mHC 48L training loss (clean convergence)

Full run outputs in `runs/*/plots/` (gitignored).

## Interpretation

The results validate the core mHC hypothesis: **unconstrained residual mixing (HC) creates unbounded signal amplification that grows exponentially with depth, while doubly-stochastic constrained mixing (mHC) maintains bounded amplification regardless of depth.**

At 48 layers, the difference is dramatic:
- HC accumulates 10^27x theoretical amplification and fails to train (loss 5.54)
- mHC maintains 10^-0.6x amplification and trains perfectly (loss 0.0002)

This demonstrates that mHC's Sinkhorn-projected doubly-stochastic constraint is the key to stable deep training with multi-stream residuals.
