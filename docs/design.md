# Design notes (mhc-demo)

## Goals
- Small, readable codebase that **actually implements mHC**
- Clear apples-to-apples comparison:
  - baseline residual vs HC vs mHC
- Metrics + plots suitable for a video walkthrough

## Non-goals
- State-of-the-art accuracy
- Peak training throughput optimizations (those come later, esp. for CUDA)

## MLX implementation choices
- Decoder-only Transformer
- Synthetic incremental-token dataset (fast, learnable, stable baseline)
- mHC stream count S defaults to 4
- Sinkhorn projection uses fixed iteration count (default 20)

## Files
- `mlx/src/mhc.py`: parametrizations + Sinkhorn projection
- `mlx/src/model_*.py`: baseline/HC/mHC models
- `mlx/src/train.py`: training loop, logging, saving plots
- `scripts/run_depth_sweep.sh`: run 12/24/48 depth sweeps for all variants
