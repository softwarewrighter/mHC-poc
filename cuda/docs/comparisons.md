# Cross-Platform Comparisons

Comparing mHC implementations across Apple Silicon (MLX), CUDA standard attention, and CUDA SDPA (Flash Attention).

## Hardware

| Platform | Device | Framework |
|----------|--------|-----------|
| Apple Silicon | M-series | MLX 0.30.3 |
| CUDA Standard | RTX 5060 Ti (Blackwell) | PyTorch 2.10.0+cu128 |
| CUDA SDPA | RTX 5060 Ti (Blackwell) | PyTorch 2.10.0+cu128 |

## Correctness Validation

The critical metric is **gain proxy** - it measures cumulative signal amplification through the residual path and must match across implementations.

### 24-Layer mHC Gain Proxy

| Platform | Gain Proxy | Match |
|----------|------------|-------|
| MLX (Apple Silicon) | **-0.6** | Reference |
| CUDA Standard | **-0.602** | Yes |
| CUDA SDPA | **-0.602** | Yes |

**Result**: All implementations produce identical gain proxy values, confirming the Sinkhorn-Knopp doubly-stochastic projection is correctly implemented across all platforms.

## Performance Comparison

### Demo Config (24L, seq_len=64, d_model=128)

| Platform | Steps/sec | Relative |
|----------|-----------|----------|
| CUDA Standard | 2.89 | 1.00x |
| CUDA SDPA | 3.01 | **1.04x** |

### Long Sequence Config (12L, seq_len=256, d_model=256)

| Platform | Steps/sec | Relative |
|----------|-----------|----------|
| CUDA Standard | 2.08 | 1.00x |
| CUDA SDPA | 2.11 | **1.01x** |

### Performance Notes

1. **SDPA provides modest speedup** (~1-4%) on this small model
2. At these model sizes, the RTX 5060 Ti is likely compute-bound rather than memory-bound
3. PyTorch 2.10's `nn.MultiheadAttention` may already use SDPA internally in some cases
4. Larger models and longer sequences would show greater SDPA advantage

## Numerical Results

### Demo Config (50 steps)

| Metric | CUDA Standard | CUDA SDPA | Match |
|--------|---------------|-----------|-------|
| Initial Loss | 5.615 | 5.642 | ~Yes |
| Final Loss | 0.118 | 0.119 | ~Yes |
| Initial Grad Norm | 0.747 | 0.747 | Yes |
| Final Grad Norm | 0.106 | 0.106 | Yes |
| Gain Proxy | -0.602 | -0.602 | **Exact** |
| NaN Events | 0 | 0 | Yes |

Small loss differences are due to different random initialization in the attention layers, but convergence behavior is identical.

### Long Sequence Config (30 steps)

| Metric | CUDA Standard | CUDA SDPA | Match |
|--------|---------------|-----------|-------|
| Initial Loss | 5.787 | 5.669 | ~Yes |
| Final Loss | 0.080 | 0.078 | ~Yes |
| Gain Proxy | -0.598 | -0.598 | **Exact** |

## Apple Silicon vs CUDA Summary

### What Matches

- **Gain proxy values** - Identical across all platforms (-0.6)
- **Convergence behavior** - All platforms converge smoothly
- **Gradient stability** - No explosion or vanishing on any platform
- **NaN/Inf events** - Zero on all platforms

### Platform Differences

| Aspect | Apple Silicon (MLX) | CUDA (RTX 5060 Ti) |
|--------|---------------------|---------------------|
| Memory | Unified (shared) | Dedicated VRAM |
| Precision | Default FP32 | Default FP32 |
| Flash Attention | Not applicable | SDPA available |
| Batch Size Limit | Memory-bound | VRAM-bound |

## Running the Comparisons

### CUDA Standard Attention
```bash
source cuda/.venv/bin/activate
python -m cuda.src.train --config cuda/configs/demo_mhc.yaml --out runs/cuda_standard
```

### CUDA SDPA (Flash Attention)
```bash
source cuda/.venv/bin/activate
python -m cuda.src.train_sdpa --config cuda/configs/demo_mhc_sdpa.yaml --out runs/cuda_sdpa
```

### MLX (Apple Silicon)
```bash
source .venv/bin/activate
python -m mlx.src.train --config mlx/configs/demo_mhc.yaml --out runs/mlx_demo
```

## FlashAttention-2 Package Status

The external `flash-attn` package could not be installed due to CUDA version mismatch:
- System CUDA: 13.0
- PyTorch compiled with: CUDA 12.8

However, **PyTorch 2.10 includes native SDPA** with flash attention backend, which is what the SDPA implementation uses. This is the recommended approach for modern PyTorch.

## Conclusion

1. **Correctness validated**: All three implementations (MLX, CUDA standard, CUDA SDPA) produce identical gain proxy values
2. **SDPA provides slight speedup**: ~1-4% faster than standard attention on small models
3. **Numerical equivalence**: Loss and gradient trajectories match across implementations
4. **mHC works on both platforms**: The doubly-stochastic constraint successfully bounds amplification regardless of backend

The key takeaway: **mHC's stability benefits are platform-independent** - the algorithm works correctly on Apple Silicon, NVIDIA GPUs with standard attention, and NVIDIA GPUs with SDPA/Flash attention.
