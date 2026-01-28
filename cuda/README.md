# CUDA/PyTorch Implementation

Phase 2 implementation of mHC using PyTorch for NVIDIA GPUs.

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA support
- NVIDIA GPU with CUDA toolkit

## Setup

```bash
# Create environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r cuda/requirements.txt

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Usage

### Standard Attention
```bash
# Quick demo (50 steps, ~17 seconds)
python -m cuda.src.train --config cuda/configs/demo_mhc.yaml --out runs/cuda_demo

# Full training run
python -m cuda.src.train --config cuda/configs/tiny_24l_mhc.yaml --out runs/cuda_test

# Run depth sweep (9 jobs: 3 depths x 3 variants)
bash scripts/run_cuda_depth_sweep.sh
```

### SDPA Attention (Flash backend)
```bash
# Quick demo with SDPA
python -m cuda.src.train_sdpa --config cuda/configs/demo_mhc_sdpa.yaml --out runs/cuda_sdpa_demo
```

## Documentation

- [results.md](docs/results.md) - CUDA training results and analysis
- [comparisons.md](docs/comparisons.md) - Cross-platform comparison (Apple/CUDA/SDPA)

## Structure

```
cuda/
├── README.md
├── requirements.txt
├── docs/
│   ├── results.md          # CUDA results and analysis
│   └── comparisons.md      # Cross-platform benchmarks
├── configs/
│   ├── demo_mhc.yaml       # Quick demo (standard attention)
│   ├── demo_mhc_sdpa.yaml  # Quick demo (SDPA/flash)
│   └── tiny_*_*.yaml       # Depth sweep configs
└── src/
    ├── __init__.py
    ├── mhc.py                    # Core Sinkhorn-Knopp algorithm
    ├── transformer_block.py      # Standard attention
    ├── transformer_block_sdpa.py # SDPA (flash) attention
    ├── model_baseline.py         # Single-stream residual
    ├── model_baseline_sdpa.py    # Single-stream with SDPA
    ├── model_hc.py               # Multi-stream unconstrained
    ├── model_hc_sdpa.py          # HC with SDPA
    ├── model_mhc.py              # Multi-stream doubly-stochastic
    ├── model_mhc_sdpa.py         # mHC with SDPA
    ├── dataset.py                # Incrementing token dataset
    ├── metrics.py                # Gradient norm, gain proxy
    ├── train.py                  # Training loop (standard)
    └── train_sdpa.py             # Training loop (SDPA)
```

## Attention Backends

### Standard (`train.py`)
Uses `nn.MultiheadAttention` - compatible with all GPUs.

### SDPA (`train_sdpa.py`)
Uses PyTorch's `scaled_dot_product_attention` which automatically selects:
- **Flash Attention** on Ampere+ GPUs (fastest, O(N) memory)
- **Memory-Efficient Attention** as fallback
- **Math Attention** on older GPUs

SDPA is recommended for PyTorch 2.0+ and provides ~1-4% speedup on small models, with greater gains on larger models and longer sequences.

## Comparison with MLX

The CUDA implementation mirrors the MLX version exactly:
- Same model architectures
- Same Sinkhorn-Knopp algorithm
- Same metrics
- Same config format

**Validation**: Gain proxy values match exactly across platforms (-0.6), confirming correct implementation.
