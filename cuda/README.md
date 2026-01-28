# CUDA/PyTorch Implementation

Phase 2 implementation of mHC using PyTorch for NVIDIA GPUs.

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA support
- NVIDIA GPU with CUDA toolkit

## Setup

```bash
# Create environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r cuda/requirements.txt

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Usage

```bash
# Run single training
python -m cuda.src.train --config cuda/configs/tiny_24l_mhc.yaml --out runs/cuda_test

# Generate variant configs (same script as MLX)
python scripts/gen_variant_configs.py --target cuda

# Run depth sweep
bash scripts/run_cuda_depth_sweep.sh
```

## Structure

```
cuda/
├── README.md
├── requirements.txt
├── configs/           # YAML configs (to be generated)
└── src/
    ├── __init__.py
    ├── mhc.py             # Core Sinkhorn-Knopp algorithm
    ├── transformer_block.py
    ├── model_baseline.py  # Single-stream residual
    ├── model_hc.py        # Multi-stream unconstrained
    ├── model_mhc.py       # Multi-stream doubly-stochastic
    ├── dataset.py         # Incrementing token dataset
    ├── metrics.py         # Gradient norm, gain proxy
    └── train.py           # Training loop
```

## Comparison with MLX

The CUDA implementation mirrors the MLX version exactly:
- Same model architectures
- Same Sinkhorn-Knopp algorithm
- Same metrics
- Same config format

This allows direct comparison of results across platforms.

## Optional: FlashAttention-2

For faster attention on supported GPUs:

```bash
pip install flash-attn --no-build-isolation
```

Then modify `transformer_block.py` to use FlashAttention.
