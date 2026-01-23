# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

mHC-POC is a minimal, correctness-first demonstration of **mHC (Manifold-Constrained Hyper-Connections)**, a technique for stabilizing deep Transformer training by constraining residual connections. The project compares three variants under depth stress testing (12/24/48 layers):

- **Baseline**: Standard single-stream residual connections
- **HC**: Multi-stream residual with unconstrained mixing (can amplify signals)
- **mHC**: Multi-stream residual with doubly-stochastic constraint via Sinkhorn-Knopp projection

Phase 1 (current): MLX/Apple Silicon implementation. Phase 2 (planned): CUDA/PyTorch.

## Commands

### Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r mlx/requirements.txt
```

### Run Full Depth Sweep (Main Entry Point)
```bash
bash scripts/run_depth_sweep.sh
```
This generates variant configs, runs all 9 training jobs (3 depths x 3 variants), and outputs to `runs/`.

### Run Single Training
```bash
python -m mlx.src.train --config mlx/configs/tiny_24l_mhc.yaml --out runs/my_run
```

### Generate Variant Configs
```bash
python scripts/gen_variant_configs.py
```

## Architecture

### Core Algorithm (`mlx/src/mhc.py`)
- `StreamMix`: Learned SxS mixing matrix with constraint modes (`softplus`, `softmax_row`, `doubly_stochastic`)
- `sinkhorn_doubly_stochastic()`: Projects matrix to doubly-stochastic via alternating row/column normalization (K=20 iterations default)
- `apply_stream_mix()`: Applies mixing to [B,T,S,D] stream representations

### Model Variants (`mlx/src/model_*.py`)
All variants share identical: embeddings, attention, MLP, optimizer. Only residual path differs.

The multi-stream flow:
1. `H_pre` mixes S streams
2. Mean-merge streams to single [D] vector per token
3. Pass through standard Transformer block
4. Broadcast back to S streams
5. Mix with `H_post^T`, combine with `H_res * x` residual

### Key Metrics (`mlx/src/metrics.py`)
- `grad_global_norm()`: L2 norm of all gradients (detects explosion/vanishing)
- `has_nan_or_inf()`: Training instability indicator
- `gain_proxy_from_Hres()`: Log10 of max entry in H_res composition across layers (measures cumulative amplification)

### Configuration
YAML configs in `mlx/configs/`: `tiny_{12,24,48}l_{baseline,hc,mhc}.yaml`

Key fields: `variant`, `n_layers`, `d_model`, `n_heads`, `streams` (default 4), `sinkhorn_iters` (default 20)

## Development Process

This project follows **TDD** (Red/Green/Refactor) and mandatory **pre-commit quality gates**.

### Pre-Commit Sequence
1. Run tests
2. Fix linting (zero warnings)
3. Format code
4. Validate markdown: `markdown-checker -f "**/*.md"` (ASCII-only)
5. Run `sw-checklist`
6. Update `docs/learnings.md` if issues found

### Code Standards
- Files < 500 lines (prefer 200-300)
- Functions < 50 lines (prefer 10-30)
- Max 3 TODO comments per file
- Never commit FIXMEs

### Commit Format
```
<type>: <summary (50 chars max)>

<detailed explanation>

[AI] Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

## Output Structure

Each run produces:
- `runs/<timestamp>_<variant>_<depth>l/metrics.jsonl` - one JSON per step
- `runs/<timestamp>_<variant>_<depth>l/config.json` - training config
- `runs/<timestamp>_<variant>_<depth>l/plots/` - loss.png, grad_norm.png, nan_inf_events.png, gain_proxy.png
