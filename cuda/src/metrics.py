"""Training metrics - PyTorch implementation.

Mirrors mlx/src/metrics.py for correctness comparison.
"""
from __future__ import annotations
import math
import torch


def grad_global_norm(model: torch.nn.Module) -> float:
    """Compute global L2 norm of all gradients.

    Args:
        model: PyTorch model with gradients computed

    Returns:
        L2 norm of concatenated gradients
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.pow(2).sum().item()
    return math.sqrt(total_norm)


def has_nan_or_inf(x: torch.Tensor) -> bool:
    """Check if tensor contains NaN or Inf values.

    Args:
        x: tensor to check

    Returns:
        True if contains NaN or Inf
    """
    return not torch.isfinite(x).all().item()


def gain_proxy_from_Hres(Hres_modules: list) -> float:
    """Compute gain proxy from H_res matrices.

    Multiplies all H_res matrices and returns log10 of max absolute entry.
    This measures cumulative amplification across depth.

    Args:
        Hres_modules: list of StreamMix modules for H_res

    Returns:
        log10 of max absolute entry in composed matrix
    """
    if not Hres_modules:
        return 0.0

    with torch.no_grad():
        P = Hres_modules[0].matrix()
        for m in Hres_modules[1:]:
            P = torch.matmul(m.matrix(), P)
        max_val = torch.abs(P).max().item()
        return math.log10(max(max_val, 1e-12))
