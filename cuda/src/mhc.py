"""mHC core algorithm - PyTorch implementation.

Mirrors mlx/src/mhc.py for correctness comparison.
"""
from __future__ import annotations
import torch
import torch.nn as nn


def sinkhorn_doubly_stochastic(
    A: torch.Tensor, iters: int = 20, eps: float = 1e-12
) -> torch.Tensor:
    """Project a positive matrix A to approximately doubly-stochastic.

    Uses Sinkhorn-Knopp algorithm: alternating row/column normalization.

    Args:
        A: [S, S] tensor, must be elementwise positive
        iters: number of Sinkhorn iterations (default 20)
        eps: numerical stability epsilon

    Returns:
        [S, S] approximately doubly-stochastic tensor
    """
    X = A
    for _ in range(iters):
        # Row normalize
        row_sum = X.sum(dim=1, keepdim=True)
        X = X / torch.clamp(row_sum, min=eps)
        # Column normalize
        col_sum = X.sum(dim=0, keepdim=True)
        X = X / torch.clamp(col_sum, min=eps)
    return X


class StreamMix(nn.Module):
    """Learned stream mixing matrix H of shape [S, S].

    Modes:
        - 'softplus': nonnegative, unconstrained (can amplify)
        - 'softmax_row': row-stochastic (each row sums to 1)
        - 'doubly_stochastic': Sinkhorn-projected (mHC residual)
    """

    def __init__(self, S: int, mode: str, sinkhorn_iters: int = 20):
        super().__init__()
        self.S = S
        self.mode = mode
        self.sinkhorn_iters = sinkhorn_iters

        # Initialize logits near identity
        eye = torch.eye(S)
        init_logits = 2.0 * eye - 1.0  # Pushes diagonal up
        self.logits = nn.Parameter(init_logits)

    def _positive(self) -> torch.Tensor:
        """Convert logits to strictly positive matrix."""
        return torch.exp(self.logits)

    def matrix(self) -> torch.Tensor:
        """Return the constrained mixing matrix."""
        A = self._positive()

        if self.mode == "softplus":
            # Positive but unconstrained
            return A

        if self.mode == "softmax_row":
            # Row-stochastic (rows sum to 1)
            return torch.softmax(self.logits, dim=1)

        if self.mode == "doubly_stochastic":
            # Doubly stochastic via Sinkhorn projection
            return sinkhorn_doubly_stochastic(A, iters=self.sinkhorn_iters)

        raise ValueError(f"Unknown mode: {self.mode}")


def apply_stream_mix(H: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Apply stream-mix matrix H:[S,S] to x:[B,T,S,D] along stream axis.

    Args:
        H: [S, S] mixing matrix
        x: [B, T, S, D] multi-stream hidden states

    Returns:
        [B, T, S, D] mixed hidden states
    """
    # x: [B, T, S, D] -> [B, T, D, S] for matmul
    xt = x.permute(0, 1, 3, 2)  # [B, T, D, S]
    # Multiply: [B, T, D, S] @ [S, S]^T -> [B, T, D, S]
    yt = torch.matmul(xt, H.T)
    # Back to [B, T, S, D]
    y = yt.permute(0, 1, 3, 2)
    return y
