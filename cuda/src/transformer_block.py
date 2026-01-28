"""Transformer block - PyTorch implementation.

Mirrors mlx/src/transformer_block.py for correctness comparison.
"""
from __future__ import annotations
import torch
import torch.nn as nn


def causal_mask(T: int, device: torch.device) -> torch.Tensor:
    """Create causal attention mask.

    Args:
        T: sequence length
        device: torch device

    Returns:
        [T, T] mask with 0 for allowed positions, -inf for masked
    """
    mask = torch.triu(torch.ones(T, T, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask


class TransformerBlock(nn.Module):
    """Standard pre-norm Transformer block with causal attention."""

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, T, D] input tensor

        Returns:
            [B, T, D] output tensor
        """
        B, T, D = x.shape
        mask = causal_mask(T, x.device)

        # Self-attention with pre-norm
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm, attn_mask=mask, need_weights=False
        )
        x = x + self.dropout(attn_out)

        # Feed-forward with pre-norm
        x = x + self.dropout(self.ff(self.ln2(x)))

        return x
