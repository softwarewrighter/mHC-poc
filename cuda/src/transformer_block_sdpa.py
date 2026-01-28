"""Transformer block with PyTorch SDPA (Scaled Dot-Product Attention).

Uses torch.nn.functional.scaled_dot_product_attention which automatically
selects the best backend (Flash Attention, Memory-Efficient, or Math)
based on hardware and input characteristics.

On supported GPUs (Ampere+), this uses Flash Attention for O(N) memory
and faster computation compared to standard O(N^2) attention.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadSDPA(nn.Module):
    """Multi-head attention using PyTorch's SDPA.

    This provides explicit control over the attention implementation,
    enabling flash attention on supported hardware.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        # Separate projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with SDPA.

        Args:
            x: [B, T, D] input tensor

        Returns:
            [B, T, D] output tensor
        """
        B, T, D = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)  # [B, T, D]
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head: [B, T, D] -> [B, n_heads, T, head_dim]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # SDPA with causal mask - automatically uses flash attention if available
        # is_causal=True enables efficient causal masking
        dropout_p = self.dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout_p, is_causal=True
        )

        # Reshape back: [B, n_heads, T, head_dim] -> [B, T, D]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)

        # Output projection
        return self.out_proj(attn_out)


class TransformerBlockSDPA(nn.Module):
    """Pre-norm Transformer block using SDPA for attention.

    This block uses PyTorch's scaled_dot_product_attention which
    automatically selects Flash Attention on supported hardware.
    """

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiheadSDPA(d_model, n_heads, dropout)
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
        # Self-attention with pre-norm
        x = x + self.dropout(self.attn(self.ln1(x)))

        # Feed-forward with pre-norm
        x = x + self.dropout(self.ff(self.ln2(x)))

        return x
