"""Baseline model - PyTorch implementation.

Standard single-stream residual connections.
Mirrors mlx/src/model_baseline.py for correctness comparison.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from .transformer_block import TransformerBlock


class TinyLM(nn.Module):
    """Baseline tiny language model with standard residuals."""

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(seq_len, d_model) * 0.01)

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            tokens: [B, T] input token indices

        Returns:
            [B, T, V] logits
        """
        B, T = tokens.shape
        x = self.embed(tokens) + self.pos[:T].unsqueeze(0)

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits
