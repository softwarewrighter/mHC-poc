"""mHC model - PyTorch implementation.

Manifold-Constrained Hyper-Connections with doubly-stochastic residual mixing.
Mirrors mlx/src/model_mhc.py for correctness comparison.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from .transformer_block import TransformerBlock
from .mhc import StreamMix, apply_stream_mix


class TinyLM_mHC(nn.Module):
    """mHC variant: multi-stream residual with doubly-stochastic constraint."""

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        streams: int = 4,
        sinkhorn_iters: int = 20,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.streams = streams

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(seq_len, d_model) * 0.01)

        # Transformer blocks operate on merged representation
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )

        # Stream mixing matrices per layer
        self.H_pre = nn.ModuleList(
            [StreamMix(streams, mode="softmax_row") for _ in range(n_layers)]
        )
        self.H_post = nn.ModuleList(
            [StreamMix(streams, mode="softmax_row") for _ in range(n_layers)]
        )
        # Doubly-stochastic residual mixer (bounded - stable!)
        self.H_res = nn.ModuleList(
            [
                StreamMix(streams, mode="doubly_stochastic", sinkhorn_iters=sinkhorn_iters)
                for _ in range(n_layers)
            ]
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
        base = self.embed(tokens) + self.pos[:T].unsqueeze(0)  # [B, T, D]

        # Initialize streams by tiling
        x = base.unsqueeze(2).expand(-1, -1, self.streams, -1)  # [B, T, S, D]

        for l, blk in enumerate(self.blocks):
            # Pre-mix streams
            Hpre = self.H_pre[l].matrix()
            x_pre = apply_stream_mix(Hpre, x)  # [B, T, S, D]

            # Merge streams -> run transformer block
            merged = x_pre.mean(dim=2)  # [B, T, D]
            f_out = blk(merged)  # [B, T, D]

            # Broadcast back to streams, then post-mix
            f_rep = f_out.unsqueeze(2).expand(-1, -1, self.streams, -1)
            Hpost = self.H_post[l].matrix()
            f_mixed = apply_stream_mix(Hpost.T, f_rep)

            # Residual mixing (doubly-stochastic - bounded!)
            Hres = self.H_res[l].matrix()
            res_mixed = apply_stream_mix(Hres, x)

            x = res_mixed + f_mixed

        # Final prediction from merged streams
        x_final = x.mean(dim=2)  # [B, T, D]
        x_final = self.ln_f(x_final)
        logits = self.head(x_final)
        return logits
