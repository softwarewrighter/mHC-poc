from __future__ import annotations
import mlx.core as mx
import mlx.nn as nn
from .transformer_block import TransformerBlock

class TinyLM(nn.Module):
    def __init__(self, vocab_size: int, seq_len: int, d_model: int, n_heads: int, d_ff: int, n_layers: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = mx.random.normal(shape=(seq_len, d_model)) * 0.01
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def __call__(self, tokens: mx.array) -> mx.array:
        # tokens: [B,T]
        B, T = tokens.shape
        x = self.embed(tokens) + self.pos[:T][None, :, :]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)  # [B,T,V]
        return logits
