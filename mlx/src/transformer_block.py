from __future__ import annotations
import mlx.core as mx
import mlx.nn as nn

def causal_mask(T: int) -> mx.array:
    # Lower-triangular mask: allow attention to self and past
    m = mx.tril(mx.ones((T, T)))
    # Convert to additive mask (0 for allowed, -inf for disallowed)
    neg_inf = mx.array(-1e9)
    return (1.0 - m) * neg_inf

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiHeadAttention(d_model, n_heads, bias=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = dropout

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B,T,D]
        B, T, D = x.shape
        m = causal_mask(T)  # [T,T]
        # MLX MHA expects mask broadcastable; we'll pass [T,T]
        h = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), mask=m)
        x = x + h
        x = x + self.ff(self.ln2(x))
        return x
