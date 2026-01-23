from __future__ import annotations
import mlx.core as mx
import mlx.nn as nn
from .transformer_block import TransformerBlock
from .mhc import StreamMix, apply_stream_mix

class TinyLM_mHC(nn.Module):
    """mHC variant: same multi-stream residual as HC, but H_res is doubly-stochastic via Sinkhorn."""
    def __init__(self, vocab_size: int, seq_len: int, d_model: int, n_heads: int, d_ff: int, n_layers: int,
                 streams: int = 4, sinkhorn_iters: int = 20):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.streams = streams

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = mx.random.normal(shape=(seq_len, d_model)) * 0.01

        self.blocks = [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]

        self.H_pre = [StreamMix(streams, mode="softmax_row") for _ in range(n_layers)]
        self.H_post = [StreamMix(streams, mode="softmax_row") for _ in range(n_layers)]
        self.H_res = [StreamMix(streams, mode="doubly_stochastic", sinkhorn_iters=sinkhorn_iters) for _ in range(n_layers)]

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def __call__(self, tokens: mx.array) -> mx.array:
        B, T = tokens.shape
        base = self.embed(tokens) + self.pos[:T][None, :, :]

        x = mx.repeat(base[:, :, None, :], self.streams, axis=2)  # [B,T,S,D]

        for l, blk in enumerate(self.blocks):
            Hpre = self.H_pre[l].matrix()
            x_pre = apply_stream_mix(Hpre, x)

            merged = mx.mean(x_pre, axis=2)  # [B,T,D]
            f_out = blk(merged)

            f_rep = mx.repeat(f_out[:, :, None, :], self.streams, axis=2)
            Hpost = self.H_post[l].matrix()
            f_mixed = apply_stream_mix(mx.transpose(Hpost, (1,0)), f_rep)

            Hres = self.H_res[l].matrix()
            res_mixed = apply_stream_mix(Hres, x)

            x = res_mixed + f_mixed

        x_final = mx.mean(x, axis=2)
        x_final = self.ln_f(x_final)
        logits = self.head(x_final)
        return logits
