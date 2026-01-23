from __future__ import annotations
import mlx.core as mx

class IncrementingTokenDataset:
    """Synthetic dataset: tokens follow (start + i) mod vocab.

    Each sample is length (seq_len + 1); we predict next token.
    This is learnable and trains quickly, good for stability stress tests.
    """
    def __init__(self, vocab_size: int, seq_len: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len

    def batch(self, batch_size: int) -> tuple[mx.array, mx.array]:
        # starts: [B,1]
        starts = mx.random.randint(0, self.vocab_size, shape=(batch_size, 1))
        idx = mx.arange(self.seq_len + 1)[None, :]  # [1, L+1]
        seq = (starts + idx) % self.vocab_size      # [B, L+1]
        x = seq[:, :-1]
        y = seq[:, 1:]
        return x.astype(mx.int32), y.astype(mx.int32)
