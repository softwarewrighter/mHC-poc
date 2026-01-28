"""Synthetic dataset - PyTorch implementation.

Mirrors mlx/src/dataset.py for correctness comparison.
"""
from __future__ import annotations
import torch


class IncrementingTokenDataset:
    """Synthetic dataset: tokens follow (start + i) mod vocab.

    Each sample is length (seq_len + 1); we predict next token.
    This is learnable and trains quickly, good for stability stress tests.
    """

    def __init__(self, vocab_size: int, seq_len: int, device: torch.device = None):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a batch of incrementing sequences.

        Args:
            batch_size: number of sequences

        Returns:
            (x, y) where x is input [B, T] and y is target [B, T]
        """
        # Random starting tokens: [B, 1]
        starts = torch.randint(
            0, self.vocab_size, (batch_size, 1), device=self.device
        )
        # Indices: [1, L+1]
        idx = torch.arange(self.seq_len + 1, device=self.device).unsqueeze(0)
        # Full sequence: [B, L+1]
        seq = (starts + idx) % self.vocab_size
        # Split into input and target
        x = seq[:, :-1]  # [B, L]
        y = seq[:, 1:]   # [B, L]
        return x.long(), y.long()
