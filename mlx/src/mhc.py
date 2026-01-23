from __future__ import annotations
import mlx.core as mx
import mlx.nn as nn

def sinkhorn_doubly_stochastic(A: mx.array, iters: int = 20, eps: float = 1e-12) -> mx.array:
    """Project a positive matrix A to approximately doubly-stochastic using Sinkhorn-Knopp.

    A: [S, S], must be elementwise positive.
    Returns: [S, S] approximately doubly-stochastic.
    """
    X = A
    for _ in range(iters):
        # Row normalize
        row_sum = mx.sum(X, axis=1, keepdims=True)
        X = X / mx.maximum(row_sum, eps)
        # Col normalize
        col_sum = mx.sum(X, axis=0, keepdims=True)
        X = X / mx.maximum(col_sum, eps)
    return X

class StreamMix(nn.Module):
    """Learned stream mixing matrix H of shape [S,S].

    mode:
      - 'softplus' : nonnegative, unconstrained (can amplify)
      - 'softmax_row': row-stochastic (nonnegative, each row sums to 1)
      - 'doubly_stochastic': Sinkhorn-projected (mHC residual)
    """
    def __init__(self, S: int, mode: str, sinkhorn_iters: int = 20):
        super().__init__()
        self.S = S
        self.mode = mode
        self.sinkhorn_iters = sinkhorn_iters
        # logits/init near identity
        eye = mx.eye(S)
        self.logits = mx.array(2.0 * eye - 1.0)  # pushes diagonal up a bit

    def _positive(self) -> mx.array:
        # exp keeps strictly positive and smooth
        return mx.exp(self.logits)

    def matrix(self) -> mx.array:
        A = self._positive()
        if self.mode == "softplus":
            # exp already positive; no normalization
            return A
        if self.mode == "softmax_row":
            return mx.softmax(self.logits, axis=1)
        if self.mode == "doubly_stochastic":
            return sinkhorn_doubly_stochastic(A, iters=self.sinkhorn_iters)
        raise ValueError(f"unknown mode: {self.mode}")

def apply_stream_mix(H: mx.array, x: mx.array) -> mx.array:
    """Apply a stream-mix matrix H:[S,S] to x:[B,T,S,D] along the stream axis.
    Returns [B,T,S,D].
    """
    # x: [B,T,S,D] -> [B,T,D,S] for matmul convenience
    xt = mx.transpose(x, (0,1,3,2))  # [B,T,D,S]
    # (..,S) @ (S,S)^T  -> keep last dim as S
    yt = mx.matmul(xt, mx.transpose(H, (1,0)))  # [B,T,D,S]
    y = mx.transpose(yt, (0,1,3,2))  # [B,T,S,D]
    return y
