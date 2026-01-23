from __future__ import annotations
import math
import mlx.core as mx

def grad_global_norm(grads) -> float:
    # grads is a pytree-like structure from mlx.value_and_grad; flatten heuristically
    flat = []
    def collect(x):
        if isinstance(x, mx.array):
            flat.append(x)
        elif isinstance(x, (list, tuple)):
            for v in x: collect(v)
        elif isinstance(x, dict):
            for v in x.values(): collect(v)
    collect(grads)

    if not flat:
        return 0.0
    s = mx.array(0.0)
    for g in flat:
        s = s + mx.sum(g * g)
    return float(mx.sqrt(s).item())

def has_nan_or_inf(x: mx.array) -> bool:
    # mx.isfinite returns bool array
    finite = mx.all(mx.isfinite(x))
    return not bool(finite.item())

def gain_proxy_from_Hres(Hres_list) -> float:
    """A tiny proxy: multiply H_res matrices across depth and return log10 of max entry.

    This is NOT the paper's exact metric; it's a demo-friendly proxy for
    "does residual mixing amplify over depth?"
    """
    if not Hres_list:
        return 0.0
    P = Hres_list[0]
    for H in Hres_list[1:]:
        P = mx.matmul(H, P)
    m = float(mx.max(mx.abs(P)).item())
    return math.log10(max(m, 1e-12))
