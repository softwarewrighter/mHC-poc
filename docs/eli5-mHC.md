# ELI5: mHC (Manifold-Constrained Hyper-Connections)

> For a simpler explanation, see [eli4-mHC.md](eli4-mHC.md).

This repo demonstrates **mHC**, a way to make deep Transformers train more stably by modifying only the
**residual connection**.

## The problem (why HC can get unstable)

Transformers rely on residual connections (“skip connections”) so signals and gradients can flow through
many layers.

**Hyper-Connections (HC)** makes residuals more expressive by splitting the residual into **S parallel streams**
(think “lanes”) and mixing those lanes between layers.

That mixing can help capacity — but it can also break the magic trick that makes deep nets stable: the residual
path should behave *approximately like an identity mapping* across depth.

If the residual mixing slightly amplifies the signal in each layer, then after many layers:
- activations can explode
- gradients can explode
- training can become unstable (NaNs / loss spikes)

## What mHC does (one sentence)

**mHC forces the residual mixing to be a “fair mixer”** so it can’t create or destroy signal mass as depth grows.

## The core update equation

We maintain `S` parallel residual streams. At layer `l` the update is:

```
x_{l+1} = H_res(l) * x_l  +  H_post(l)^T * F( H_pre(l) * x_l )
```

Where:
- `x_l` is the multi-stream hidden state (per token), shape `[S, D]`
- `F(·)` is a standard Transformer block (attention + MLP)
- `H_pre`, `H_post`, `H_res` are learned `S×S` mixing matrices

## The “manifold constraint” (the important part)

mHC constrains `H_res` to be **doubly stochastic**:
- all entries are non-negative
- each **row** sums to 1
- each **column** sums to 1

Why this helps:
- each output stream is a convex combination of input streams
- it prevents runaway amplification/attenuation on the residual path
- composing many layers stays well-behaved

## How we enforce it (Sinkhorn-Knopp projection)

We don’t learn `H_res` directly.
We learn `H_res_logits` and turn it into a valid doubly-stochastic matrix using **Sinkhorn iterations**:

1) Make it positive: `A = exp(H_res_logits)`
2) Repeat K times:
   - normalize rows of `A` to sum to 1
   - normalize columns of `A` to sum to 1
3) Use `A` as `H_res`

In this repo:
- default streams: `S=4`
- default Sinkhorn iterations: `K=20` (configurable)

## How THIS repo implements mHC (important specifics)

### What we keep identical across variants
- same token embedding, positional encoding, attention, MLP, optimizer, dataset
- only the residual path differs

### Variants
1) **Baseline**: standard residual (effectively `S=1`)
2) **HC**: multi-stream residual with *unconstrained* `H_res` (can amplify/attenuate)
3) **mHC**: same as HC, but `H_res` is Sinkhorn-projected to be doubly stochastic every forward pass

### How the multi-stream flows through the Transformer block
This repo uses a minimal, demo-friendly routing:
- `H_pre` mixes streams
- we then **merge streams** (mean across streams) to get a single `[D]` vector per token
- we run the standard Transformer block `F(·)` on that merged representation
- we **broadcast** back to `S` streams and mix with `H_post^T`

This keeps the demo small and isolates the effect of residual mixing.

## What to look for in the plots

The primary demo is a **depth stress test** (12/24/48 layers):
- HC tends to show larger grad norm spikes and sometimes NaNs at higher depth
- mHC stays stable longer under the same depth/LR

Measured outputs:
- loss vs step
- grad norm vs step
- NaN/Inf events
- gain proxy (how “amplifying” the residual mixing is across depth)
