"""Muon optimizer for LoCED (JAX + optax, no extra dependencies).

Mirrors the `jax_mokre/muon.py` API already used by the MoRE runs:

    mask = make_muon_mask(params)
    opt  = muon_adamw(mask, ns_steps=3, ns_dtype=jnp.float32)

Muon (Keller Jordan et al.) = Shampoo-style steepest descent under the
spectral norm: keep an EMA of the gradient, orthogonalize it with a
Newton-Schulz iteration (approximating the orthogonal polar factor
`U V^T` of `M = U S V^T`), then take a step along that direction. The
orthogonalized direction has a flat singular spectrum, which is why it
generalizes better than raw momentum on matrix-shaped weights.

Mask policy (Kimi K2 / Moonlight convention):
  * Muon          -- hidden 2-D weight matrices (W_q, W_fx, W_fv, W_o, the
                     shared-expert SwiGLU matrices, the KV bridge, the indexer).
  * Muon per-expert -- stacked expert tensors (E, d, I): each expert slice is
                     orthogonalized independently (vmapped), never the stack.
  * AdamW         -- embeddings, the output head, the MoE router gate, and
                     every 1-D parameter (norms, visit gates, visit conditioning).

Scaling: a Muon update has per-element RMS `1/sqrt(max(rows, cols))` while an
AdamW update has per-element RMS ~1. `consistent_rms=True` (default) rescales
the orthogonalized matrix by `sqrt(max(rows, cols))` so that swapping the
optimizer does NOT silently change the effective learning rate. Set
`consistent_rms=False` for the Keller-Jordan `sqrt(max(1, rows/cols))` rule.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

# Newton-Schulz quintic coefficients (modded-nanogpt / Keller Jordan).
NS_COEFFS = (3.4445, -4.7750, 2.0315)

MUON, MUON_EXPERT, ADAMW = "muon", "muon_expert", "adamw"

# never Muon: embeddings, output head, router, visit conditioning.
_ADAMW_NAMES = {"embed", "head", "W_gate", "cond"}


# ---------------------------------------------------------------------------
# orthogonalization
# ---------------------------------------------------------------------------

def orthogonalize(g: jax.Array, steps: int = 5) -> jax.Array:
    """Approximate the orthogonal polar factor of a 2-D matrix.

    Runs `steps` Newton-Schulz iterations of the quintic
    `X <- a X + (b A + c A^2) X`, `A = X X^T`, which drives the singular
    values towards 1. The matrix is Frobenius-normalized first (so the
    spectral norm is <= 1, the domain where the coefficients are stable) and
    transposed when it is tall, so the iteration always runs on the short
    side.
    """
    a, b, c = NS_COEFFS
    X = g
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    X = X / (jnp.linalg.norm(X) + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


def _muon_direction(g: jax.Array, steps: int) -> jax.Array:
    """Orthogonalize one matrix, or a stack of matrices (per-expert)."""
    if g.ndim == 3:                       # (E, rows, cols): per-expert
        return jax.vmap(lambda e: orthogonalize(e, steps))(g)
    return orthogonalize(g, steps)


# ---------------------------------------------------------------------------
# mask
# ---------------------------------------------------------------------------

def _leaf_name(path) -> str:
    for key in reversed(path):
        name = getattr(key, "key", None)
        if isinstance(name, str):
            return name
    return ""


def make_muon_mask(params) -> dict:
    """Partition the parameter tree into {MUON, MUON_EXPERT, ADAMW} tags.

    The returned pytree mirrors `params` exactly, so it can be zipped with
    the parameter leaves inside the optimizer transform.
    """
    def tag(path, x):
        name = _leaf_name(path)
        if name in _ADAMW_NAMES:
            return ADAMW
        if x.ndim == 3:                    # stacked expert weights
            return MUON_EXPERT
        if x.ndim == 2:
            return MUON
        return ADAMW                       # norms, gates, scalars
    return jax.tree_util.tree_map_with_path(tag, params)


# ---------------------------------------------------------------------------
# transform
# ---------------------------------------------------------------------------

class _MuonAdamWState(NamedTuple):
    count: jax.Array
    mu: jax.Array
    m: jax.Array
    v: jax.Array


class _LeafOut(NamedTuple):
    """Per-leaf optimizer outputs (a named tuple so it can be transposed)."""
    mu: jax.Array
    m: jax.Array
    v: jax.Array
    u: jax.Array


def _split(out, i: int):
    """Unpack field `i` of every `_LeafOut` into a tree of arrays."""
    return jax.tree_util.tree_map(
        lambda t: t[i], out, is_leaf=lambda x: isinstance(x, _LeafOut))


def muon_adamw(mask: dict,
                learning_rate: float | jax.Array = 1e-3,
                ns_steps: int = 3,
                ns_dtype=jnp.float32,
                momentum: float = 0.95,
                nesterov: bool = True,
                adam_b1: float = 0.9,
                adam_b2: float = 0.95,
                adam_eps: float = 1e-8,
                weight_decay: float = 0.01,
                consistent_rms: bool = True):
    """Muon for matrix weights + AdamW for everything else.

    The returned transform applies `learning_rate` internally (the update is
    already scaled to AdamW's RMS when `consistent_rms=True`), so it can be
    chained after `optax.clip_by_global_norm`.
    """
    def init_fn(params):
        zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
        return _MuonAdamWState(
            count=jnp.zeros((), jnp.int32),
            mu=zeros, m=zeros, v=zeros)

    def update_fn(updates, state, params=None):
        count = state.count + 1
        params = params if params is not None else updates

        def leaf(tag, g, mu, m, v, p):
            if tag == ADAMW:
                m = adam_b1 * m + (1 - adam_b1) * g
                v = adam_b2 * v + (1 - adam_b2) * (g * g)
                mh = m / (1 - adam_b1 ** count)
                vh = v / (1 - adam_b2 ** count)
                u = mh / (jnp.sqrt(vh) + adam_eps)
                if weight_decay:
                    u = u + weight_decay * p
                return _LeafOut(mu, m, v, u)

            g32 = g.astype(ns_dtype)
            mu = momentum * mu + g32
            # NOTE: `mu` is the un-normalized momentum sum (buf = b*buf + g),
            # so nesterov uses `g + b*buf` with NO bias correction -- adding a
            # 1/(1-b^t) factor amplifies the very first step by ~1/(1-b) = 20x
            # and diverges.
            direction = g32 + momentum * mu if nesterov else mu
            u = _muon_direction(direction, ns_steps)
            rows, cols = u.shape[-2], u.shape[-1]
            if consistent_rms:
                scale = jnp.sqrt(jnp.maximum(rows, cols))
            else:
                scale = jnp.sqrt(jnp.maximum(1.0, rows / cols))
            u = (u * scale).astype(g.dtype)
            if weight_decay:
                u = u + weight_decay * p
            return _LeafOut(mu, m, v, u)

        out = jax.tree_util.tree_map(
            leaf, mask, updates, state.mu, state.m, state.v, params)
        mu, m, v = (_split(out, i) for i in range(3))
        # optax adds `updates` to `params`: descent is -lr * direction.
        u = jax.tree_util.tree_map(
            lambda x: -learning_rate * x, _split(out, 3))
        return u, _MuonAdamWState(count=count, mu=mu, m=m, v=v)

    return optax.GradientTransformation(init_fn, update_fn)


def muon_optimizer(mask: dict, learning_rate: float = 1e-3,
                   weight_decay: float = 0.01, clip_norm: float = 1.0,
                   ns_steps: int = 3, **kw) -> optax.GradientTransformation:
    """Full chain: global-norm clipping -> Muon/AdamW -> LR."""
    return optax.chain(
        optax.clip_by_global_norm(clip_norm),
        muon_adamw(mask, learning_rate=learning_rate,
                    weight_decay=weight_decay, ns_steps=ns_steps, **kw),
    )
