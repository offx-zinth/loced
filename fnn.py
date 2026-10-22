"""Mixture-of-Experts FFN for LoCED (DeepSeek-style: shared + routed SwiGLU).

Token-choice router, top-`top_k_experts` routed experts, one always-on shared
expert. Every physical layer owns its own MoE weight set (the stack has 4);
the two decode passes revisit those same layer weights.
A Switch-style load-balance auxiliary loss is returned for the train step.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import LoCEDConfig


def init_moe(cfg: LoCEDConfig, key: jax.Array) -> dict:
    d, I, E = cfg.hidden_size, cfg.intermediate_size, cfg.num_experts

    def _w(shape, k):
        return jax.random.normal(k, shape, jnp.float32) * cfg.init_std

    ks = jax.random.split(key, 6)
    return dict(
        W_gate=_w((d, E), ks[0]),
        Ws1=_w((d, I), ks[1]), Ws3=_w((d, I), ks[1]), Ws2=_w((I, d), ks[2]),
        We1=_w((E, d, I), ks[3]), We3=_w((E, d, I), ks[4]),
        We2=_w((E, I, d), ks[5]),
    )


def _swiglu(x1, x3):
    return jax.nn.silu(x1) * x3


def moe_forward(cfg: LoCEDConfig, p: dict, x):
    """x: [B,T,d] -> (out [B,T,d], aux scalar)."""
    # shared expert
    shared = _swiglu(x @ p["Ws1"], x @ p["Ws3"]) @ p["Ws2"]
    # routed experts (dense per-expert matmul at reference scale)
    a = jnp.einsum("btd,edi->btei", x, p["We1"])
    g = jnp.einsum("btd,edi->btei", x, p["We3"])
    h = _swiglu(a, g)
    y = jnp.einsum("btei,eid->bted", h, p["We2"])          # [B,T,E,d]
    probs = jax.nn.softmax(x @ p["W_gate"], axis=-1)       # [B,T,E]
    k = cfg.top_k_experts
    top_idx = jax.lax.top_k(probs, k)[1]                   # [B,T,k]
    w = jnp.take_along_axis(probs, top_idx, axis=-1)
    w = w / jnp.maximum(w.sum(-1, keepdims=True), 1e-9)
    mask = jax.nn.one_hot(top_idx, cfg.num_experts, dtype=jnp.float32)
    mask = mask * w[..., None]                             # [B,T,k,E] -> sum k
    mask = mask.sum(axis=2)                                # [B,T,E]
    routed = jnp.einsum("bte,bteh->bth", mask, y)
    out = shared + routed
    # load balance: E * sum_e (fraction selected_e * mean prob_e)
    frac = jnp.mean((mask > 0).astype(jnp.float32), axis=(0, 1))      # [E]
    mean_p = jnp.mean(probs, axis=(0, 1))                             # [E]
    aux = cfg.num_experts * jnp.sum(frac * mean_p)
    return out, aux


def moe_param_names() -> tuple:
    return ("W_gate", "Ws1", "Ws3", "Ws2", "We1", "We3", "We2")
