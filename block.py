"""The LoCED stack: 4 physical layers, each run in FULL mode for encode and
revisited once per decode pass (see model.py for the schedule).

  * 4 physical layers = 4 x {attention compute + MoE + norms}. The layers are
    weight-tied ACROSS ROLES, not across depth: the decoder owns no weights of
    its own. Every layer serves both passes:
        pass A (= the encode stack): layer k runs in FULL mode at visit k
        pass B (sparse)            : layer k runs again at visit V + k with
                                     mode (REUSE, REUSE, REINDEX, REINDEX)[k]
  * Per-visit identity (8 visits = 4 layers x 2 passes) comes from
    `cond[v]`, `gate_a[v]`, `gate_m[v]`.
  * Model-level (outside the stack): embeddings, head, the final norm, the
    single KV bridge and the single indexer (the store is shared by all
    layers), so no physical layer carries parameters that are never used.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import LoCEDConfig
from .fnn import init_moe, moe_forward


def rmsnorm(cfg: LoCEDConfig, x, w):
    var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    x = x * jax.lax.rsqrt(var + cfg.rms_norm_eps)
    return (x * w.astype(x.dtype)).astype(x.dtype)


def init_layer(cfg: LoCEDConfig, key: jax.Array) -> dict:
    """ONE physical layer's own parameters: pre-norms + MoE FFN.

    (Attention compute is merged in by `init_model` via `init_layer_attn`.)
    """
    d = cfg.hidden_size
    return dict(
        norm_a=jnp.ones((d,), jnp.float32),
        norm_m=jnp.ones((d,), jnp.float32),
        **init_moe(cfg, key),
    )


def init_loop(cfg: LoCEDConfig, key: jax.Array, num_visits: int = None) -> dict:
    """Model-level visit parameters: loop conditioning, gates, final norm.

    `num_visits` = len(decode_layers) = 4 layers x 2 passes (default 8).
    """
    NV = cfg.num_visits if num_visits is None else num_visits
    d = cfg.hidden_size
    return dict(
        cond=jax.random.normal(key, (NV, d), jnp.float32) * cfg.init_std,
        gate_a=jnp.zeros((NV,), jnp.float32),               # sigmoid gates
        gate_m=jnp.zeros((NV,), jnp.float32),
        norm_f=jnp.ones((d,), jnp.float32),
    )


def cell_visit(cfg: LoCEDConfig, params: dict, pc: dict, x, visit: int,
               attn_fn):
    """One visit of ONE physical layer (`pc` = its own parameter dict).

    x:        [B,T,d] residual state entering the visit.
    visit:    static visit index (loop-conditioning + gate selection; the
              layer index is implied by `pc`).
    attn_fn:  callable (h_normed [B,T,d]) -> (attention output [B,T,H*qd],
              info dict). The caller closes over the visit's mode / view.

    Returns (x_new, h_normed, attn_info, moe_aux): h_normed is the attention
    normed hidden (the caller reuses it as this visit's indexer query), and
    moe_aux is the router load-balance scalar of this visit.
    """
    h = rmsnorm(cfg, x + params["cond"][visit].astype(x.dtype), pc["norm_a"])
    attn_out, ainfo = attn_fn(h)
    x = x + jax.nn.sigmoid(params["gate_a"][visit]) * (attn_out @ pc["W_o"])
    hm = rmsnorm(cfg, x, pc["norm_m"])
    mout, maux = moe_forward(cfg, pc, hm)
    x = x + jax.nn.sigmoid(params["gate_m"][visit]) * mout
    return x, h, ainfo, maux
