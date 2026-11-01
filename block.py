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


def sinkhorn(logits, iters: int, eps: float = 1e-6):
    """Project a matrix onto the Birkhoff polytope (doubly stochastic).

    This is the mHC manifold constraint: every row and column sums to one, so
    the residual mixing matrix has spectral norm <= 1 and cannot amplify the
    residual stream. Works on the last two axes, so dynamic per-token
    coefficients are supported too.
    """
    h = jax.nn.softmax(logits, axis=-1)
    for _ in range(iters):
        h = h / (h.sum(axis=-2, keepdims=True) + eps)
        h = h / (h.sum(axis=-1, keepdims=True) + eps)
    return h


def init_layer(cfg: LoCEDConfig, key: jax.Array) -> dict:
    """ONE physical layer's own parameters: pre-norms + MoE FFN.

    (Attention compute is merged in by `init_model` via `init_layer_attn`.)
    """
    d = cfg.hidden_size
    out = dict(
        norm_a=jnp.ones((d,), jnp.float32),
        norm_m=jnp.ones((d,), jnp.float32),
        **init_moe(cfg, key),
    )
    if cfg.magic_norm:
        out["norm_magic"] = jnp.ones((d,), jnp.float32)
    return out


def init_loop(cfg: LoCEDConfig, key: jax.Array, num_visits: int = None) -> dict:
    """Model-level visit parameters: loop conditioning, gates, final norm.

    `num_visits` = len(decode_layers) = 4 layers x 2 passes (default 8).
    """
    NV = cfg.num_visits if num_visits is None else num_visits
    d = cfg.hidden_size
    out = dict(
        cond=jax.random.normal(key, (NV, d), jnp.float32) * cfg.init_std,
        gate_a=jnp.zeros((NV,), jnp.float32),               # sigmoid gates
        gate_m=jnp.zeros((NV,), jnp.float32),
        norm_f=jnp.ones((d,), jnp.float32),
    )
    if cfg.mhc_streams > 1:
        n = cfg.mhc_streams
        # Diagonal-biased logits -> Sinkhorn starts close to identity.
        res = jnp.zeros((NV, n, n), jnp.float32)
        res = res.at[:, jnp.arange(n), jnp.arange(n)].set(1.0)
        # Break stream symmetry: if H_pre/H_post were uniform and all streams
        # started identical, H_res (row-stochastic) would keep them identical
        # forever and mHC would collapse to a single stream.
        bias = jnp.linspace(-0.2, 0.2, n, dtype=jnp.float32)
        out.update(
            mhc_res=res,
            mhc_pre=jnp.broadcast_to(bias, (NV, n)),
            mhc_post=jnp.broadcast_to(-bias, (NV, n)),
        )
        if cfg.mhc_dynamic:
            out["mhc_dyn"] = jnp.zeros((NV, d, n * n + 2 * n), jnp.float32)
    return out


def _mhc_coefficients(cfg: LoCEDConfig, params: dict, x, visit: int):
    """Return (H_res, H_pre, H_post) for one visit.

    Static mode uses learnable per-visit logits. Dynamic mode predicts
    per-token offsets from the mean stream (the single-pass coefficient idea),
    then applies the same Birkhoff/softmax constraints.
    """
    n = cfg.mhc_streams
    res = params["mhc_res"][visit]
    pre = params["mhc_pre"][visit]
    post = params["mhc_post"][visit]
    if cfg.mhc_dynamic:
        c = x.mean(axis=2) @ params["mhc_dyn"][visit]
        res = res[None, None] + c[..., : n * n].reshape(c.shape[0], c.shape[1], n, n)
        pre = pre[None, None] + c[..., n * n: n * n + n]
        post = post[None, None] + c[..., n * n + n:]
    return (sinkhorn(res, cfg.mhc_sinkhorn_iters),
            jax.nn.softmax(pre, axis=-1),
            jax.nn.softmax(post, axis=-1))


def cell_visit_mhc(cfg: LoCEDConfig, params: dict, pc: dict, x, visit: int,
                   attn_fn):
    """One visit with mHC multi-stream residuals.

    x: [B,T,n,d]. The sublayer still sees one [B,T,d] feature stream
    (H_pre @ x); the update is injected back into all n streams through
    H_post, and H_res mixes the streams. H_res is projected onto the
    Birkhoff polytope by Sinkhorn, so residual propagation is non-expansive.
    """
    H_res, H_pre, H_post = _mhc_coefficients(cfg, params, x, visit)
    if H_pre.ndim == 1:
        pre = jnp.einsum("i,btid->btd", H_pre, x)
    else:
        pre = jnp.einsum("bti,btid->btd", H_pre, x)

    h = rmsnorm(cfg, pre + params["cond"][visit].astype(pre.dtype), pc["norm_a"])
    attn_out, ainfo = attn_fn(h)
    update = jax.nn.sigmoid(params["gate_a"][visit]) * (attn_out @ pc["W_o"])
    hm = rmsnorm(cfg, pre + update, pc["norm_m"])
    mout, maux = moe_forward(cfg, pc, hm)
    update = update + jax.nn.sigmoid(params["gate_m"][visit]) * mout

    if H_res.ndim == 2:
        mixed = jnp.einsum("ij,btjd->btid", H_res, x)
        x_new = mixed + H_post[None, None, :, None] * update[:, :, None, :]
    else:
        mixed = jnp.einsum("btij,btjd->btid", H_res, x)
        x_new = mixed + H_post[..., None] * update[:, :, None, :]
    if cfg.magic_norm:
        x_new = rmsnorm(cfg, x_new, pc["norm_magic"])
    return x_new, h, ainfo, maux


def cell_visit(cfg: LoCEDConfig, params: dict, pc: dict, x, visit: int,
               attn_fn):
    """One visit of ONE physical layer (`pc` = its own parameter dict).

    x:        [B,T,d] residual state entering the visit, or [B,T,n,d] when
              mHC is enabled.
    visit:    static visit index (loop-conditioning + gate selection; the
              layer index is implied by `pc`).
    attn_fn:  callable (h_normed [B,T,d]) -> (attention output [B,T,H*qd],
              info dict). The caller closes over the visit's mode / view.

    Returns (x_new, h_normed, attn_info, moe_aux): h_normed is the attention
    normed hidden (the caller reuses it as this visit's indexer query), and
    moe_aux is the router load-balance scalar of this visit.
    """
    if cfg.mhc_streams > 1:
        return cell_visit_mhc(cfg, params, pc, x, visit, attn_fn)

    h = rmsnorm(cfg, x + params["cond"][visit].astype(x.dtype), pc["norm_a"])
    attn_out, ainfo = attn_fn(h)
    x = x + jax.nn.sigmoid(params["gate_a"][visit]) * (attn_out @ pc["W_o"])
    hm = rmsnorm(cfg, x, pc["norm_m"])
    mout, maux = moe_forward(cfg, pc, hm)
    x = x + jax.nn.sigmoid(params["gate_m"][visit]) * mout
    if cfg.magic_norm:
        x = rmsnorm(cfg, x, pc["norm_magic"])
    return x, h, ainfo, maux
