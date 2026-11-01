"""LoCED model: 4 physical layers, reused as encoder and as decoder.

Schedule (the single source of truth is `cfg.encode_layers` /
`cfg.decode_layers`):

  * PREFILL / ENCODE -- one pass over the 4 physical layers, no loop:
        encode_layers = (full, full, full, full)
    Every prompt token runs the stack with exact causal attention (the FULL
    mode doubles as the oracle indexer). The exit residual H_out is bridged
    into the shared KV store (pool_m:1 pool + exact window ring) and the
    stack exits early.

  * DECODE -- exactly TWO forward passes over the SAME 4 physical layers:
        pass A = encode_layers          (full x4) -- the ENC itself
        pass B = decode_layers[4:]      (reuse, reuse, reindex, reindex)
    The decoder owns no weights: layer k of pass A and layer k of pass B are
    the same parameters (visit v runs on layer `v % 4`). Pass A processes the
    generated token, writes its own KV entry (delta m=1 + ring shift) and the
    anchor scan picks rows with the indexer. Pass B re-reads that selection
    (2 x REUSE, own Q each) and re-scores the block-restricted candidate pool
    (2 x REINDEX). 4 layers x 2 passes = 8 layer visits per token.

The first generated token's logits come from the same decode path (pass B
only: pass A already ran for every prompt token during prefill).

Parameter layout:
  * `params["cells"]`          -- tuple of 4 per-layer dicts (attention
                                  compute + norms + MoE): the physical stack.
  * `embed`, `head`, `norm_f`, `cond[v]`, `gate_a[v]`, `gate_m[v]`,
    `W_bk/W_bv/W_ik/W_iq`      -- model-level: one shared store, one bridge,
                                  one indexer, 8 per-visit identities.
The indexer is trained only by distillation against the oracle (block-max)
targets at the scoring hiddens (stop-gradient isolated from the main KV path).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .attention import (assemble_store_view, assemble_train_view, attend,
                        full_exact, gather_by_idx, gather_view,
                        indexer_scores_view, init_layer_attn,
                        init_shared_attn, iter_query_chunks, kl_distill,
                        oracle_scores, reindex_valid, select, slice_view,
                        view_n_candidates)
from .block import cell_visit, init_layer, init_loop, rmsnorm
from .cache import append_delta, build_pool, new_store, pool_kv, seed_ring
from .config import LoCEDConfig


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------

def init_model(cfg: LoCEDConfig, key: jax.Array) -> dict:
    """One flat pytree: 4 physical layers + model-level shared parameters."""
    assert cfg.kv_dim == cfg.head_dim, "MQA invariant: kv_dim == head_dim"
    assert cfg.max_seq_len % cfg.pool_m == 0, "max_seq_len must be a multiple of pool_m"
    V = cfg.num_layers
    assert len(cfg.decode_layers) == 2 * V, \
        "decode = pass A + pass B over the same physical layers"
    assert tuple(cfg.decode_layers[:V]) == tuple(cfg.encode_layers), \
        "the first decode layers ARE the encode layers"
    assert len(cfg.decode_layers[V:]) == V, \
        "pass B visits every physical layer exactly once"

    d = cfg.hidden_size
    ks = jax.random.split(key, 4 + V)
    params = dict(
        embed=jax.random.normal(ks[0], (cfg.vocab_size, d), jnp.float32) * cfg.init_std,
        head=jax.random.normal(ks[1], (d, cfg.vocab_size), jnp.float32) * cfg.init_std,
    )
    params.update(init_shared_attn(cfg, ks[2]))          # W_bk, W_bv, W_ik, W_iq
    params.update(init_loop(cfg, ks[3], num_visits=cfg.num_visits))
    params["cells"] = tuple(
        {**init_layer(cfg, ks[4 + k]), **init_layer_attn(cfg, ks[4 + k])}
        for k in range(V))
    return params


def count_params(params) -> int:
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


# ---------------------------------------------------------------------------
# shared visit primitives
# ---------------------------------------------------------------------------

def _streams(cfg, x):
    """Lift [B,T,d] to [B,T,n,d] when mHC is enabled."""
    if cfg.mhc_streams == 1 or x.ndim == 4:
        return x
    return jnp.broadcast_to(
        x[..., None, :], x.shape[:2] + (cfg.mhc_streams, x.shape[-1]))


def _merge(cfg, x):
    """Collapse mHC streams to the single feature stream used by the head,
    bridge, indexer, and KV store."""
    return x if x.ndim == 3 else jnp.mean(x, axis=2)


def _q(cfg, pc, h):
    B, T, _ = h.shape
    return (h @ pc["W_q"]).reshape(B, T, cfg.num_attention_heads, cfg.head_dim)


def _head(cfg, p, x):
    return rmsnorm(cfg, _merge(cfg, x), p["norm_f"]) @ p["head"]


def _ce(logits, targets):
    logp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.take_along_axis(logp, targets[..., None], axis=-1).mean()


def _L(cfg, view):
    return min(view_n_candidates(view), cfg.window + cfg.top_k)


def _gather(view, idx):
    return gather_view(view, idx, "k"), gather_view(view, idx, "v")


def _gather_valid(valid, idx):
    return gather_by_idx(valid[..., None].astype(jnp.float32), idx)[..., 0] > 0.5


def _bridge(cfg, params, H):
    """KV Bridging: hidden states -> shared store entries (k, v, indexer-K)."""
    H = _merge(cfg, H)
    k = H @ params["W_bk"]
    v = H @ params["W_bv"]
    ik = jax.lax.stop_gradient(k) @ params["W_ik"]
    return k, v, ik


def _tail(buf, n: int, pad):
    """Last `n` rows of a [B,T,*] buffer (front-padded with `pad` if T < n)."""
    T = buf.shape[1]
    if T >= n:
        return buf[:, -n:]
    fill = jnp.full((buf.shape[0], n - T) + buf.shape[2:], pad, buf.dtype)
    return jnp.concatenate([fill, buf], axis=1)


# ---------------------------------------------------------------------------
# layer bodies: one physical layer, three operating modes
# ---------------------------------------------------------------------------

def _run_full(cfg, params, pc, x, v, positions):
    """FULL layer: exact causal attention (the oracle at training time)."""
    def attn_fn(h):
        out, probs = full_exact(cfg, pc, h, positions)
        return out, dict(probs=probs)
    return cell_visit(cfg, params, pc, x, v, attn_fn)


def _run_reuse(cfg, params, pc, x, v, Kg, Vg, vg):
    """REUSE layer: same rows / indices as the anchor scoring visit, own Q."""
    def attn_fn(h):
        out, probs = attend(_q(cfg, pc, h), Kg, Vg, vg)
        return out, dict(probs=probs)
    return cell_visit(cfg, params, pc, x, v, attn_fn)


def _run_reindex(cfg, params, pc, x, v, view):
    """REINDEX layer: block-restricted rescoring of the candidate pool."""
    L = _L(cfg, view)

    def attn_fn(h):
        hs = jax.lax.stop_gradient(h)          # indexer distill isolation
        valid2 = reindex_valid(cfg, params, hs, view)
        lg = indexer_scores_view(cfg, params, hs, view)
        idx2 = select(lg, valid2, view["boost"], L)
        Kg, Vg = _gather(view, idx2)
        vg = _gather_valid(valid2, idx2)
        out, probs = attend(_q(cfg, pc, h), Kg, Vg, vg)
        return out, dict(lg=lg, valid2=valid2, idx2=idx2, Kg=Kg, Vg=Vg, vg=vg,
                         probs=probs)

    return cell_visit(cfg, params, pc, x, v, attn_fn)


# ---------------------------------------------------------------------------
# the two decode passes (shared by train / prefill tail / decode_step)
# ---------------------------------------------------------------------------

def _encode_pass(cfg, params, x, positions):
    """Pass A: the encode stack (all FULL, one visit per physical layer).

    Returns (x, info_of_last_layer, moe_aux_sum).
    """
    info = dict()
    maux = jnp.asarray(0.0, jnp.float32)
    x = _streams(cfg, x)
    for k, mode in enumerate(cfg.encode_layers):
        if mode != "full":
            raise ValueError(f"encode layer {k} is '{mode}', expected 'full'")
        x, _, info, m = _run_full(cfg, params, params["cells"][k], x, k, positions)
        maux = maux + m
    return x, info, maux


def _decode_pass(cfg, params, x, Kg, Vg, vg, view, oracle=None,
                 anchor_idx=None, collect=False):
    """Pass B: the sparse decode modes, one visit per physical layer.

    Visit v = V + k runs on layer k with mode `sparse_layers[k]`. `oracle`
    (training) distills the indexer at every scoring visit.
    Returns (x, moe_aux_sum, kl, n_kl_terms, visit_infos). `visit_infos` is
    empty unless `collect=True`; it is used by the diagnostics harness.
    """
    V = cfg.num_layers
    maux = jnp.asarray(0.0, jnp.float32)
    kls = []
    infos = []
    x = _streams(cfg, x)
    for k, mode in enumerate(cfg.sparse_layers):
        v = V + k
        pc = params["cells"][k]
        if mode == "reuse":
            x, _, info, m = _run_reuse(cfg, params, pc, x, v, Kg, Vg, vg)
            if collect:
                infos.append(dict(visit=v, mode=mode, idx=anchor_idx,
                                  probs=info["probs"]))
        elif mode == "reindex":
            x, _, info, m = _run_reindex(cfg, params, pc, x, v, view)
            if oracle is not None:
                kls.append(kl_distill(oracle, info["lg"], info["valid2"]))
            if collect:
                infos.append(dict(visit=v, mode=mode, idx=info["idx2"],
                                  probs=info["probs"]))
            Kg, Vg, vg = info["Kg"], info["Vg"], info["vg"]
        else:
            raise ValueError(f"decode layer {k} is '{mode}'")
        maux = maux + m
    kl = jnp.mean(jnp.stack(kls)) if kls else jnp.asarray(0.0, jnp.float32)
    return x, maux, kl, len(kls), infos


def _anchor_select(cfg, params, x, view):
    """Inference anchor: indexer scores at the exit hidden pick the rows the
    REUSE layers inherit (top-k + forced window)."""
    lg = indexer_scores_view(cfg, params, _merge(cfg, x), view)
    idx = select(lg, view["valid"], view["boost"], _L(cfg, view))
    Kg, Vg = _gather(view, idx)
    vg = _gather_valid(view["valid"], idx)
    return Kg, Vg, vg


# ---------------------------------------------------------------------------
# training forward (teacher forcing)
# ---------------------------------------------------------------------------

def _forward_train_impl(cfg: LoCEDConfig, params: dict, tokens,
                        diagnostics: bool = False,
                        query_chunk_size: int | None = None,
                        prefill_len: int | None = None,
                        reuse_p: float = 0.0):
    """Teacher-forced forward with optional raw intermediates for diagnostics.

    The default `forward_train` wrapper returns exactly the previous contract;
    `forward_train_diagnostics` exposes the anchor scoring hidden, candidate
    view, oracle/indexer selections, per-visit attention probabilities and the
    pooled store rows. The extra fields are never part of the loss.
    """
    B, T = tokens.shape
    assert T % cfg.pool_m == 0, "train length must be a multiple of pool_m"
    positions = jnp.broadcast_to(jnp.arange(T, dtype=jnp.int32)[None, :], (B, T))
    x = params["embed"][tokens]

    # pass A: encode stack (FULL) -> the oracle probs + the exit residual
    H_out, info_a, maux_a = _encode_pass(cfg, params, x, positions)
    probs_mean = info_a["probs"]

    # KV bridging: prompt pool + simulated generated delta + trailing window.
    pk, pv, pik = _bridge(cfg, params, H_out)
    m = cfg.pool_m
    if prefill_len is None and cfg.train_delta:
        prefill_len = int(round(T * cfg.train_prefill_frac))
    if prefill_len is not None:
        prefill_len = (int(prefill_len) // m) * m
    use_delta = (cfg.train_delta and prefill_len is not None
                 and m <= prefill_len < T)
    if use_delta:
        Tp = int(prefill_len)
        P = Tp // m
        pool_k, pool_v = pool_kv(cfg, pk[:, :Tp], pv[:, :Tp])
        pool_last = positions[:, m - 1: Tp: m]
        delta_start = max(0, Tp - cfg.window)
        D = min(cfg.max_decode_len, T - delta_start)
        if D > 0:
            delta_pos = jnp.arange(delta_start, delta_start + D,
                                   dtype=jnp.int32)
            delta_k = pk[:, delta_pos]
            delta_v = pv[:, delta_pos]
            delta_ik = pik[:, delta_pos]
        else:
            delta_pos = delta_k = delta_v = delta_ik = None
        pool_ik = jax.lax.stop_gradient(pool_k) @ params["W_ik"]
        view = assemble_train_view(
            cfg, pool_k, pool_v, pool_ik, pool_last, pk, pv, pik, positions,
            delta_k=delta_k, delta_v=delta_v, delta_ik=delta_ik,
            delta_pos=delta_pos, delta_query_start=Tp,
            lazy=cfg.lazy_view)
    else:
        P = T // m
        pool_k, pool_v = pool_kv(cfg, pk, pv)
        pool_ik = jax.lax.stop_gradient(pool_k) @ params["W_ik"]
        pool_last = positions[:, m - 1: P * m: m]
        view = assemble_train_view(cfg, pool_k, pool_v, pool_ik, pool_last,
                                   pk, pv, pik, positions, lazy=cfg.lazy_view)
    oracle = oracle_scores(cfg, view, probs_mean)

    # anchor selection: the exact attention picks the oracle rows; on-policy
    # REUSE mixes in the indexer-selected rows (1.3) so training matches the
    # inference selector. `reuse_p=0` keeps the original oracle path.
    idx_a = select(oracle, view["valid"], view["boost"], _L(cfg, view))
    lg_a = indexer_scores_view(cfg, params,
                               jax.lax.stop_gradient(_merge(cfg, H_out)),
                               view)
    kl_a = kl_distill(oracle, lg_a, view["valid"])
    idx_i = (select(lg_a, view["valid"], view["boost"], _L(cfg, view))
             if (diagnostics or cfg.reuse_onpolicy) else None)
    if cfg.reuse_onpolicy and idx_i is not None:
        # Deterministic per-query mix: hash the position to [0, 1000) and
        # compare against reuse_p. p=0 -> oracle, p=1 -> indexer everywhere.
        phase = ((positions * 37 + 17) % 1000).astype(jnp.float32)
        use_idx = phase[..., None] < (jnp.asarray(reuse_p, jnp.float32) * 1000.0)
        idx_train = jnp.where(use_idx, idx_i, idx_a)
    else:
        idx_train = idx_a
    Kg, Vg = _gather(view, idx_train)
    vg = _gather_valid(view["valid"], idx_train)

    # pass B: reuse, reuse, reindex, reindex (mirrors inference)
    if query_chunk_size is None or query_chunk_size >= T:
        x2, maux_b, kl_b, _, infos = _decode_pass(
            cfg, params, H_out, Kg, Vg, vg, view, oracle=oracle,
            anchor_idx=idx_train, collect=diagnostics)
    else:
        if diagnostics:
            raise ValueError("diagnostics=True currently requires a full-query pass")
        # Pass B is query-local: every query owns its candidate rows, so the
        # four visits can run independently per query chunk. The candidate
        # view is still built once here; the lazy-scoring path (M3.1) removes
        # that allocation.
        x2_parts = []
        infos = []
        maux_b = jnp.asarray(0.0, jnp.float32)
        kl_num = jnp.asarray(0.0, jnp.float32)
        kl_den = 0
        for start, stop in iter_query_chunks(T, query_chunk_size):
            view_c = slice_view(view, start, stop)
            idx_c = idx_train[:, start:stop]
            Kg_c, Vg_c = _gather(view_c, idx_c)
            vg_c = _gather_valid(view_c["valid"], idx_c)
            xb, mb, klc, _, _ = _decode_pass(
                cfg, params, H_out[:, start:stop], Kg_c, Vg_c, vg_c, view_c,
                oracle=oracle[:, start:stop])
            x2_parts.append(xb)
            weight = stop - start
            maux_b = maux_b + mb * (weight / T)
            kl_num = kl_num + klc * weight
            kl_den += weight
        x2 = jnp.concatenate(x2_parts, axis=1)
        kl_b = kl_num / max(kl_den, 1)
    kl = 0.5 * (kl_a + kl_b)

    logits = _head(cfg, params, x2)
    aux_logits = _head(cfg, params, H_out)
    ce = _ce(logits[:, :-1], tokens[:, 1:])
    ce_aux = _ce(aux_logits[:, :-1], tokens[:, 1:])
    moe_aux = (maux_a + maux_b) / cfg.num_visits
    loss = (ce + cfg.aux_deep_weight * ce_aux
            + cfg.indexer_kl_coef * kl + cfg.moe_aux_weight * moe_aux)
    out = dict(loss=loss, ce=ce, ce_aux=ce_aux, kl=kl, moe_aux=moe_aux,
               logits=logits)
    if diagnostics:
        out.update(
            H_out=_merge(cfg, H_out), probs_mean=probs_mean, view=view,
            oracle=oracle,
            idx_oracle=idx_a, idx_indexer=idx_i, anchor_logits=lg_a,
            pool_k=pool_k, per_token_k=pk, pool_last=pool_last,
            visit_infos=infos)
    return out


def forward_train(cfg: LoCEDConfig, params: dict, tokens,
                  query_chunk_size: int | None = None,
                  prefill_len: int | None = None,
                  reuse_p: float = 0.0):
    """Train forward. Returns the same loss/logit contract as before."""
    out = _forward_train_impl(cfg, params, tokens, diagnostics=False,
                              query_chunk_size=query_chunk_size,
                              prefill_len=prefill_len, reuse_p=reuse_p)
    return {k: out[k] for k in ("loss", "ce", "ce_aux", "kl", "moe_aux",
                                "logits")}


def forward_train_diagnostics(cfg: LoCEDConfig, params: dict, tokens,
                              prefill_len: int | None = None,
                              reuse_p: float = 0.0):
    """Train forward plus raw selection/view/attention intermediates.

    Not intended for the jitted train step: the returned `visit_infos` is a
    Python list and the per-visit probabilities are kept for analysis.
    """
    return _forward_train_impl(cfg, params, tokens, diagnostics=True,
                               prefill_len=prefill_len, reuse_p=reuse_p)


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------

def prefill(cfg: LoCEDConfig, params: dict, tokens):
    """One prompt pass: encode stack -> bridge store -> pass B at the last
    position (first generated token's logits). Returns (store, [B,1,V])."""
    B, T = tokens.shape
    positions = jnp.broadcast_to(jnp.arange(T, dtype=jnp.int32)[None, :], (B, T))
    x = params["embed"][tokens]
    store = new_store(cfg, B, params["embed"].dtype)

    H_out, _, _ = _encode_pass(cfg, params, x, positions)
    store = build_pool(cfg, store, _merge(cfg, H_out), positions,
                       params["W_bk"], params["W_bv"], params["W_ik"])
    pk, pv, pik = _bridge(cfg, params, H_out)
    W = cfg.window
    store = seed_ring(store, _tail(positions, W, -1), _tail(pk, W, 0.0),
                      _tail(pv, W, 0.0), _tail(pik, W, 0.0))

    view = assemble_store_view(cfg, store, positions[:, -1],
                               lazy=cfg.lazy_view)
    Kg, Vg, vg = _anchor_select(cfg, params, H_out[:, -1:], view)
    x2, _, _, _, _ = _decode_pass(cfg, params, H_out[:, -1:], Kg, Vg, vg, view)
    return store, _head(cfg, params, x2)


def prefill_extend(cfg: LoCEDConfig, params: dict, store: dict, tokens,
                   pos0: int):
    """Multi-turn: append a new prompt turn; returns its per-token logits.

    The ring is reseeded with the last `window` exact rows of the whole
    sequence (old ring + new turn); exact rows pushed out remain available
    through the compressed pool."""
    B, T = tokens.shape
    positions = jnp.broadcast_to(
        (pos0 + jnp.arange(T, dtype=jnp.int32))[None, :], (B, T))
    x = params["embed"][tokens]

    H_out, _, _ = _encode_pass(cfg, params, x, positions)
    store = build_pool(cfg, store, _merge(cfg, H_out), positions,
                       params["W_bk"], params["W_bv"], params["W_ik"])
    pk, pv, pik = _bridge(cfg, params, H_out)
    W, n = cfg.window, min(T, cfg.window)
    rp = jnp.concatenate([store["ring_pos"], positions[:, T - n:]], axis=1)[:, -W:]
    rk = jnp.concatenate([store["ring_k"], pk[:, T - n:]], axis=1)[:, -W:]
    rv = jnp.concatenate([store["ring_v"], pv[:, T - n:]], axis=1)[:, -W:]
    ri = jnp.concatenate([store["ring_ik"], pik[:, T - n:]], axis=1)[:, -W:]
    store = seed_ring(store, rp, rk, rv, ri)

    view = assemble_store_view(cfg, store, positions, lazy=cfg.lazy_view)
    Kg, Vg, vg = _anchor_select(cfg, params, H_out, view)
    x2, _, _, _, _ = _decode_pass(cfg, params, H_out, Kg, Vg, vg, view)
    return store, _head(cfg, params, x2)


def decode_step(cfg: LoCEDConfig, params: dict, store: dict, token, pos: int):
    """One generated token: pass A (enc stack + own KV write + anchor select)
    then pass B (reuse, reuse, reindex, reindex). Returns (store, [B,1,V])."""
    B = token.shape[0]
    x = params["embed"][token][:, None, :]
    posv = jnp.asarray(pos, jnp.int32)
    positions = jnp.broadcast_to(posv[None, None], (B, 1))

    x, _, _ = _encode_pass(cfg, params, x, positions)
    pk, pv, pik = _bridge(cfg, params, x)
    store = append_delta(store, pk, pv, pik, posv)
    view = assemble_store_view(cfg, store, posv)
    Kg, Vg, vg = _anchor_select(cfg, params, x, view)
    x, _, _, _, _ = _decode_pass(cfg, params, x, Kg, Vg, vg, view)
    return store, _head(cfg, params, x)


def generate(cfg: LoCEDConfig, params: dict, prompt, max_new: int):
    """Greedy decode helper for smoke tests/evals."""
    store, logits = prefill(cfg, params, prompt)
    out = [jnp.argmax(logits[:, -1], axis=-1)]
    pos = prompt.shape[1]
    for i in range(max_new - 1):
        store, lg = decode_step(cfg, params, store, out[-1], pos + i)
        out.append(jnp.argmax(lg[:, -1], axis=-1))
    return jnp.stack(out, axis=1)
