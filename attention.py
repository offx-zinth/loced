"""Hybrid sparse attention for LoCED (pure JAX, reference scale).

Combined good parts:

  * HySparse2 - token-level top-k selection; FORCED local window always
    selected from the SAME shared KV (no separate SWA branch); the FULL
    layers are the oracle indexer during training (their exact attention
    scores pick the selection the REUSE layers consume); NoPE.
  * CSA2 - FULL / REINDEX / REUSE operating modes over the shared store;
    indexer K projected FROM the main KV entries (`entry @ W_ik`);
    hierarchical candidate blocks (candidate pool restricting deeper
    indexers); every visit keeps its own Q.

Entry-space invariant: store entries (pool / delta / ring) all come from the
same bridge projections (W_bk, W_bv) applied to hidden states, and indexer
keys are always `entry @ W_ik`, so one indexer scores every source uniformly.

Selection budget: `L = window + top_k`; the window budget is force-boosted, so
the most recent `window` entries are always attended and `top_k` further
entries are selected outside it.

Candidate view contract (returned by assemble_train_view/assemble_store_view):
    rows_k/rows_v/rows_ik : [B, Tq, C, w]   candidate rows per query
    valid                 : [B, Tq, C]      causal + storage validity
    boost                 : [B, Tq, C]      1.0 on forced-window rows
    n_pool, n_win, n_own  : ints            segment sizes (pool|window|own)
    block_key             : [B, Tq, NB, idim] block keys (hierarchical indexer)
    block_last            : [B, Tq, NB]     latest covered pos per block (-1)
    block_id              : [P]             pool entry -> block id (static)
    positions             : [B, Tq, 1]      query positions

Implementation note: candidate rows are materialized per query. At reference
scale (tests, small models) this is fine; production training/inference should
fuse score -> top-k -> gather into Pallas kernels over query chunks, exactly
like MaxText attention kernels fuse dense attention.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import LoCEDConfig

NEG = -1e4


def init_layer_attn(cfg: LoCEDConfig, key: jax.Array) -> dict:
    """Attention compute of ONE physical layer (its own Q / exact-KV / out
    projections). Every layer runs this in FULL mode during encode and in
    `full`/`reuse`/`reindex` mode during decode depending on the pass."""
    assert cfg.kv_dim == cfg.head_dim, "MQA: one KV head of width head_dim"
    d, kv = cfg.hidden_size, cfg.kv_dim
    kd = cfg.num_attention_heads * cfg.head_dim

    def _w(shape, k):
        return jax.random.normal(k, shape, jnp.float32) * cfg.init_std

    k1, k2, k3 = jax.random.split(key, 3)
    return dict(
        W_q=_w((d, kd), k1),
        W_fx=_w((d, kv), k2), W_fv=_w((d, kv), k3),      # FULL exact self KV
        W_o=_w((kd, d), k1),
    )


def init_shared_attn(cfg: LoCEDConfig, key: jax.Array) -> dict:
    """Model-level attention state: ONE KV bridge and ONE indexer, because
    there is ONE shared store per sequence (writes happen at the stack exit
    of the pass that just ran)."""
    d, kv, idim = cfg.hidden_size, cfg.kv_dim, cfg.indexer_dim

    def _w(shape, k):
        return jax.random.normal(k, shape, jnp.float32) * cfg.init_std

    k4, k5, k6, k7 = jax.random.split(key, 4)
    return dict(
        W_bk=_w((d, kv), k4), W_bv=_w((d, kv), k5),      # bridge (pool/delta/ring)
        W_ik=_w((kv, idim), k6),                          # indexer K from KV
        W_iq=_w((d, cfg.indexer_heads * idim), k7),       # indexer query
    )


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------

def attend(q, K, V, valid_sel):
    """Sparse attention over gathered candidate rows.

    q: [B,Tq,H,qd]; K/V: [B,Tq,L,kv] (kv == qd, MQA shared KV head);
    valid_sel: [B,Tq,L]. Returns (out [B,Tq,H*qd], probs [B,Tq,H,L]).
    """
    scale = q.shape[-1] ** -0.5
    logits = jnp.einsum("bthd,btld->bhtl", q, K) * scale
    logits = logits + jnp.where(valid_sel[:, None, :, :], 0.0, NEG)
    probs = jax.nn.softmax(logits, axis=-1)          # [B,H,T,L]
    out = jnp.einsum("bhtl,btld->bthd", probs, V)
    # Public contract is [B,T,H,L] (head axis after the query axis), so the
    # diagnostics harness can broadcast a [B,T,1,L] segment mask directly.
    return (out.reshape(q.shape[0], q.shape[1], -1),
            jnp.transpose(probs, (0, 2, 1, 3)))


def gather_by_idx(rows, idx):
    """Per-query row gather.
    rows [B,C,w] (shared) or [B,Tq,C,w] (per-query); idx [B,Tq,L] -> [B,Tq,L,w].
    """
    if rows.ndim == 4:
        return jnp.take_along_axis(rows, idx[..., None], axis=2)
    oh = jax.nn.one_hot(idx, rows.shape[1], dtype=rows.dtype)
    return jnp.einsum("btlc,bcw->btlw", oh, rows)


def view_n_candidates(view: dict) -> int:
    """Candidate count for either a materialized or lazy train view."""
    if "n_candidates" in view:
        return int(view["n_candidates"])
    return int(view["rows_k"].shape[2])


def view_n_queries(view: dict) -> int:
    """Query count for either a materialized or lazy train view."""
    if "n_queries" in view:
        return int(view["n_queries"])
    return int(view["rows_k"].shape[1])


def _gather_segment(seg, idx, offset: int):
    """Gather one segment's rows for global candidate indices `idx`."""
    B, T, L = idx.shape
    if seg is None or seg.shape[-2] == 0:
        return jnp.zeros((B, T, L, 0 if seg is None else seg.shape[-1]),
                         jnp.float32 if seg is None else seg.dtype)
    S = seg.shape[-2]
    local = jnp.clip(idx - offset, 0, S - 1)
    if seg.ndim == 3:
        # Shared [B,S,w] segment: gather per batch to [B,Tq,L,w]. Do NOT
        # broadcast to [B,Tq,S,w] first; that reintroduces the pool-sized
        # allocation this lazy view is meant to remove.
        out = jnp.stack(
            [jnp.take(seg[b], local[b], axis=0) for b in range(B)], axis=0)
    else:
        out = jnp.take_along_axis(seg, local[..., None], axis=2)
    valid = (idx >= offset) & (idx < offset + S)
    return jnp.where(valid[..., None], out, jnp.zeros_like(out))


def gather_view(view: dict, idx, kind: str = "k"):
    """Gather selected rows from a materialized or lazy candidate view.

    `kind` is one of ``k``, ``v``, ``ik``. Lazy views keep the pool/delta rows
    shared and only the ring rows per-query, so this never allocates
    `[B,T,C,w]`; it gathers `[B,T,L,w]` from each segment and sums the
    disjoint segment masks.
    """
    if not view.get("lazy", False):
        key = {"k": "rows_k", "v": "rows_v", "ik": "rows_ik"}[kind]
        return gather_by_idx(view[key], idx)
    if kind == "k":
        segs = ((view["pool_k"], 0),
                (view.get("delta_k"), view["n_pool"]),
                (view["win_k"], view["n_pool"] + view["n_delta"]))
    elif kind == "v":
        segs = ((view["pool_v"], 0),
                (view.get("delta_v"), view["n_pool"]),
                (view["win_v"], view["n_pool"] + view["n_delta"]))
    else:
        segs = ((view["pool_ik"], 0),
                (view.get("delta_ik"), view["n_pool"]),
                (view["win_ik"], view["n_pool"] + view["n_delta"]))
    dtype = next(seg for seg, _ in segs if seg is not None).dtype
    out = jnp.zeros((idx.shape[0], idx.shape[1], idx.shape[2],
                     next(seg for seg, _ in segs if seg is not None).shape[-1]),
                    dtype)
    for seg, offset in segs:
        if seg is not None and seg.shape[-2] > 0:
            out = out + _gather_segment(seg, idx, offset)
    return out


def gather_selected(view: dict, idx):
    """Gather only the selected K/V rows from a candidate view."""
    return gather_view(view, idx, "k"), gather_view(view, idx, "v")


def indexer_scores_view(cfg: LoCEDConfig, p: dict, h, view: dict):
    """Indexer logits for a materialized or lazy candidate view.

    Lazy views score each segment independently; the result is the same
    concatenated `[B,Tq,C]` logits as scoring the materialized rows.
    """
    if not view.get("lazy", False):
        return indexer_logits(cfg, p, h, view["rows_ik"])
    Wq = p["W_iq"].reshape(cfg.hidden_size, cfg.indexer_heads,
                           cfg.indexer_dim)
    qi = jnp.einsum("btd,dhi->bthi", h, Wq)
    parts = []
    for seg in (view["pool_ik"], view.get("delta_ik"), view["win_ik"]):
        if seg is None or seg.shape[-2] == 0:
            continue
        if seg.ndim == 3:
            s = jnp.einsum("bthi,bsi->bhts", qi, seg)
        else:
            s = jnp.einsum("bthi,btci->bhtc", qi, seg)
        parts.append(s.mean(axis=1))
    return jnp.concatenate(parts, axis=2)


def iter_query_chunks(n_queries: int, chunk_size: int | None):
    """Static `(start, stop)` query chunks for training / prefill.

    `None` or a size >= n_queries yields one chunk, preserving the original
    full-query path.
    """
    if chunk_size is None or chunk_size <= 0 or chunk_size >= n_queries:
        yield (0, n_queries)
        return
    for start in range(0, n_queries, chunk_size):
        yield (start, min(start + chunk_size, n_queries))


def slice_view(view: dict, start: int, stop: int) -> dict:
    """Slice the per-query axis of a candidate view (axis 1).

    Segment metadata (`n_pool`, `n_delta`, `n_win`, `n_hier`) is query
    independent and is preserved. Shared pool/block arrays are not touched.
    """
    out = dict(view)
    n_q = view_n_queries(view)
    # Shared segment arrays are query-independent even when their length
    # happens to equal the query count (e.g. delta length == T in tiny runs).
    shared = {"pool_k", "pool_v", "pool_ik",
              "delta_k", "delta_v", "delta_ik",
              "n_pool", "n_delta", "n_win", "n_hier",
              "n_candidates", "n_queries", "lazy"}
    for key, value in view.items():
        if (hasattr(value, "ndim") and value.ndim >= 2
                and value.shape[1] == n_q and key not in shared):
            out[key] = value[:, start:stop]
    return out


def indexer_logits(cfg: LoCEDConfig, p: dict, h, cand_ik):
    """Indexer score: mean over indexer heads of q_i . k_i.

    h: [B,Tq,d] pre-attention hidden of the visit; cand_ik: [B,Tq,C,idim].
    Returns [B,Tq,C] logits.
    """
    Wq = p["W_iq"].reshape(cfg.hidden_size, cfg.indexer_heads, cfg.indexer_dim)
    qi = jnp.einsum("btd,dhi->bthi", h, Wq)
    s = jnp.einsum("bthi,btci->bhtc", qi, cand_ik)
    return s.mean(axis=1)


def select(scores, valid, boost, k):
    """Top-k with forced window: boost==1 entries are always in the set."""
    b = scores + jnp.where(valid, 0.0, NEG) + boost * 1e6
    return jax.lax.top_k(b, k)[1]                          # [B,Tq,k]


def trailing_rows(buf, window: int):
    """Per-query trailing window rows of a per-position buffer.

    buf: [B,T,w_]; query t gets rows [t-window+1 .. t] (front zero-padded).
    Returns rows [B,T,window,w_], src_pos [B,T,window], valid [B,T,window].
    """
    B, T, W = buf.shape
    pad = window - 1
    bp = jnp.concatenate([jnp.zeros((B, pad, W), buf.dtype), buf], axis=1)
    rows = jnp.stack([bp[:, i: i + T] for i in range(window)], axis=2)
    src_pos = (jnp.arange(T)[None, :, None] - pad
               + jnp.arange(window)[None, None, :])
    valid = jnp.broadcast_to(src_pos >= 0, (B, T, window))
    return rows, src_pos, valid


# ---------------------------------------------------------------------------
# FULL layers, exact world (the encode stack: exact causal attn = oracle)
# ---------------------------------------------------------------------------

def full_exact(cfg: LoCEDConfig, p: dict, h, positions):
    """Exact causal attention of one FULL layer. Returns (out, probs_mean)."""
    B, T, _ = h.shape
    k_st = h @ p["W_fx"].astype(h.dtype)                    # [B,T,kv]
    v_st = h @ p["W_fv"].astype(h.dtype)
    q = (h @ p["W_q"].astype(h.dtype)).reshape(
        B, T, cfg.num_attention_heads, cfg.head_dim)
    logits = jnp.einsum("bthd,bsd->bhts", q, k_st) * (cfg.head_dim ** -0.5)
    causal = positions[:, None, :] <= positions[:, :, None]   # [B,T,S]
    probs = jax.nn.softmax(
        logits + jnp.where(causal[:, None], 0.0, NEG), axis=-1)
    out = jnp.einsum("bhts,bsd->bthd", probs, v_st)
    return out.reshape(B, T, -1), probs.mean(axis=1)


def select_exact(cfg: LoCEDConfig, probs_mean, positions):
    """Oracle selection over exact positions: forced window + top-k."""
    T = probs_mean.shape[1]
    t_idx = positions[:, :, None]                           # [B,T,1]
    c_idx = positions[:, None, :]                           # [B,1,T]
    causal = c_idx <= t_idx
    in_window = causal & (c_idx > t_idx - cfg.window)
    valid = jnp.broadcast_to(causal, probs_mean.shape)
    boost = jnp.broadcast_to(in_window, probs_mean.shape).astype(jnp.float32)
    return select(probs_mean, valid, boost, min(cfg.window + cfg.top_k, T))


# ---------------------------------------------------------------------------
# pooled world: assembled candidate views (pass-2 and decode)
# ---------------------------------------------------------------------------

def n_pool_blocks(cfg: LoCEDConfig, n_pool: int) -> int:
    """Number of candidate blocks spanning `n_pool` pool entries.

    CEIL division: with `indexer_block` not dividing the pool length, the
    last block is partial, so floor division would leave the trailing
    entries without a block (their `block_id` would point past the block
    array and `jnp.take` would silently CLIP them onto the last block).
    """
    ib = cfg.indexer_block
    assert ib >= 1, "indexer_block must be >= 1"
    return max(1, (n_pool + ib - 1) // ib)


def _block_arrays(cfg: LoCEDConfig, pool_ik, pool_last, NB: int):
    """Block keys / stamps from pool entries (masked mean, static NB span).

    The pool axis is padded up to NB * indexer_block so that EVERY pool
    entry belongs to a block (partial last block included); padding rows are
    invalid (stamp -1) and contribute nothing to the block key.
    """
    B, P, idim = pool_ik.shape
    ib = cfg.indexer_block
    need = NB * ib
    if P < need:                                  # pad the partial last block
        span = jnp.concatenate(
            [pool_ik, jnp.zeros((B, need - P, idim), pool_ik.dtype)], axis=1)
        last = jnp.concatenate(
            [pool_last, jnp.full((B, need - P), -1, pool_last.dtype)], axis=1)
    else:                                         # truncate to the block grid
        span = pool_ik[:, :need]
        last = pool_last[:, :need]
    valid = (last >= 0).astype(span.dtype)
    key = (span * valid[..., None]).reshape(B, NB, ib, idim).sum(axis=2)
    cnt = valid.reshape(B, NB, ib).sum(axis=2)
    key = key / jnp.maximum(cnt, 1.0)[..., None]
    stamp = jnp.where(valid > 0, last, -1)
    blk_last = jnp.maximum(stamp.reshape(B, NB, ib).max(axis=2),
                           jnp.full((B, NB), -1, jnp.int32))
    return key, blk_last


def assemble_train_view(cfg: LoCEDConfig, pool_k, pool_v, pool_ik, pool_last,
                        buf_k, buf_v, buf_ik, positions,
                        block_key=None, block_last=None, n_hier=None,
                        delta_k=None, delta_v=None, delta_ik=None,
                        delta_pos=None, delta_query_start: int = 0,
                        lazy: bool = False):
    """Training candidate view over the bridged store (pass-2 visits).

    pool_*: [B,P,*] bridged pool entries; buf_*: [B,T,*] per-position bridge
    projections (trailing window rows). positions: [B,T]. `n_hier` leading
    pool entries are subject to hierarchical block restriction.

    Canonical segment order is `pool | delta | ring` (the same order as
    `assemble_store_view`). The optional delta rows are the first archived
    m=1 entries of the simulated generation span; they become valid at
    `delta_query_start` (the first generated query position) and for positions
    `delta_pos <= t`. Without delta arguments the function keeps the original
    pool + window layout.
    """
    B, T, _ = buf_k.shape
    P = pool_k.shape[1]
    n_hier = P if n_hier is None else n_hier
    pool_rows_k = jnp.broadcast_to(pool_k[:, None], (B, T, P, pool_k.shape[2]))
    pool_rows_v = jnp.broadcast_to(pool_v[:, None], (B, T, P, pool_v.shape[2]))
    pool_rows_i = jnp.broadcast_to(pool_ik[:, None], (B, T, P, pool_ik.shape[2]))
    win_k, src_pos, win_valid = trailing_rows(buf_k, cfg.window)
    win_v, _, _ = trailing_rows(buf_v, cfg.window)
    win_i, _, _ = trailing_rows(buf_ik, cfg.window)
    valid_pool = (pool_last >= 0)[:, None, :] & \
                 (pool_last[:, None, :] <= positions[:, :, None] - 1)

    parts_k = [pool_rows_k]
    parts_v = [pool_rows_v]
    parts_i = [pool_rows_i]
    parts_valid = [valid_pool]
    parts_boost = [jnp.zeros((B, T, P), jnp.float32)]
    D = 0
    delta_src = jnp.zeros((B, T, 0), jnp.int32)
    if delta_k is not None:
        if delta_pos is None:
            raise ValueError("delta_pos is required with delta_k")
        D = delta_k.shape[1]
        if delta_v is None or delta_ik is None:
            raise ValueError("delta_v and delta_ik are required with delta_k")
        dk = jnp.broadcast_to(delta_k[:, None], (B, T, D, delta_k.shape[2]))
        dv = jnp.broadcast_to(delta_v[:, None], (B, T, D, delta_v.shape[2]))
        di = jnp.broadcast_to(delta_ik[:, None], (B, T, D, delta_ik.shape[2]))
        dpos = jnp.broadcast_to(jnp.asarray(delta_pos, jnp.int32)[None, None, :],
                                (B, T, D))
        # Archived rows are exactly the positions that have left the ring:
        # ring covers [t-window+1 .. t], delta covers the older tail.
        valid_delta = ((positions[:, :, None] >= delta_query_start)
                       & (dpos <= positions[:, :, None] - cfg.window)
                       & (dpos >= 0))
        parts_k.append(dk)
        parts_v.append(dv)
        parts_i.append(di)
        parts_valid.append(valid_delta)
        parts_boost.append(jnp.zeros((B, T, D), jnp.float32))
        delta_src = dpos

    parts_k.append(win_k)
    parts_v.append(win_v)
    parts_i.append(win_i)
    parts_valid.append(win_valid)
    parts_boost.append(win_valid.astype(jnp.float32))
    valid = jnp.concatenate(parts_valid, axis=2)
    boost = jnp.concatenate(parts_boost, axis=2)

    if block_key is None:
        NB = n_pool_blocks(cfg, P)
        bkey, blast = _block_arrays(cfg, pool_ik, pool_last, NB)
    else:
        bkey, blast = block_key, block_last
    NB = bkey.shape[1]
    block_key = jnp.broadcast_to(bkey[:, None], (B, T, NB, bkey.shape[2]))
    block_last = jnp.broadcast_to(blast[:, None], (B, T, NB))
    out = dict(
        valid=valid, boost=boost, n_pool=P, n_delta=D, n_own=D,
        n_hier=n_hier, n_win=cfg.window,
        n_candidates=P + D + cfg.window, n_queries=T, lazy=lazy,
        block_key=block_key, block_last=block_last,
        block_id=jnp.arange(P, dtype=jnp.int32) // cfg.indexer_block,
        src_pos=src_pos, delta_src_pos=delta_src,
        positions=positions[:, :, None])
    if lazy:
        out.update(
            pool_k=pool_k, pool_v=pool_v, pool_ik=pool_ik,
            delta_k=delta_k, delta_v=delta_v, delta_ik=delta_ik,
            win_k=win_k, win_v=win_v, win_ik=win_i)
    else:
        out.update(
            rows_k=jnp.concatenate(parts_k, axis=2),
            rows_v=jnp.concatenate(parts_v, axis=2),
            rows_ik=jnp.concatenate(parts_i, axis=2))
    return out


def assemble_store_view(cfg: LoCEDConfig, store: dict, pos_q,
                        lazy: bool = False):
    """Decode candidate view: pool + delta + ring, per query position.

    pos_q: scalar (one shared position), [B] (per-sequence position), or
    [B,Tq] (one position per query, e.g. a whole extended turn). Traced
    values are fine (jitted decode / generate).

    Causality (mirrors the training view):
      * pool entries valid iff 0 <= pool_last <= t - 1 (the window covers t),
      * delta entries valid iff 0 <= delta_last <= t,
      * ring entries valid iff 0 <= ring_pos <= t; only valid ring rows are
        boosted (the forced local window).
    All rows [B,Tq,C,w]; only `n_hier` leading pool entries are block-restricted.
    """
    P = store["pool_k"].shape[1]
    B = store["pool_k"].shape[0]
    pq = jnp.asarray(pos_q)
    if pq.ndim == 0:
        pq1 = jnp.broadcast_to(pq[None, None], (B, 1))       # [B,1] or [1,1]
    elif pq.ndim == 1:
        pq1 = pq[:, None]
    else:
        pq1 = pq
    Tq = pq1.shape[1]

    def rows(x):
        return jnp.broadcast_to(x[:, None], (B, Tq) + x.shape[1:])

    if not lazy:
        rows_k = jnp.concatenate(
            [rows(store["pool_k"]), rows(store["delta_k"]),
             rows(store["ring_k"])], axis=2)
        rows_v = jnp.concatenate(
            [rows(store["pool_v"]), rows(store["delta_v"]),
             rows(store["ring_v"])], axis=2)
        rows_ik = jnp.concatenate(
            [rows(store["pool_ik"]), rows(store["delta_ik"]),
             rows(store["ring_ik"])], axis=2)
    tq = pq1[:, :, None]                                     # [B,Tq,1]
    pool_last = store["pool_last"][:, None, :]               # [B,1,P]
    delta_last = store["delta_last"][:, None, :]
    ring_pos = store["ring_pos"][:, None, :]
    valid_pool = (pool_last >= 0) & (pool_last <= tq - 1)
    valid_delta = (delta_last >= 0) & (delta_last <= tq)
    valid_ring = (ring_pos >= 0) & (ring_pos <= tq)
    valid = jnp.concatenate([valid_pool, valid_delta, valid_ring], axis=2)
    boost = jnp.concatenate([
        jnp.zeros_like(valid_pool, jnp.float32),
        jnp.zeros_like(valid_delta, jnp.float32),
        valid_ring.astype(jnp.float32)], axis=2)
    block_key = jnp.broadcast_to(
        store["block_key"][:, None], (B, Tq) + store["block_key"].shape[1:])
    block_last = jnp.broadcast_to(
        store["block_last"][:, None], (B, Tq) + store["block_last"].shape[1:])
    out = dict(
        valid=valid, boost=boost, n_pool=P,
        n_delta=store["delta_k"].shape[1], n_hier=P,
        n_win=store["ring_k"].shape[1], n_own=store["delta_k"].shape[1],
        n_candidates=P + store["delta_k"].shape[1] + store["ring_k"].shape[1],
        n_queries=Tq, lazy=lazy,
        block_key=block_key, block_last=block_last,
        block_id=jnp.arange(P, dtype=jnp.int32) // cfg.indexer_block,
        positions=tq)
    if lazy:
        out.update(
            pool_k=store["pool_k"], pool_v=store["pool_v"],
            pool_ik=store["pool_ik"], delta_k=store["delta_k"],
            delta_v=store["delta_v"], delta_ik=store["delta_ik"],
            win_k=store["ring_k"], win_v=store["ring_v"],
            win_ik=store["ring_ik"])
    else:
        out.update(rows_k=rows_k, rows_v=rows_v, rows_ik=rows_ik)
    return out


def oracle_scores(cfg: LoCEDConfig, view: dict, probs_mean):
    """Oracle (exact-attention) score for every candidate row of a train view.

    Pool part: block-max aggregation over `pool_m` groups, excluding every
    position that is represented exactly by the delta or ring segments
    (non-overlapping union: a position is either in an exact row or in a pool
    group). Delta part: exact probs at the delta source positions (valid rows
    only). Ring part: exact probs at the trailing-window source positions.
    Returns [B,T,C] probabilities in [0,1] in canonical `pool|delta|ring`
    order; callers apply validity masks.
    """
    B, T, _ = probs_mean.shape
    m = cfg.pool_m
    assert T % m == 0, "train length must be a multiple of pool_m"
    P = view["n_pool"]
    D = view.get("n_delta", 0)
    gm = T // m
    pos_grid = jnp.arange(T, dtype=jnp.int32)[None, None, :]        # [1,1,T]

    # Exact rows: ring (always present) + valid delta rows.
    ring_src = jnp.broadcast_to(view["src_pos"], (B, T, cfg.window))
    if D:
        delta_src = jnp.broadcast_to(view["delta_src_pos"], (B, T, D))
        delta_valid = view["valid"][:, :, P:P + D]
        delta_src = jnp.where(delta_valid, delta_src, -1)
    else:
        delta_src = jnp.zeros((B, T, 0), jnp.int32)
    exact_src = jnp.concatenate([ring_src, delta_src.astype(jnp.int32)],
                                axis=2)
    exact = jnp.zeros((B, T, T), bool)
    for i in range(exact_src.shape[2]):
        p = exact_src[:, :, i]
        exact = exact | ((pos_grid == p[:, :, None]) & (p[:, :, None] >= 0))

    prob_ex = jnp.where(~exact, probs_mean, -1.0)
    grouped = prob_ex[:, :, : min(gm, P) * m].reshape(
        B, T, min(gm, P), m).max(axis=3)
    grouped = jnp.maximum(grouped, 0.0)
    if gm < P:
        grouped = jnp.concatenate(
            [grouped, jnp.zeros((B, T, P - gm), probs_mean.dtype)], axis=2)
    elif gm > P:
        grouped = grouped[:, :, :P]

    if D:
        dpos = jnp.broadcast_to(view["delta_src_pos"], (B, T, D))
        delta = jnp.take_along_axis(
            probs_mean, jnp.clip(dpos, 0, T - 1), axis=2)
        delta = jnp.where(view["valid"][:, :, P:P + D], delta, 0.0)
    else:
        delta = jnp.zeros((B, T, 0), probs_mean.dtype)
    pos = jnp.broadcast_to(view["src_pos"], (B, T, cfg.window))
    win = jnp.take_along_axis(
        probs_mean, jnp.clip(pos, 0, T - 1), axis=2)
    win = jnp.where(pos >= 0, win, 0.0)
    return jnp.concatenate([grouped, delta, win], axis=2)


def block_entry_mask(cfg: LoCEDConfig, p: dict, h, view: dict,
                     top_blocks: int) -> jax.Array:
    """Hierarchical candidate restriction: top-BB blocks -> pool entry mask.

    Scores blocks with the indexer q of the visit h against block keys, keeps
    the best `top_blocks` causally-valid blocks, returns [B,Tq,n_hier] bool
    over the block-restricted pool entries (True = may be considered).
    """
    n_hier = view["n_hier"]
    bs = indexer_logits(cfg, p, h, view["block_key"])      # [B,Tq,NB]
    pos = view["positions"]                                 # [B,Tq,1]
    bvalid = (view["block_last"] >= 0) & (view["block_last"] <= pos - 1)
    BB = min(top_blocks, bs.shape[-1])
    sel = select(bs, bvalid, jnp.zeros((), jnp.float32), BB)   # [B,Tq,BB]
    oh = jax.nn.one_hot(sel, bs.shape[-1], dtype=jnp.float32).sum(axis=2)
    entry_block = view["block_id"][:n_hier]                 # [n_hier]
    entry_mask = jnp.take(oh, entry_block, axis=2)          # [B,Tq,n_hier]
    return entry_mask > 0.5


def reindex_valid(cfg: LoCEDConfig, p: dict, h, view: dict) -> jax.Array:
    """Valid mask for a REINDEX visit: leading pool entries restricted."""
    n_hier = view["n_hier"]
    v = view["valid"]
    em = block_entry_mask(cfg, p, h, view, cfg.hier_blocks)
    pool_part = v[:, :, :n_hier] & em
    return jnp.concatenate([pool_part, v[:, :, n_hier:]], axis=2)


def kl_distill(oracle, logits, valid) -> jax.Array:
    """KL(oracle || softmax(indexer)) over the gated candidates.

    oracle/logits: [B,Tq,C]; valid: [B,Tq,C]. The oracle distribution is
    renormalized over the valid candidates (zeros elsewhere). The distill
    loss touches only the indexer projections through `logits`.
    """
    t = jnp.where(valid, jnp.maximum(oracle, 0.0), 0.0)
    t = t / jnp.maximum(t.sum(-1, keepdims=True), 1e-9)
    logp = jax.nn.log_softmax(
        logits + jnp.where(valid, 0.0, NEG), axis=-1)
    kl = jnp.where(t > 0, t * (jnp.log(t + 1e-9) - logp), 0.0)
    return kl.sum(-1).mean()
