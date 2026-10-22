"""LoCED KV store: the two-level shared cache (plain pytree, static shapes).

Layout (every entry carries `last_pos` = last sequence position it covers,
used for causal masking):

  * pool (KV Bridging)  : projected from the encode-stack exit residual H_out
                          (YOCO / CED / HySparse2 outer level).
                          Every pool_m prompt tokens are mean-pooled into one
                          entry (CSA2 compressor, no overlap). Multi-turn
                          prompts append at slots [old_n, old_n+P).
  * delta pool          : per generated token ONE entry written from the
                          decode pass-A exit hidden ("decoding layers have
                          their own cache"), m = 1
                          (CSA2 decoder compression ratio).
  * ring                : the exact entries of the last `window` tokens
                          (prompt tail + generated) = the forced local window.

Ring protocol: on append with a full ring, the OLDEST ring entry is archived
into the delta pool, everyone shifts, the new entry enters at the tail.

Hierarchical indexer: pool entries are grouped into candidate blocks of
`indexer_block` entries; each block carries a masked-mean indexer-K key and a
`block_last` causal stamp (CSA2: 2048 blocks x 8 positions).

Precision note: MXFP4 main-KV + FP4 indexer-K (DeepSeek-V4.1-Flash) is a
deployment-time format swap; the reference implementation keeps float32.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .attention import _block_arrays, n_pool_blocks


def _zeros(B: int, shape, dtype) -> jax.Array:
    return jnp.zeros((B, *shape), dtype)


def new_store(cfg, batch: int, dtype) -> dict:
    """Empty store. Capacities are static so `decode_step` can be jitted."""
    B = batch
    P = cfg.max_seq_len // cfg.pool_m                      # prompt-pool cap
    NB = n_pool_blocks(cfg, P)                             # candidate blocks
    D = cfg.max_decode_len                                 # delta-pool cap
    R = cfg.window                                         # ring = window
    return dict(
        pool_k=_zeros(B, (P, cfg.kv_dim), dtype),
        pool_v=_zeros(B, (P, cfg.kv_dim), dtype),
        pool_ik=_zeros(B, (P, cfg.indexer_dim), dtype),
        pool_last=jnp.full((B, P), -1, jnp.int32),
        pool_n=jnp.zeros((B,), jnp.int32),
        block_key=_zeros(B, (NB, cfg.indexer_dim), dtype),
        block_last=jnp.full((B, NB), -1, jnp.int32),
        delta_k=_zeros(B, (D, cfg.kv_dim), dtype),
        delta_v=_zeros(B, (D, cfg.kv_dim), dtype),
        delta_ik=_zeros(B, (D, cfg.indexer_dim), dtype),
        delta_last=jnp.full((B, D), -1, jnp.int32),
        delta_n=jnp.zeros((B,), jnp.int32),
        ring_k=_zeros(B, (R, cfg.kv_dim), dtype),
        ring_v=_zeros(B, (R, cfg.kv_dim), dtype),
        ring_ik=_zeros(B, (R, cfg.indexer_dim), dtype),
        ring_pos=jnp.full((B, R), -1, jnp.int32),
    )


def project_entries(cfg, H: jax.Array, Wk, Wv, Wz):
    """Project hidden states into main-KV entries; indexer K comes from KV."""
    k = H @ Wk.astype(H.dtype)                    # [.., kv_dim]
    v = H @ Wv.astype(H.dtype)
    return k, v, k @ Wz.astype(H.dtype)           # ik = KV @ Wz  (CSA2)


def build_pool(cfg, store: dict, H_out: jax.Array, pos: jax.Array,
               Wk, Wv, Wz) -> dict:
    """KV Bridging: pool_m consecutive H_out entries -> one pool entry.

    H_out: [B, T, d] exit residual of the encode stack; pos: [B, T] positions.
    New groups are APPENDED at slots [old_n, old_n + P) (multi-turn reuse);
    entry i of a group covers positions [.., i] => pool_last = pos[m-1::m].
    """
    B, T, _ = H_out.shape
    m = cfg.pool_m
    assert T % m == 0, "prefill length must be a multiple of pool_m"
    P = T // m
    k, v, _ = project_entries(cfg, H_out, Wk, Wv, Wz)
    gk = k[:, : P * m, :].reshape(B, P, m, -1).mean(axis=2)       # [B,P,kv]
    gv = v[:, : P * m, :].reshape(B, P, m, -1).mean(axis=2)
    group_last = pos[:, m - 1: P * m: m]                         # [B,P]
    bidx = jnp.arange(B)[:, None]
    old_n = store["pool_n"][0]                     # batched turns share length
    cap = store["pool_k"].shape[1]
    start = jnp.clip(old_n, 0, cap - P)
    slot = start + jnp.arange(P)[None, :]                        # [1,P]
    pool_k = store["pool_k"].at[bidx, slot].set(gk)
    pool_v = store["pool_v"].at[bidx, slot].set(gv)
    pool_last = store["pool_last"].at[bidx, slot].set(group_last)
    pool_ik = pool_k @ Wz.astype(H_out.dtype)
    pool_n = jnp.broadcast_to(jnp.clip(start + P, 0, cap), (B,))
    # hierarchical candidate blocks over the whole (multi-turn) pool
    NB = store["block_key"].shape[1]
    keys, block_last = _block_arrays(cfg, pool_ik, pool_last, NB)
    return dict(store) | dict(
        pool_k=pool_k, pool_v=pool_v, pool_ik=pool_ik, pool_last=pool_last,
        pool_n=pool_n, block_key=keys, block_last=block_last)


def seed_ring(store: dict, pos_last: jax.Array, k, v, ik) -> dict:
    """Fill the ring with the last `window` per-position exact entries.

    k/v/ik: [B, R, kv/(idim)] entries; pos_last: [B, R] positions (-1 = pad).
    """
    return dict(store) | dict(ring_k=k, ring_v=v, ring_ik=ik,
                              ring_pos=pos_last.astype(jnp.int32),)


def append_delta(store: dict, k, v, ik, pos: int) -> dict:
    """Anchor visit writes its OWN KV entry (m=1) into delta pool + ring.

    When the ring is already full, its oldest entry is archived into the delta
    pool first (so every exact entry eventually reaches the delta pool exactly
    once), then everything shifts and the new entry enters at the ring tail.
    """
    B = store["ring_k"].shape[0]
    D = store["delta_k"].shape[1]
    bidx = jnp.arange(B)
    n = store["delta_n"]
    live = (store["ring_pos"][:, 0] >= 0) & (n < D)   # archive oldest entry
    slot = jnp.clip(n, 0, D - 1)
    delta_k = store["delta_k"].at[bidx, slot].set(
        jnp.where(live[:, None], store["ring_k"][:, 0],
                  store["delta_k"][bidx, slot]))
    delta_v = store["delta_v"].at[bidx, slot].set(
        jnp.where(live[:, None], store["ring_v"][:, 0],
                  store["delta_v"][bidx, slot]))
    delta_ik = store["delta_ik"].at[bidx, slot].set(
        jnp.where(live[:, None], store["ring_ik"][:, 0],
                  store["delta_ik"][bidx, slot]))
    delta_last = store["delta_last"].at[bidx, slot].set(
        jnp.where(live, store["ring_pos"][:, 0],
                  store["delta_last"][bidx, slot]))
    delta_n = jnp.minimum(n + live.astype(jnp.int32), jnp.int32(D))
    ring_k = jnp.concatenate([store["ring_k"][:, 1:], k], axis=1)
    ring_v = jnp.concatenate([store["ring_v"][:, 1:], v], axis=1)
    ring_ik = jnp.concatenate([store["ring_ik"][:, 1:], ik], axis=1)
    new_pos = jnp.broadcast_to(jnp.asarray(pos, jnp.int32)[None, None], (B, 1))
    ring_pos = jnp.concatenate([store["ring_pos"][:, 1:], new_pos], axis=1)
    return dict(store) | dict(delta_k=delta_k, delta_v=delta_v,
                              delta_ik=delta_ik, delta_last=delta_last,
                              delta_n=delta_n, ring_k=ring_k, ring_v=ring_v,
                              ring_ik=ring_ik, ring_pos=ring_pos)
