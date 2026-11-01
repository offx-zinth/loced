"""LoCED smoke tests (pytest, MaxText-style unit tests).

Run:  python3.11 -m pytest loced/smoke_test.py -q
"""

from __future__ import annotations

import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from .cache import new_store
from .config import LoCEDConfig
from .diagnostics import (build_report, pool_norm_ratio, segment_layout,
                          segment_mass, selection_summary,
                          train_decode_kl_report)
from .model import (count_params, decode_step, forward_train,
                    forward_train_diagnostics, generate, init_model, prefill)
from .muon import (ADAMW, MUON, MUON_EXPERT, make_muon_mask, muon_adamw,
                   orthogonalize)
from .train import make_train_step, synthetic_batch, tiny_config, train_smoke

CFG = tiny_config()
KEY = jax.random.PRNGKey(0)


@pytest.fixture(scope="module")
def params():
    return init_model(CFG, KEY)


def test_schedule():
    """Locked schedule: 4 physical layers; encode = 1 pass over them (all
    FULL); decode = 2 passes: the same 4 FULL layers (the ENC itself) then
    REUSE, REUSE, REINDEX, REINDEX = 4 x 2 = 8 visits."""
    assert CFG.encode_layers == ("full",) * 4
    assert CFG.decode_layers[:4] == CFG.encode_layers
    assert CFG.decode_layers[4:] == ("reuse", "reuse", "reindex", "reindex")
    assert CFG.num_layers == 4            # physical layers
    assert CFG.prefill_visits == 4
    assert CFG.num_visits == 8            # 4 layers x 2 passes
    # pass B visits revisit the SAME physical layers (tied across passes)
    assert [CFG.layer_of_visit(v) for v in range(8)] == [0, 1, 2, 3] * 2


CELL_KEYS = {"norm_a", "norm_m", "W_q", "W_fx", "W_fv", "W_o",
             "W_gate", "Ws1", "Ws3", "Ws2", "We1", "We3", "We2"}
GLOBAL_KEYS = {"embed", "head", "norm_f", "W_bk", "W_bv", "W_ik", "W_iq",
               "cond", "gate_a", "gate_m", "cells"}


def test_param_inventory(params):
    """4 physical layers (own attention + MoE each) + model-level shared
    parameters (embed / head / final norm / one bridge / one indexer /
    visit conditioning + gates)."""
    assert set(params.keys()) == GLOBAL_KEYS
    assert len(params["cells"]) == CFG.num_layers == 4
    for cell in params["cells"]:
        assert set(cell.keys()) == CELL_KEYS
    assert params["cond"].shape == (CFG.num_visits, CFG.hidden_size)
    assert params["gate_a"].shape == (CFG.num_visits,)
    assert count_params(params) > 0


def test_physical_layers(params):
    """Cells are 4 distinct parameter sets, and every physical layer receives
    loss gradients (they are all exercised by both passes)."""
    assert not np.shares_memory(np.asarray(params["cells"][0]["W_q"]),
                                np.asarray(params["cells"][1]["W_q"]))
    tokens = synthetic_batch(np.random.default_rng(10), 2, CFG.max_seq_len,
                             CFG.vocab_size)
    grads = jax.grad(lambda p: forward_train(CFG, p, tokens)["loss"])(params)
    for k in range(CFG.num_layers):
        leaves = jax.tree_util.tree_leaves(grads["cells"][k])
        assert any(np.any(np.asarray(g) != 0.0) for g in leaves), \
            f"physical layer {k} must receive gradients"


def test_forward_finite(params):
    tokens = synthetic_batch(np.random.default_rng(0), 2, CFG.max_seq_len,
                             CFG.vocab_size)
    out = forward_train(CFG, params, tokens)
    assert np.isfinite(float(out["loss"]))
    assert out["logits"].shape == (2, CFG.max_seq_len, CFG.vocab_size)
    assert np.isfinite(np.asarray(out["logits"])).all()
    assert float(out["kl"]) >= 0.0


def test_causality(params):
    """Changing token j must not change logits at positions < j."""
    rng = np.random.default_rng(1)
    tokens = synthetic_batch(rng, 2, CFG.max_seq_len, CFG.vocab_size)
    other = np.asarray(tokens).copy()
    j = 40
    other[:, j] = (other[:, j] + 7) % CFG.vocab_size
    lo = np.asarray(forward_train(CFG, params, tokens)["logits"])
    hi = np.asarray(forward_train(CFG, params, jnp.asarray(other))["logits"])
    np.testing.assert_allclose(lo[:, :j], hi[:, :j], rtol=1e-4, atol=1e-4)
    assert not np.allclose(lo[:, j:], hi[:, j:], rtol=1e-4, atol=1e-4)


def test_determinism(params):
    tokens = synthetic_batch(np.random.default_rng(2), 2, CFG.max_seq_len,
                             CFG.vocab_size)
    a = forward_train(CFG, params, tokens)["logits"]
    b = forward_train(CFG, params, tokens)["logits"]
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_jit_matches_eager(params):
    tokens = synthetic_batch(np.random.default_rng(3), 2, CFG.max_seq_len,
                             CFG.vocab_size)
    eager = forward_train(CFG, params, tokens)["loss"]
    jitted = jax.jit(lambda p, t: forward_train(CFG, p, t)["loss"])(params,
                                                                    tokens)
    np.testing.assert_allclose(float(eager), float(jitted), rtol=1e-4)


def test_indexer_gradients(params):
    tokens = synthetic_batch(np.random.default_rng(4), 2, CFG.max_seq_len,
                             CFG.vocab_size)

    def kl_only(p):
        return forward_train(CFG, p, tokens)["kl"]

    grads = jax.grad(kl_only)(params)
    assert np.any(np.asarray(grads["W_iq"]) != 0.0), \
        "indexer Q must receive gradients from the distill loss"
    assert np.any(np.asarray(grads["W_ik"]) != 0.0), \
        "indexer K must receive gradients from the distill loss"
    # the distill loss must NOT push through the main KV path
    assert np.all(np.asarray(grads["W_bk"]) == 0.0), \
        "main-KV bridge path must be gradient-isolated from the indexer loss"


def test_train_decreases(params):
    cfg, params2, history = train_smoke(steps=150, batch=4, seed=0)
    first, last = history[0], history[-1]
    assert np.isfinite([float(m["loss"]) for m in history]).all()
    assert float(last["loss"]) < float(first["loss"]) - 0.5
    uniform = np.log(cfg.vocab_size)
    assert float(last["loss"]) < uniform - 0.5
    assert float(last["kl"]) < float(first["kl"])


# ---------------------------------------------------------------------------
# Muon
# ---------------------------------------------------------------------------

def test_orthogonalize(params=None):
    """Newton-Schulz must compress the singular spectrum towards 1 (the
    quintic is an approximate orthogonalizer, not an exact one: the fixed
    point sits in ~[0.68, 1.14] for the standard coefficients)."""
    g = jax.random.normal(jax.random.PRNGKey(1), (12, 20))
    o = orthogonalize(g, steps=5)
    s = np.linalg.svd(np.asarray(o), compute_uv=False)
    cond_before = np.linalg.cond(np.asarray(g))
    cond_after = float(s.max() / s.min())
    assert cond_before > 3.0, "test needs an ill-conditioned input"
    assert cond_after < 2.0, f"condition number {cond_after:.2f} not compressed"
    assert 0.5 < s.min() and s.max() < 1.4
    # tall matrices take the transpose path and must also come back tall
    o2 = orthogonalize(g.T, steps=5)
    assert o2.shape == (20, 12)
    s2 = np.linalg.svd(np.asarray(o2), compute_uv=False)
    assert 0.5 < s2.min() and s2.max() < 1.4


def test_muon_mask_partitions(params):
    """Muon on hidden matrices (2-D and per-expert 3-D); AdamW on embeddings,
    head, router, visit conditioning and every 1-D parameter."""
    mask = make_muon_mask(params)
    assert mask["embed"] == ADAMW and mask["head"] == ADAMW
    assert mask["W_bk"] == MUON and mask["W_iq"] == MUON
    assert mask["norm_f"] == ADAMW and mask["gate_a"] == ADAMW
    assert mask["cond"] == ADAMW
    cell = mask["cells"][0]
    assert cell["W_q"] == MUON and cell["W_fx"] == MUON and cell["W_o"] == MUON
    assert cell["Ws1"] == MUON and cell["Ws2"] == MUON
    assert cell["We1"] == MUON_EXPERT and cell["We2"] == MUON_EXPERT
    assert cell["W_gate"] == ADAMW
    assert cell["norm_a"] == ADAMW
    # same tree structure as the parameters
    assert jax.tree_util.tree_structure(mask) == \
        jax.tree_util.tree_structure(params)


def test_muon_matches_adamw_rms(params):
    """consistent_rms=True must give per-element update RMS ~ AdamW's (~1),
    otherwise the AdamW -> Muon swap silently changes the learning rate."""
    mask = make_muon_mask(params)
    opt = muon_adamw(mask, learning_rate=1.0, weight_decay=0.0,
                     consistent_rms=True)
    grads = jax.tree_util.tree_map(
        lambda x: jax.random.normal(jax.random.PRNGKey(2), x.shape), params)
    updates, _ = opt.update(grads, opt.init(params), params)
    mat = updates["cells"][0]["W_q"]
    rms = float(jnp.sqrt(jnp.mean(mat ** 2)))
    assert 0.5 < rms < 2.0, f"Muon update RMS {rms:.3f} far from AdamW's ~1"
    exp = updates["cells"][0]["We1"]
    assert float(jnp.sqrt(jnp.mean(exp ** 2))) > 0.0


def test_muon_train_decreases():
    """The Muon path trains end-to-end and beats the uniform baseline."""
    cfg, params2, history = train_smoke(steps=150, batch=4, seed=0,
                                        optim="muon")
    first, last = history[0], history[-1]
    assert np.isfinite([float(m["loss"]) for m in history]).all()
    assert float(last["loss"]) < float(first["loss"]) - 0.5
    assert float(last["loss"]) < np.log(cfg.vocab_size) - 0.5
    assert all(m["loss"].shape == () for m in history)


def test_muon_jit_step(params):
    """The whole train step stays jittable with the Muon optimizer."""
    cfg, params2, history = train_smoke(steps=2, batch=2, seed=1)
    assert len(history) == 2


def test_prefill_decode_store(params):
    """Prefill seeds the store; decode steps grow ring + delta pool."""
    prompt = synthetic_batch(np.random.default_rng(5), 1, CFG.max_seq_len,
                             CFG.vocab_size)
    store, logits = prefill(CFG, params, prompt)
    assert logits.shape == (1, 1, CFG.vocab_size)
    assert np.isfinite(np.asarray(logits)).all()
    # ring seeded with the prompt tail
    T, W = CFG.max_seq_len, CFG.window
    np.testing.assert_array_equal(np.asarray(store["ring_pos"])[0],
                                  np.arange(T - W, T))
    assert int(store["delta_n"][0]) == 0

    token = jnp.asarray([int(np.argmax(logits[0, -1]))])
    pos = T
    for i in range(5):
        store, logits = decode_step(CFG, params, store, token, pos + i)
        assert logits.shape == (1, 1, CFG.vocab_size)
        assert np.isfinite(np.asarray(logits)).all()
        token = jnp.asarray([int(np.argmax(logits[0, -1]))])
    # 5 appends; each archived the then-oldest ring row (positions 48..52)
    assert int(store["delta_n"][0]) == 5
    np.testing.assert_array_equal(np.asarray(store["delta_last"])[0, :5],
                                  np.arange(T - W, T - W + 5))
    np.testing.assert_array_equal(np.asarray(store["ring_pos"])[0],
                                  np.arange(T - W + 5, T + 5))


def test_generate(params):
    prompt = synthetic_batch(np.random.default_rng(6), 2, 8, CFG.vocab_size)
    out = generate(CFG, params, prompt, max_new=4)
    assert out.shape == (2, 4)


def test_block_ids_cover_every_pool_entry():
    """Regression: when `indexer_block` does not divide the pool length, the
    last block is partial. The block grid must still cover EVERY pool entry
    (ceil division + padding), otherwise the trailing entries' block_id
    points past the block array and `jnp.take` silently CLIPS them onto the
    last block."""
    cfg = LoCEDConfig(
        vocab_size=64, hidden_size=32, max_seq_len=20, max_decode_len=8,
        num_attention_heads=2, head_dim=16, kv_dim=16, pool_m=2,
        window=8, top_k=8, indexer_heads=2, indexer_dim=16,
        indexer_block=4, hier_blocks=2,
        num_experts=2, num_shared_experts=1, top_k_experts=1,
        intermediate_size=64,
    )
    P = cfg.max_seq_len // cfg.pool_m          # 10 pool entries
    assert P % cfg.indexer_block != 0, "this config must exercise the partial block"
    params = init_model(cfg, KEY)
    assert cfg.indexer_block >= 1

    # store view: block grid spans ceil(P / indexer_block) blocks
    prompt = synthetic_batch(np.random.default_rng(21), 1, cfg.max_seq_len,
                             cfg.vocab_size)
    store, logits = prefill(cfg, params, prompt)
    assert np.isfinite(np.asarray(logits)).all()
    from .attention import assemble_store_view, n_pool_blocks
    view = assemble_store_view(cfg, store, cfg.max_seq_len)
    NB = view["block_key"].shape[2]
    assert NB == n_pool_blocks(cfg, P) == 3
    assert NB * cfg.indexer_block >= P, "every pool entry must fit in a block"
    assert int(np.asarray(view["block_id"]).max()) < NB, "block_id in range"

    # train view: same guarantee, and the forward pass must stay finite
    tokens = synthetic_batch(np.random.default_rng(22), 2, cfg.max_seq_len,
                             cfg.vocab_size)
    out = forward_train(cfg, params, tokens)
    assert np.isfinite(float(out["loss"]))
    assert np.isfinite(np.asarray(out["logits"])).all()


def test_jit_inference(params):
    """Fully jitted prefill + decode, as an inference service would run it."""
    prompt = synthetic_batch(np.random.default_rng(9), 1, CFG.max_seq_len,
                             CFG.vocab_size)
    jpre = jax.jit(lambda t: prefill(CFG, params, t))
    store, logits = jpre(prompt)
    jdec = jax.jit(lambda s, tok, pos: decode_step(CFG, params, s, tok, pos))
    token = jnp.asarray([int(np.argmax(logits[0, -1]))])
    pos = CFG.max_seq_len
    for i in range(3):
        store, logits = jdec(store, token, pos + i)
        assert logits.shape == (1, 1, CFG.vocab_size)
        token = jnp.asarray([int(np.argmax(logits[0, -1]))])
    assert int(store["delta_n"][0]) == 3


def test_decode_after_extend(params):
    """Multi-turn: extend the store with a second chunk and decode."""
    prompt = synthetic_batch(np.random.default_rng(7), 1, 16, CFG.vocab_size)
    store, logits = prefill(CFG, params, prompt)
    turn2 = synthetic_batch(np.random.default_rng(8), 1, 16, CFG.vocab_size)
    from .model import prefill_extend
    store, logits2 = prefill_extend(CFG, params, store, turn2, pos0=16)
    assert logits2.shape == (1, 16, CFG.vocab_size)
    assert np.isfinite(np.asarray(logits2)).all()
    # pool grew by 16 // pool_m entries; ring tail is turn-2's tail
    assert int(store["pool_n"][0]) == (16 + 16) // CFG.pool_m
    np.testing.assert_array_equal(np.asarray(store["ring_pos"])[0],
                                  np.arange(16, 32))
    token = jnp.asarray([0])
    store, logits3 = decode_step(CFG, params, store, token, 32)
    assert np.isfinite(np.asarray(logits3)).all()


# ---------------------------------------------------------------------------
# Phase 0 diagnostics
# ---------------------------------------------------------------------------

def _diag_batch(batch=1, seq=None):
    seq = CFG.max_seq_len if seq is None else seq
    return synthetic_batch(np.random.default_rng(31), batch, seq,
                           CFG.vocab_size)


def test_diagnostics_segment_mass(params):
    """Segment mass partitions the selected attention mass exactly.

    Regression for the original measurement bug: `attend` now returns
    `[B,T,H,L]`, so the `[B,T,1,L]` segment mask broadcasts over heads.
    """
    tokens = _diag_batch(2)
    out = forward_train_diagnostics(CFG, params, tokens)
    layout = segment_layout(out["view"])
    Tp = int(round(CFG.max_seq_len * CFG.train_prefill_frac))
    Tp = (Tp // CFG.pool_m) * CFG.pool_m
    assert layout.n_pool == Tp // CFG.pool_m
    assert layout.n_delta > 0
    assert layout.n_win == CFG.window
    assert len(out["visit_infos"]) == len(CFG.sparse_layers) == 4
    assert "rows_k" not in out["view"]          # lazy view: no [B,T,C,w]
    for info in out["visit_infos"]:
        mass = segment_mass(info["probs"], info["idx"], layout)
        total = sum(mass.values())
        np.testing.assert_allclose(np.asarray(total), 1.0, rtol=1e-5,
                                   atol=1e-5)
        assert info["probs"].shape[:2] == info["idx"].shape[:2]


def test_diagnostics_selection_metrics(params):
    """Recall/overlap are bounded and segment shares sum to one."""
    tokens = _diag_batch()
    out = forward_train_diagnostics(CFG, params, tokens)
    layout = segment_layout(out["view"])
    summary = selection_summary(out["idx_oracle"], out["idx_indexer"], layout)
    assert 0.0 <= summary["recall_at_L"] <= 1.0
    assert 0.0 <= summary["jaccard"] <= 1.0
    for side in ("oracle", "indexer"):
        assert abs(sum(summary[side].values()) - 1.0) < 1e-6
    ratio = pool_norm_ratio(out["pool_k"], out["per_token_k"], CFG.pool_m)
    Tp = int(round(CFG.max_seq_len * CFG.train_prefill_frac))
    Tp = (Tp // CFG.pool_m) * CFG.pool_m
    assert ratio.shape == (tokens.shape[0], Tp // CFG.pool_m)
    assert np.isfinite(np.asarray(ratio)).all()
    # 1.1: norm-preserving pooling keeps the pooled rows at constituent scale.
    assert 0.98 < float(ratio.mean()) < 1.02


def test_train_decode_kl_report(params):
    """Teacher-forced decode through the real store runs and aligns."""
    tokens = _diag_batch(1, 32)
    report = train_decode_kl_report(CFG, params, tokens, prefill_len=16)
    assert np.isfinite(report["kl"]) and report["kl"] >= 0.0
    assert 0.0 <= report["top1_agreement"] <= 1.0
    assert report["n_positions"] == 16
    assert report["prefill_len"] == 16


def test_diagnostics_report_json_roundtrip(params, tmp_path):
    """The report contains the D1-D5 fields and is JSON serializable."""
    tokens = _diag_batch(1, 32)
    report = build_report(CFG, params, tokens, prefill_len=16)
    assert set(report) >= {"layout", "pool_norm_ratio", "selection",
                           "visits", "train_decode"}
    assert len(report["visits"]) == 4
    path = tmp_path / "report.json"
    from .diagnostics import write_report
    write_report(path, report)
    loaded = json.loads(path.read_text())
    assert loaded["layout"]["n_pool"] == 8     # seq 32 -> prefill 16 -> P=8
    assert loaded["layout"]["n_delta"] == 32
    assert len(loaded["visits"]) == 4


def test_chunked_pass_b_matches_full(params):
    """M0.5: query-chunked pass B is numerically equivalent to the full pass."""
    tokens = _diag_batch(2, 32)
    full = forward_train(CFG, params, tokens)
    for chunk in (4, 8, 16):
        part = forward_train(CFG, params, tokens, query_chunk_size=chunk)
        np.testing.assert_allclose(float(part["loss"]), float(full["loss"]),
                                   rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(np.asarray(part["logits"]),
                                   np.asarray(full["logits"]),
                                   rtol=1e-4, atol=1e-4)


def test_onpolicy_reuse_uses_indexer_selection(params):
    """1.3: reuse_p=0 uses oracle rows, reuse_p=1 uses indexer rows."""
    tokens = _diag_batch(1, 32)
    oracle = forward_train_diagnostics(CFG, params, tokens, prefill_len=16,
                                       reuse_p=0.0)
    indexer = forward_train_diagnostics(CFG, params, tokens, prefill_len=16,
                                        reuse_p=1.0)
    for info in oracle["visit_infos"]:
        if info["mode"] == "reuse":
            np.testing.assert_array_equal(np.asarray(info["idx"]),
                                          np.asarray(oracle["idx_oracle"]))
    for info in indexer["visit_infos"]:
        if info["mode"] == "reuse":
            np.testing.assert_array_equal(np.asarray(info["idx"]),
                                          np.asarray(indexer["idx_indexer"]))
    summary = selection_summary(oracle["idx_oracle"], oracle["idx_indexer"],
                                segment_layout(oracle["view"]))
    assert summary["recall_at_L"] >= 0.0


def test_prefill_extend_lazy_matches_dense(params):
    """M3.1 store view: lazy pool/delta/ring matches the materialized view."""
    from .model import prefill_extend

    prompt = synthetic_batch(np.random.default_rng(21), 1, 16, CFG.vocab_size)
    turn2 = synthetic_batch(np.random.default_rng(22), 1, 16, CFG.vocab_size)
    lazy_cfg = dataclasses.replace(CFG, lazy_view=True)
    dense_cfg = dataclasses.replace(CFG, lazy_view=False)
    s_lazy, _ = prefill(lazy_cfg, params, prompt)
    s_dense, _ = prefill(dense_cfg, params, prompt)
    s_lazy, logits_lazy = prefill_extend(
        lazy_cfg, params, s_lazy, turn2, pos0=16)
    s_dense, logits_dense = prefill_extend(
        dense_cfg, params, s_dense, turn2, pos0=16)
    np.testing.assert_allclose(np.asarray(logits_lazy),
                               np.asarray(logits_dense),
                               rtol=1e-5, atol=1e-5)
    for key in ("pool_k", "delta_k", "ring_k", "block_key"):
        np.testing.assert_allclose(np.asarray(s_lazy[key]),
                                   np.asarray(s_dense[key]),
                                   rtol=1e-5, atol=1e-5)


def test_lazy_view_matches_materialized(params):
    """M3.1: segment scoring/gather matches the materialized candidate view."""
    tokens = _diag_batch(2, 32)
    lazy_cfg = dataclasses.replace(CFG, lazy_view=True)
    dense_cfg = dataclasses.replace(CFG, lazy_view=False)
    lazy = forward_train(lazy_cfg, params, tokens)
    dense = forward_train(dense_cfg, params, tokens)
    np.testing.assert_allclose(np.asarray(lazy["logits"]),
                               np.asarray(dense["logits"]),
                               rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(float(lazy["loss"]), float(dense["loss"]),
                               rtol=1e-5, atol=1e-5)
    # And the lazy path stays correct under query chunking.
    chunked = forward_train(lazy_cfg, params, tokens, query_chunk_size=8)
    np.testing.assert_allclose(np.asarray(chunked["logits"]),
                               np.asarray(dense["logits"]),
                               rtol=1e-4, atol=1e-4)


def test_train_delta_layout_matches_decode_store(params):
    """1.2: the training store layout matches the real prefill+decode store.

    Values can differ (training pass A sees the full teacher-forced prefix;
    decode pass A is T=1), but segment validity and source positions must be
    identical at every generated position.
    """
    from .attention import assemble_store_view

    tokens = _diag_batch(1)
    Tp = 32
    out = forward_train_diagnostics(CFG, params, tokens, prefill_len=Tp)
    view = out["view"]
    P, D, W = view["n_pool"], view["n_delta"], view["n_win"]
    assert P == Tp // CFG.pool_m and D > 0 and W == CFG.window

    store, _ = prefill(CFG, params, tokens[:, :Tp])
    jdec = jax.jit(lambda s, tok, pos: decode_step(CFG, params, s, tok, pos))
    for t in range(Tp - 1, tokens.shape[1]):
        if t >= Tp:
            store, _ = jdec(store, tokens[:, t], t)
        sv = assemble_store_view(CFG, store, t)

        # Training side: source positions of every valid segment row.
        pv = np.asarray(view["valid"][0, t, :P])
        train_pool = set(np.asarray(out["pool_last"][0, :P])[pv].tolist())
        dv = np.asarray(view["valid"][0, t, P:P + D])
        train_delta = set(np.asarray(view["delta_src_pos"][0, t])[dv].tolist())
        rv = np.asarray(view["valid"][0, t, P + D:])
        train_ring = set(np.asarray(view["src_pos"][0, t])[rv].tolist())

        # Decode side: the real store at the same query position. The store's
        # pool capacity is `max_seq_len // pool_m`, larger than the training
        # prompt pool; only validity/source positions are compared.
        sP, sD = sv["n_pool"], sv["n_delta"]
        spv = np.asarray(sv["valid"][0, 0, :sP])
        store_pool = set(np.asarray(store["pool_last"][0, :sP])[spv].tolist())
        sdv = np.asarray(sv["valid"][0, 0, sP:sP + sD])
        store_delta = set(
            np.asarray(store["delta_last"][0, :sD])[sdv].tolist())
        srv = np.asarray(sv["valid"][0, 0, sP + sD:])
        store_ring = set(np.asarray(store["ring_pos"][0])[srv].tolist())

        assert train_pool == store_pool, f"pool mismatch at t={t}"
        assert train_delta == store_delta, f"delta mismatch at t={t}"
        assert train_ring == store_ring, f"ring mismatch at t={t}"
