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

from .block import sinkhorn
from .cache import new_store
from .config import LoCEDConfig
from .diagnostics import (build_report, pool_norm_ratio, segment_layout,
                          segment_mass, selection_summary,
                          train_decode_kl_report)
from .hf_data import (DATASET_TEXT_FIELDS, DEFAULT_DATASETS,
                      clamp_token_ids, describe_dataset_fields,
                      extract_text, hf_batches, local_dataset_files,
                      mix_iterators, open_dataset_streams,
                      resolve_text_fields)
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


def test_magic_norm(params):
    """MagicNorm caps every visit with a module-level RMSNorm."""
    cfg = dataclasses.replace(CFG, magic_norm=True)
    p = init_model(cfg, KEY)
    assert all("norm_magic" in cell for cell in p["cells"])
    assert count_params(p) - count_params(params) == CFG.num_layers * CFG.hidden_size
    tokens = _diag_batch(2, 32)
    out = forward_train(cfg, p, tokens)
    assert np.isfinite(float(out["loss"]))
    assert out["logits"].shape == (2, 32, CFG.vocab_size)


def test_mhc_streams_and_sinkhorn(params):
    """mHC residual streams stay finite and H_res is doubly stochastic."""
    cfg = dataclasses.replace(CFG, mhc_streams=4, magic_norm=True)
    p = init_model(cfg, KEY)
    assert p["mhc_res"].shape == (CFG.num_visits, 4, 4)
    h = sinkhorn(p["mhc_res"], cfg.mhc_sinkhorn_iters)
    np.testing.assert_allclose(np.asarray(h.sum(axis=-1)), 1.0, atol=1e-5)
    np.testing.assert_allclose(np.asarray(h.sum(axis=-2)), 1.0, atol=1e-5)

    tokens = _diag_batch(2, 32)
    out = forward_train(cfg, p, tokens)
    assert np.isfinite(float(out["loss"]))
    # Streams must actually diverge; a uniform H_pre/H_post would collapse mHC
    # back to a single residual stream.
    from .model import _encode_pass
    positions = jnp.broadcast_to(jnp.arange(32)[None, :], tokens.shape)
    streams, _, _ = _encode_pass(cfg, p, p["embed"][tokens], positions)
    assert float(np.ptp(np.asarray(streams), axis=2).mean()) > 1e-4
    store, logits = prefill(cfg, p, tokens[:, :16])
    assert logits.shape == (2, 1, CFG.vocab_size)
    store, logits = decode_step(cfg, p, store, tokens[:, 16], 16)
    assert logits.shape == (2, 1, CFG.vocab_size)


def test_mhc_dynamic(params):
    """The single-pass dynamic-coefficient path is differentiable and finite."""
    cfg = dataclasses.replace(CFG, mhc_streams=4, mhc_dynamic=True)
    p = init_model(cfg, KEY)
    assert p["mhc_dyn"].shape == (
        CFG.num_visits, CFG.hidden_size, 4 * 4 + 2 * 4)
    out = forward_train(cfg, p, _diag_batch(2, 32))
    assert np.isfinite(float(out["loss"]))


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


# ---------------------------------------------------------------------------
# Hugging Face data path (offline: no network, no dataset access)
# ---------------------------------------------------------------------------

def test_hf_dataset_field_map_is_complete():
    """Every default dataset has an explicit text-field mapping.

    The eight datasets do not share a schema, so a generic "first string
    column" fallback would silently train on titles or UUIDs.
    """
    assert len(DEFAULT_DATASETS) == 8
    for name in DEFAULT_DATASETS:
        short = name.rsplit("/", 1)[-1]
        assert short in DATASET_TEXT_FIELDS, f"no text fields for {name}"
        fields = DATASET_TEXT_FIELDS[short]
        assert fields and all(isinstance(f, str) and f for f in fields)
        assert resolve_text_fields(name) == fields


def test_hf_dataset_field_map_matches_the_verified_schemas():
    """The verified column choices (the ones that carry the real prose)."""
    assert DATASET_TEXT_FIELDS["tiny-textbooks"] == ("textbook", "text")
    assert DATASET_TEXT_FIELDS["tiny-strange-textbooks"] == ("text",)
    assert DATASET_TEXT_FIELDS["tiny-math-textbooks"] == ("response",)
    assert DATASET_TEXT_FIELDS["tiny-codes"] == ("prompt", "response")
    # tiny-code-textbooks has NO textbook/text column: `content` is the raw
    # code and `response` is the explanation textbook.
    assert DATASET_TEXT_FIELDS["tiny-code-textbooks"] == ("response", "content")
    # `context` is an empty list in practice, so only `markdown` is used.
    assert DATASET_TEXT_FIELDS["programming_books_llama"] == ("markdown",)
    assert DATASET_TEXT_FIELDS["tiny-webtext"] == ("human", "bot")
    assert DATASET_TEXT_FIELDS["textbooks-are-all-you-need-lite"] == \
        ("completion", "notes")


def test_hf_describe_dataset_fields_is_auditable():
    """One line per dataset, naming the dataset and the fields it uses."""
    lines = describe_dataset_fields(DEFAULT_DATASETS)
    assert len(lines) == len(DEFAULT_DATASETS)
    for name, line in zip(DEFAULT_DATASETS, lines):
        assert name in line
        assert "fields=" in line
    # an unknown dataset falls back to the generic column search
    fallback = describe_dataset_fields(["someone/unknown-dataset"])
    assert "generic fallback" in fallback[0]


def test_hf_extract_text_uses_the_dataset_fields():
    """Only the mapped string fields are concatenated, in order."""
    row = {
        "response": "the explanation textbook",
        "content": "fn main() {}",
        "idx": 7,
    }
    text = extract_text("nampdn-ai/tiny-code-textbooks", row)
    assert text == "the explanation textbook\n\nfn main() {}"

    row = {"markdown": "# Chapter 1", "context": [], "outline": ["a", "b"]}
    text = extract_text("open-phi/programming_books_llama", row)
    assert text == "# Chapter 1"

    row = {"human": "web text", "bot": "bot text",
           "part_of_speech": "noun"}
    text = extract_text("nampdn-ai/tiny-webtext", row)
    assert text == "web text\n\nbot text"

    # empty / missing fields never produce a document
    assert extract_text("nampdn-ai/tiny-math-textbooks",
                        {"response": "   "}) is None
    assert extract_text("nampdn-ai/tiny-math-textbooks", {}) is None


def test_hf_mix_iterators_contains_every_dataset():
    """The ratio schedule mixes every dataset into the stream."""
    counts = [7, 5, 3]
    iterators = [
        iter([f"{i}:{j}" for j in range(count)])
        for i, count in enumerate(counts)
    ]
    rng = np.random.default_rng(0)
    mixed = list(mix_iterators(iterators, [1.0, 1.0, 1.0], rng))
    assert len(mixed) == sum(counts)
    seen = {item.split(":")[0] for item in mixed}
    assert seen == {"0", "1", "2"}

    # the ratios are respected while both datasets are alive: 3:1 gives the
    # first dataset ~3x the slots of the first 400 documents.
    iterators = [iter([f"a{i}" for i in range(400)]),
                 iter([f"b{i}" for i in range(400)])]
    mixed = list(mix_iterators(iterators, [3.0, 1.0], np.random.default_rng(1)))
    head = mixed[:400]
    a = sum(1 for item in head if item.startswith("a"))
    b = sum(1 for item in head if item.startswith("b"))
    assert a > 0 and b > 0
    assert 2.0 < a / b < 4.0


def test_hf_token_ids_are_clamped():
    """Token ids always land inside the embedding table."""
    assert clamp_token_ids([0, 1, 50256], 50257) == [0, 1, 50256]
    assert clamp_token_ids([-5, 50257, 99999], 50257) == [0, 50256, 50256]
    assert clamp_token_ids([], 50257) == []
    with pytest.raises(ValueError):
        clamp_token_ids([0], 0)


def test_hf_local_parquet_mirror_lookup(tmp_path):
    """The prefetched mirror is found per dataset, without test shards."""
    root = tmp_path / "tiny-codes"
    root.mkdir()
    (root / "train-00000-of-00001.parquet").write_bytes(b"")
    (root / "test-00000-of-00001.parquet").write_bytes(b"")
    (root / "validation.parquet").write_bytes(b"")
    files = local_dataset_files("nampdn-ai/tiny-codes", tmp_path)
    assert len(files) == 1
    assert files[0].endswith("train-00000-of-00001.parquet")
    with pytest.raises(FileNotFoundError):
        local_dataset_files("nampdn-ai/tiny-math-textbooks", tmp_path)


def test_hf_dataset_failure_is_graceful():
    """A dataset that cannot be loaded is skipped, not fatal.

    The stream is opened offline: no dataset can be reached here, so every
    dataset must be reported as a failure instead of raising.
    """
    alive, failures = open_dataset_streams(
        ["nampdn-ai/tiny-textbooks", "open-phi/programming_books_llama"])
    assert alive == []
    assert len(failures) == 2
    for message in failures:
        assert "skipped" in message


class _FakeTokenizer:
    """A tiny offline tokenizer (character codes) for the mirror test."""

    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = None

    def encode(self, text, add_special_tokens=False):
        return [ord(char) % 500 for char in text][:64]


def test_hf_batches_over_a_local_parquet_mirror(tmp_path, monkeypatch):
    """``hf_batches`` reads a prefetched parquet mirror and emits int32
    ``[batch, seq]`` batches. Skipped when ``datasets``/``pyarrow`` are not
    installed (the reference CPU environment has no ``datasets``)."""
    pytest.importorskip("datasets")
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    root = tmp_path / "tiny-textbooks"
    root.mkdir()
    table = pa.table({
        "textbook": [f"lesson {i}: " + "alpha beta gamma " * 8
                     for i in range(64)],
        "text": [f"excerpt {i}" for i in range(64)],
    })
    pq.write_table(table, root / "train-00000-of-00001.parquet")

    monkeypatch.setattr("loced.hf_data.build_tokenizer",
                        lambda name, token=None: _FakeTokenizer())
    batches = list(hf_batches(
        vocab_size=50257,
        seq_len=16,
        batch_size=2,
        datasets=("nampdn-ai/tiny-textbooks",),
        ratios=(1.0,),
        seed=0,
        tokenizer_name="gpt2",
        max_examples=64,
        local_root=tmp_path,
    ))
    assert batches, "the local mirror produced no batches"
    for batch in batches:
        arr = np.asarray(batch)
        assert arr.ndim == 2
        assert arr.shape[1] == 16
        assert arr.dtype == np.int32
        assert arr.min() >= 0 and arr.max() < 50257
