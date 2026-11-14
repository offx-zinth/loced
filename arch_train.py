"""LoCED 12-layer reference training entry point (synthetic or real HF data).

This is the validated 12-layer LoCED script, promoted from
``LoCED_100M_12L_Dynamic_mHC_GPT2_Synth_Kaggle_TPU.ipynb`` into an importable
module with a ``main()`` and an argparse CLI.

Stages
  1. build the 12-layer config and audit the parameter budget
  2. install a sparse top-1 MoE dispatch (ragged_dot) with a dense fallback
  3. verify the data-parallel gradient reduction on the real accelerator
  4. train on synthetic GPT-2-token data OR on real Hugging Face text
  5. evaluate, generate, and write a machine-readable report

The data source is selected with ``--data {synthetic,hf}``.  In ``hf`` mode the
batches come from :func:`loced.hf_data.hf_batches`, which streams the eight
real datasets (mixed with a deterministic ratio schedule) and tokenises them
with the GPT-2 tokenizer.

The training path itself is unchanged: a 1-D ``data`` mesh, replicated
parameters, a batch sharded over the data axis, Muon via
:func:`loced.train.make_optimizer`, and the ``reuse_p`` ramp (0 for the first
20% of the steps, ramped to 1 over the next 30%, then 1).

Usage (Kaggle TPU v5e-8)::

    python -m loced.arch_train --data hf --steps 2000 --batch-size 8 \
        --out /kaggle/working/loced_report.json
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

import jax
import jax.numpy as jnp
import optax

from . import block as loced_block
from .block import sinkhorn
from .config import LoCEDConfig
from .fnn import _swiglu, init_moe
from .model import count_params, forward_train, generate, init_model
from .train import make_optimizer, tiny_config

# ---------------------------------------------------------------------------
# the validated 12-layer budget (do not change these numbers)
# ---------------------------------------------------------------------------

N_LAYERS = 12
D_MODEL = 256
HEADS = 8
HEAD_DIM = 32
KV_DIM = 32
WINDOW = 32
TOP_K = 32
INDEXER_HEADS = 4
INDEXER_DIM = 64
INDEXER_BLOCK = 4
HIER_BLOCKS = 4
POOL_M = 2
MAGIC_NORM = True
MHC_STREAMS = 4
MHC_DYNAMIC = True
MHC_SINKHORN_ITERS = 10
NUM_EXPERTS = 62
TOP_K_EXPERTS = 1
NUM_SHARED_EXPERTS = 1
INTERMEDIATE_SIZE = 168
VOCAB_SIZE = 50257
SEQ_LEN = 128
DECODE_LEN = 32

#: Number of devices in the data-parallel mesh (set at runtime).
_N_DEVICES = 1


# ---------------------------------------------------------------------------
# 1. config + parameter budget
# ---------------------------------------------------------------------------

def build_config(vocab_size: int = VOCAB_SIZE, seq_len: int = SEQ_LEN,
                 decode_len: int = DECODE_LEN,
                 num_experts: int = NUM_EXPERTS) -> LoCEDConfig:
    """The validated 12-layer config: 12 physical layers, 24 visits."""
    encode = tuple("full" for _ in range(N_LAYERS))
    decode = (encode + tuple("reuse" for _ in range(6))
              + tuple("reindex" for _ in range(6)))
    return LoCEDConfig(
        vocab_size=vocab_size,
        hidden_size=D_MODEL,
        max_seq_len=seq_len,
        max_decode_len=decode_len,
        magic_norm=MAGIC_NORM,
        mhc_streams=MHC_STREAMS,
        mhc_sinkhorn_iters=MHC_SINKHORN_ITERS,
        mhc_dynamic=MHC_DYNAMIC,
        num_attention_heads=HEADS,
        head_dim=HEAD_DIM,
        kv_dim=KV_DIM,
        pool_m=POOL_M,
        window=min(WINDOW, seq_len // 2),
        top_k=min(TOP_K, seq_len // 2),
        indexer_heads=INDEXER_HEADS,
        indexer_dim=INDEXER_DIM,
        indexer_block=INDEXER_BLOCK,
        hier_blocks=HIER_BLOCKS,
        encode_layers=encode,
        decode_layers=decode,
        num_experts=num_experts,
        num_shared_experts=NUM_SHARED_EXPERTS,
        top_k_experts=TOP_K_EXPERTS,
        intermediate_size=INTERMEDIATE_SIZE,
    )


def count_active_body(cfg: LoCEDConfig, params: dict) -> int:
    """Body weights one token touches: attention, norms, mHC, router, the
    shared expert, and exactly top_k routed experts per physical layer."""
    n = 0
    for name in ("W_bk", "W_bv", "W_ik", "W_iq"):
        n += int(params[name].size)
    for name in ("cond", "gate_a", "gate_m", "norm_f",
                 "mhc_res", "mhc_pre", "mhc_post", "mhc_dyn"):
        n += int(params[name].size)
    for cell in params["cells"]:
        for name in ("norm_a", "norm_m", "norm_magic", "W_q", "W_fx", "W_fv",
                     "W_o", "W_gate", "Ws1", "Ws3", "Ws2"):
            n += int(cell[name].size)
        per_expert = int(cell["We1"][0].size + cell["We3"][0].size
                         + cell["We2"][0].size)
        n += cfg.top_k_experts * per_expert
    return int(n)


def selection_budget(cfg: LoCEDConfig, seq_len: int) -> dict:
    """Candidates available to a pass-B visit vs the rows it actually selects.

    L = min(n_candidates, window + top_k). When L == n_candidates the
    selection is a no-op (every candidate is picked), so the indexer cannot
    influence the REUSE layers and reuse_p has no effect. This must be
    checked, otherwise the sparse path is silently untested.
    """
    m = cfg.pool_m
    tp = (int(round(seq_len * cfg.train_prefill_frac)) // m) * m
    pool = tp // m
    delta_start = max(0, tp - cfg.window)
    delta = min(cfg.max_decode_len, seq_len - delta_start) if tp < seq_len else 0
    n_cand = pool + delta + cfg.window
    limit = min(n_cand, cfg.window + cfg.top_k)
    return {"n_candidates": int(n_cand), "selection_budget": int(limit),
            "is_sparse": bool(limit < n_cand),
            "selection_ratio": float(limit / n_cand)}


# ---------------------------------------------------------------------------
# 2. sparse top-1 MoE dispatch
# ---------------------------------------------------------------------------

def _ragged_dot_supported() -> bool:
    try:
        lhs = jnp.zeros((4, 8), jnp.float32)
        rhs = jnp.zeros((2, 8, 4), jnp.float32)
        sizes = jnp.array([2, 2], jnp.int32)
        out = jax.jit(lambda a, b, s: jax.lax.ragged_dot(a, b, s))(
            lhs, rhs, sizes)
        out.block_until_ready()
        return True
    except Exception as exc:  # pragma: no cover - backend dependent
        print(f"ragged_dot unavailable on this backend ({exc}); "
              f"falling back to the dense reference MoE")
        return False


def _psum_div(x, axis: str = "data"):
    """Average a batch statistic over the data-parallel axis.

    Inside shard_map the MoE load-balance auxiliary loss must use GLOBAL
    batch statistics: it is a product of two batch-level means, so a per-shard
    value cannot be recombined into the global one. Outside shard_map the
    axis is unbound and this is the identity.
    """
    try:
        return jax.lax.psum(x, axis) / _N_DEVICES
    except Exception as exc:  # unbound 'data' axis outside shard_map
        if "axis" not in str(exc).lower():
            raise
        return x


def _global_aux_moe(cfg: LoCEDConfig, p: dict, x):
    """Dense reference MoE whose load-balance auxiliary uses global stats."""
    shared = _swiglu(x @ p["Ws1"], x @ p["Ws3"]) @ p["Ws2"]
    a = jnp.einsum("btd,edi->btei", x, p["We1"])
    g = jnp.einsum("btd,edi->btei", x, p["We3"])
    h = _swiglu(a, g)
    y = jnp.einsum("btei,eid->bted", h, p["We2"])
    probs = jax.nn.softmax(x @ p["W_gate"], axis=-1)
    k = cfg.top_k_experts
    top_idx = jax.lax.top_k(probs, k)[1]
    w = jnp.take_along_axis(probs, top_idx, axis=-1)
    w = w / jnp.maximum(w.sum(-1, keepdims=True), 1e-9)
    mask = jax.nn.one_hot(top_idx, cfg.num_experts, dtype=jnp.float32)
    mask = (mask * w[..., None]).sum(axis=2)
    routed = jnp.einsum("bte,bteh->bth", mask, y)
    out = shared + routed
    frac = jnp.mean((mask > 0).astype(jnp.float32), axis=(0, 1))
    mean_p = jnp.mean(probs, axis=(0, 1))
    aux = cfg.num_experts * jnp.sum(_psum_div(frac) * _psum_div(mean_p))
    return out, aux


def sparse_top1_moe(cfg: LoCEDConfig, p: dict, x):
    """Top-1 MoE: a token reads only its selected expert's three SwiGLU
    matrices. Mathematically identical to the dense kernel for top-1."""
    b, t, d = x.shape
    shared = _swiglu(x @ p["Ws1"], x @ p["Ws3"]) @ p["Ws2"]

    xf = x.reshape(b * t, d)
    probs = jax.nn.softmax(xf @ p["W_gate"], axis=-1)
    top_w, top_idx = jax.lax.top_k(probs, 1)
    top_w = top_w / jnp.maximum(top_w.sum(-1, keepdims=True), 1e-9)
    flat_idx = top_idx[:, 0]

    perm = jnp.argsort(flat_idx)
    inv = jnp.argsort(perm)
    xs = xf[perm]
    sizes = jnp.bincount(flat_idx, length=cfg.num_experts)

    a = jax.lax.ragged_dot(xs, p["We1"], sizes)
    g = jax.lax.ragged_dot(xs, p["We3"], sizes)
    h = _swiglu(a, g)
    y = jax.lax.ragged_dot(h, p["We2"], sizes)
    routed = (y * top_w[:, 0][:, None])[inv].reshape(b, t, d)

    frac = jnp.mean(
        jax.nn.one_hot(flat_idx, cfg.num_experts, dtype=jnp.float32), axis=0)
    mean_p = jnp.mean(probs, axis=0)
    aux = cfg.num_experts * jnp.sum(_psum_div(frac) * _psum_div(mean_p))
    return shared + routed, aux


@contextlib.contextmanager
def moe_dispatch(cfg: LoCEDConfig) -> Iterator[dict]:
    """Install the sparse top-1 dispatch for the duration of the block.

    The dense fallback keeps the SAME global auxiliary-loss semantics, so the
    two paths are numerically equivalent.  The original kernel is restored on
    exit, so importing this module never changes the repository behaviour.
    """
    original = loced_block.moe_forward
    info: dict = {"dispatch": "dense_reference", "parity": None}
    try:
        if cfg.top_k_experts == 1 and _ragged_dot_supported():
            tiny = dataclasses.replace(tiny_config(), top_k_experts=1)
            tiny_params = init_moe(tiny, jax.random.PRNGKey(1))
            tiny_x = jax.random.normal(jax.random.PRNGKey(2),
                                       (2, 8, tiny.hidden_size))
            dense_out, dense_aux = _global_aux_moe(tiny, tiny_params, tiny_x)
            sparse_out, sparse_aux = sparse_top1_moe(tiny, tiny_params, tiny_x)
            value_err = float(jnp.max(jnp.abs(dense_out - sparse_out)))
            aux_err = float(jnp.abs(dense_aux - sparse_aux))
            dense_grad = jax.grad(
                lambda q: _global_aux_moe(tiny, q, tiny_x)[0].sum())(tiny_params)
            sparse_grad = jax.grad(
                lambda q: sparse_top1_moe(tiny, q, tiny_x)[0].sum())(tiny_params)
            grad_err = max(
                float(jnp.max(jnp.abs(a - b)))
                for a, b in zip(jax.tree.leaves(dense_grad),
                                jax.tree.leaves(sparse_grad)))
            info["parity"] = {"value": value_err, "aux": aux_err,
                              "grad": grad_err}
            print(f"top-1 dense-vs-sparse parity: value={value_err:.3e} "
                  f"aux={aux_err:.3e} grad={grad_err:.3e}", flush=True)
            if value_err < 1e-6 and aux_err < 1e-6 and grad_err < 1e-5:
                loced_block.moe_forward = sparse_top1_moe
                info["dispatch"] = "sparse_top1_ragged_dot"
                print("installed sparse top-1 MoE dispatch (ragged_dot)")
            else:
                # Fall back to the dense reference kernel, but keep the SAME
                # global auxiliary-loss semantics so the training path is
                # identical either way.
                loced_block.moe_forward = _global_aux_moe
                print("sparse dispatch failed parity; using the dense MoE with "
                      "global aux statistics")
        else:
            loced_block.moe_forward = _global_aux_moe
            print("using the dense reference MoE (top_k_experts != 1 or no "
                  "ragged_dot)")
        yield info
    finally:
        loced_block.moe_forward = original


# ---------------------------------------------------------------------------
# 3. data-parallel gradient reduction on the real accelerator
# ---------------------------------------------------------------------------

def _shard_map(fn, mesh, in_specs, out_specs):
    """Version-tolerant shard_map (jax.shard_map vs the experimental one)."""
    try:
        return jax.shard_map(fn, mesh=mesh, in_specs=in_specs,
                             out_specs=out_specs)
    except AttributeError:  # pragma: no cover - older jax
        from jax.experimental.shard_map import shard_map
        return shard_map(fn, mesh, in_specs, out_specs)


def mesh_grad_check(n_devices: int) -> dict:
    """Compare the sharded gradient against the single-device gradient.

    Inside shard_map the gradient of a REPLICATED parameter is already
    all-reduced (summed) across the data axis, so the local loss must be
    scaled by 1/N_DEVICES for the result to match the single-device gradient
    of the global mean. This check validates that on the actual accelerator.
    """
    if n_devices == 1:
        return {"checked": False, "reason": "single device"}

    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    tiny = dataclasses.replace(tiny_config(), top_k_experts=1)
    tokens = np.zeros((n_devices, tiny.max_seq_len), dtype=np.int32)
    rng = np.random.default_rng(0)
    for i in range(n_devices):
        row = rng.integers(0, tiny.vocab_size, size=tiny.max_seq_len)
        row[0] = i % tiny.vocab_size
        tokens[i] = row
    tokens = jnp.asarray(tokens)

    small_params = init_model(tiny, jax.random.PRNGKey(3))

    def loss_of(p, toks):
        return forward_train(tiny, p, toks, reuse_p=0.0)["loss"]

    g_single = jax.grad(lambda p: loss_of(p, tokens))(small_params)
    mesh = Mesh(np.array(jax.devices()), axis_names=("data",))
    replicated = NamedSharding(mesh, P())
    batch_sharded = NamedSharding(mesh, P("data"))

    def sharded_loss(p, toks):
        # toks is the LOCAL shard of the batch; the loss is a mean over the
        # local rows, so dividing by N_DEVICES turns the cross-shard sum into
        # the global mean.
        return loss_of(p, toks) / n_devices

    sharded_grad_fn = _shard_map(
        lambda p, toks: jax.grad(sharded_loss)(p, toks),
        mesh=mesh,
        in_specs=(P(), P("data")),
        out_specs=P(),
    )
    sharded_grad_fn = jax.jit(sharded_grad_fn)

    p_dev = jax.device_put(small_params, replicated)
    t_dev = jax.device_put(tokens, batch_sharded)
    g_sharded = sharded_grad_fn(p_dev, t_dev)

    diffs = [
        float(jnp.max(jnp.abs(a - b)))
        for a, b in zip(jax.tree.leaves(g_single), jax.tree.leaves(g_sharded))
    ]
    max_diff = max(diffs)
    scale = max(1e-6, max(float(jnp.max(jnp.abs(a)))
                          for a in jax.tree.leaves(g_single)))
    rel = max_diff / scale
    print(f"mesh gradient check: max abs diff {max_diff:.3e}  "
          f"relative {rel:.3e}  ({n_devices} devices)", flush=True)
    if rel > 1e-2:
        raise RuntimeError(
            f"data-parallel gradient mismatch (relative {rel:.3e}); the "
            f"sharded gradient does not match the single-device gradient")
    if rel > 1e-4:
        print("WARNING: mesh gradient relative difference above 1e-4 "
              "(float32 accumulation noise); continuing", flush=True)
    return {"checked": True, "max_abs_diff": max_diff, "relative_diff": rel,
            "within_1e-4": bool(rel <= 1e-4), "devices": n_devices}


# ---------------------------------------------------------------------------
# 4. data
# ---------------------------------------------------------------------------

def synthetic_batch_gpt2(rng: np.random.Generator, batch: int, seq: int,
                          rule_p: float = 0.9,
                          active_vocab: int = 512) -> jnp.ndarray:
    """Causal local rule: next token is (prev + 1) % ACTIVE_VOCAB with
    probability rule_p, else uniform noise over the same subset of GPT-2
    token IDs. The model keeps the full 50,257-way GPT-2 head."""
    tok = np.zeros((batch, seq), dtype=np.int32)
    tok[:, 0] = rng.integers(0, active_vocab, size=batch)
    for t in range(1, seq):
        step = (tok[:, t - 1] + 1) % active_vocab
        noise = rng.integers(0, active_vocab, size=batch)
        take_rule = rng.random(batch) < rule_p
        tok[:, t] = np.where(take_rule, step, noise)
    return jnp.asarray(tok)


def _parse_list(value: str | None, name: str) -> list[str]:
    if not value:
        return []
    items = [part.strip() for part in value.split(",")]
    items = [part for part in items if part]
    if not items:
        raise ValueError(f"--{name} is empty")
    return items


def make_batches(args: argparse.Namespace, cfg: LoCEDConfig,
                 global_batch: int) -> tuple[Iterator[jnp.ndarray], jnp.ndarray]:
    """The batch iterator and the evaluation batch.

    In ``synthetic`` mode the iterator is the validated local-rule generator.
    In ``hf`` mode it streams the real datasets; when the stream is exhausted
    it is restarted (a fresh seed) so a long run is never starved.
    """
    if args.data == "synthetic":
        rng = np.random.default_rng(args.seed + 1)

        def synthetic() -> Iterator[jnp.ndarray]:
            while True:
                yield synthetic_batch_gpt2(rng, global_batch, args.seq_len,
                                           rule_p=args.rule_p,
                                           active_vocab=args.active_vocab)

        eval_rng = np.random.default_rng(123)
        eval_tokens = synthetic_batch_gpt2(eval_rng, 4, args.seq_len,
                                           rule_p=args.rule_p,
                                           active_vocab=args.active_vocab)
        return synthetic(), eval_tokens

    from .hf_data import DEFAULT_DATASETS, hf_batches

    datasets = _parse_list(args.datasets, "datasets") or list(DEFAULT_DATASETS)
    ratios = _parse_list(args.ratios, "ratios")
    if ratios:
        ratios = [float(r) for r in ratios]
        if len(ratios) != len(datasets):
            raise ValueError(
                f"--ratios has {len(ratios)} values for {len(datasets)} datasets")
    else:
        ratios = [1.0] * len(datasets)

    state = {"iteration": 0}

    def hf() -> Iterator[jnp.ndarray]:
        while True:
            state["iteration"] += 1
            batches = hf_batches(
                vocab_size=cfg.vocab_size,
                seq_len=args.seq_len,
                batch_size=global_batch,
                datasets=datasets,
                ratios=ratios,
                seed=args.seed + state["iteration"],
                split=args.split,
                tokenizer_name=args.tokenizer,
                max_examples=args.max_examples,
                local_root=args.local_data,
            )
            produced = 0
            for batch in batches:
                produced += 1
                yield batch
            print(f"[arch_train] the HF stream was exhausted after "
                  f"{produced} batches; restarting it", flush=True)

    iterator = hf()
    eval_batch = next(iterator)
    eval_tokens = eval_batch[:4]
    return iterator, eval_tokens


# ---------------------------------------------------------------------------
# 5. training + evaluation
# ---------------------------------------------------------------------------

def run_training(cfg: LoCEDConfig, params: dict, args: argparse.Namespace,
                  global_batch: int, per_device_batch: int,
                  batches: Iterator[jnp.ndarray]):
    """The validated data-parallel training loop (1-D ``data`` mesh)."""
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    n_devices = len(jax.devices())
    optimizer = make_optimizer(cfg, params, optim="muon", lr=args.lr)
    opt_state = optimizer.init(params)

    mesh = Mesh(np.array(jax.devices()), axis_names=("data",))
    replicated = NamedSharding(mesh, P())
    batch_sharded = NamedSharding(mesh, P("data"))

    params = jax.device_put(params, replicated)
    opt_state = jax.device_put(opt_state, replicated)

    def _train_body(params, state, tokens, reuse_p):
        def loss_fn(p):
            out = forward_train(cfg, p, tokens, reuse_p=reuse_p)
            metrics = {k: out[k] for k in ("loss", "ce", "ce_aux", "kl",
                                           "moe_aux")}
            metrics["reuse_p"] = jnp.asarray(reuse_p, jnp.float32)
            # Scale by 1/N_DEVICES: shard_map already all-reduces the gradient
            # of a replicated parameter across the data axis, so the local
            # mean loss must be divided by the device count to reproduce the
            # global mean.
            return out["loss"] / n_devices, metrics

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        metrics = jax.lax.pmean(metrics, "data")
        updates, state = optimizer.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        return params, state, metrics

    train_step = _shard_map(
        _train_body,
        mesh=mesh,
        in_specs=(P(), P(), P("data"), P()),
        out_specs=(P(), P(), P()),
    )
    train_step = jax.jit(train_step)

    warmup = int(args.steps * cfg.reuse_warmup_frac) if cfg.reuse_onpolicy \
        else args.steps
    ramp = max(1, int(args.steps * cfg.reuse_ramp_frac))
    history: list[dict] = []
    print(f"\ntraining: {args.steps} steps, batch {global_batch} x "
          f"{args.seq_len} tokens, reuse ramp from step {warmup} over "
          f"{ramp} steps\n", flush=True)

    for i in range(args.steps):
        reuse_p = (0.0 if not cfg.reuse_onpolicy
                   else min(max((i - warmup) / ramp, 0.0), 1.0))
        tokens = next(batches)
        tokens = jax.device_put(tokens, batch_sharded)
        reuse = jax.device_put(jnp.asarray(reuse_p, jnp.float32), replicated)

        t0 = time.perf_counter()
        params, opt_state, metrics = train_step(params, opt_state, tokens,
                                                reuse)
        row = {k: float(v) for k, v in metrics.items()}
        row["ms_step"] = (time.perf_counter() - t0) * 1000.0
        row["step"] = i
        history.append(row)

        if i == 0 or (i + 1) % max(1, args.steps // 10) == 0 \
                or i == args.steps - 1:
            print(f"step {i:3d}  loss {row['loss']:.4f}  ce {row['ce']:.4f}  "
                  f"kl {row['kl']:.4f}  moe_aux {row['moe_aux']:.4f}  "
                  f"reuse_p {row['reuse_p']:.2f}  {row['ms_step']:.1f} ms",
                  flush=True)
    return params, history, warmup, ramp


def evaluate(cfg: LoCEDConfig, params: dict, eval_tokens: jnp.ndarray,
             run_generation: bool, max_new: int, active_vocab: int,
             synthetic: bool) -> tuple[dict, float, dict | None]:
    """Evaluate at reuse_p 0 / 0.5 / 1.0 and optionally generate."""
    params_trained = jax.tree.map(
        lambda x: jax.device_put(x, jax.devices()[0]), params)

    @jax.jit
    def eval_forward(p, tokens, reuse_p):
        out = forward_train(cfg, p, tokens, reuse_p=reuse_p)
        pred = jnp.argmax(out["logits"][:, :-1], axis=-1)
        acc = jnp.mean((pred == tokens[:, 1:]).astype(jnp.float32))
        return out["loss"], out["ce"], out["kl"], acc

    eval_results: dict = {}
    print()
    for reuse_p in (0.0, 0.5, 1.0):
        loss, ce, kl, acc = eval_forward(params_trained, eval_tokens, reuse_p)
        eval_results[str(reuse_p)] = {
            "loss": float(loss), "ce": float(ce), "kl": float(kl),
            "next_token_acc": float(acc),
        }
        print(f"eval reuse_p={reuse_p:.1f}  loss={float(loss):.4f}  "
              f"ce={float(ce):.4f}  kl={float(kl):.4f}  "
              f"next-token acc={float(acc):.3f}", flush=True)

    # The oracle and the indexer must NOT agree exactly, otherwise the sparse
    # selection is not being exercised and the reuse_p ramp trains nothing.
    oracle_loss = eval_results["0.0"]["loss"]
    indexer_loss = eval_results["1.0"]["loss"]
    eval_spread = abs(oracle_loss - indexer_loss)
    print(f"eval oracle-vs-indexer loss spread: {eval_spread:.6f}", flush=True)
    if eval_spread <= 0.0:
        print("WARNING: oracle and indexer give identical eval loss; the "
              "sparse selection may not be affecting the REUSE layers",
              flush=True)

    generation = None
    if run_generation:
        prompt = eval_tokens[:1, :max(2, min(16, cfg.max_seq_len // 2))]
        t0 = time.perf_counter()
        gen = generate(cfg, params_trained, prompt, max_new)
        gen_ids = np.asarray(gen)[0]
        rule_acc = (float(np.mean(gen_ids[1:] == (gen_ids[:-1] + 1) % active_vocab))
                    if synthetic and len(gen_ids) > 1 else None)
        generation = {
            "prompt_ids": np.asarray(prompt)[0].tolist(),
            "generated_ids": gen_ids.tolist(),
            "greedy_rule_accuracy": rule_acc,
            "seconds": time.perf_counter() - t0,
        }
        print(f"generation: {gen_ids.tolist()}", flush=True)
    return eval_results, eval_spread, generation


# ---------------------------------------------------------------------------
# 6. report
# ---------------------------------------------------------------------------

def _git_commit(path: str | os.PathLike = ".") -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            text=True).strip()
    except Exception:
        return None


def write_report(report: dict, out_path: Path) -> tuple[Path, list[str]]:
    """Write the JSON report and a plain-text summary next to it."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=float))

    p = report["params"]
    t = report["training"]
    history = t["history"]
    lines = [
        f"backend={report['backend']} devices={report['devices']} "
        f"jax={report['jax_version']}",
        f"data={report['data']['source']}",
        f"total_params={p['total']} body={p['body']} "
        f"active_body={p['active_body_per_token']} "
        f"({p['body_active_fraction']:.4%} of body)",
        f"moe_dispatch={report['moe_dispatch']}",
        f"mesh_grad_check={report['mesh_gradient_check']}",
        f"selection={p['selection']}",
        f"steps={t['steps']} batch={t['global_batch']} seq_len={t['seq_len']} "
        f"ms_per_step={t['mean_ms_per_step_after_first']:.1f}",
    ]
    if history:
        lines.append(f"loss_first={history[0]['loss']:.4f} "
                     f"loss_last={history[-1]['loss']:.4f}")
    for k, v in report["eval"].items():
        lines.append(
            f"eval reuse_p={k}: loss={v['loss']:.4f} ce={v['ce']:.4f} "
            f"acc={v['next_token_acc']:.4f}")
    lines.append(
        f"eval_oracle_vs_indexer_spread={report['eval_oracle_vs_indexer_spread']:.6f}")
    if report.get("generation"):
        gen = report["generation"]
        lines.append(f"generation={gen['generated_ids']}")
    out_path.with_suffix(".summary.txt").write_text("\n".join(lines) + "\n")
    return out_path, lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoCED 12-layer (100M body) training: synthetic or real HF data")
    parser.add_argument("--data", default="synthetic", choices=["synthetic", "hf"],
                        help="training data source")
    parser.add_argument("--datasets", default=None,
                        help="comma-separated Hugging Face dataset names "
                             "(hf mode)")
    parser.add_argument("--ratios", default=None,
                        help="comma-separated mixing ratios, one per dataset")
    parser.add_argument("--split", default="train", help="HF split (hf mode)")
    parser.add_argument("--tokenizer", default="gpt2",
                        help="Hugging Face tokenizer name (hf mode)")
    parser.add_argument("--local-data", default=None,
                        help="prefetched parquet mirror root (hf mode)")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="max documents per dataset (hf mode)")
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--decode-len", type=int, default=DECODE_LEN)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="GLOBAL batch (sharded over the data mesh)")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None,
                        help="path of the JSON report")
    parser.add_argument("--rule-p", type=float, default=0.9,
                        help="synthetic local-rule probability")
    parser.add_argument("--active-vocab", type=int, default=512,
                        help="synthetic token subset size")
    parser.add_argument("--max-new", type=int, default=16,
                        help="tokens generated by the greedy sample")
    parser.add_argument("--run-generation", dest="run_generation",
                        action="store_true", default=True)
    parser.add_argument("--no-run-generation", dest="run_generation",
                        action="store_false")
    parser.add_argument("--expected-backend", default=None,
                        help="fail unless JAX runs on this backend")
    parser.add_argument("--expected-devices", type=int, default=None,
                        help="fail unless this many devices are visible")
    parser.add_argument("--min-devices", type=int, default=None,
                        help="fail with fewer devices than this")
    args = parser.parse_args(argv)

    n_devices = len(jax.devices())
    if args.batch_size is None:
        args.batch_size = 8 if n_devices >= 8 else max(1, 2 * n_devices)
    if args.steps is None:
        args.steps = 20 if n_devices >= 2 else 3
    # The global batch must be a multiple of the number of devices: the batch
    # is sharded over the data axis.
    args.batch_size = max(n_devices, (args.batch_size // n_devices) * n_devices)
    for flag, env in (("expected_backend", "LOCED_EXPECTED_BACKEND"),
                      ("expected_devices", "LOCED_EXPECTED_DEVICES"),
                      ("min_devices", "LOCED_MIN_DEVICES")):
        value = getattr(args, flag)
        if value is None and os.environ.get(env):
            raw = os.environ[env]
            setattr(args, flag, int(raw) if raw.isdigit() else raw)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    global _N_DEVICES

    args = parse_args(argv)
    n_devices = len(jax.devices())
    _N_DEVICES = n_devices
    global_batch = int(args.batch_size)
    per_device_batch = global_batch // n_devices

    work = Path(os.environ.get("LOCED_WORK", "/kaggle/working"))
    out_path = Path(args.out) if args.out else \
        work / "loced_100m_body_12l_mhc_gpt2_report.json"

    print("=" * 70)
    print(f"accelerator : {os.environ.get('LOCED_ACCELERATOR', 'auto')}  "
          f"backend: {jax.default_backend()}  devices: {n_devices}")
    print(f"jax         : {jax.__version__}   optax: {optax.__version__}")
    print(f"data        : {args.data}")
    print(f"batch       : global {global_batch} / per-device {per_device_batch}")
    print(f"seq_len     : {args.seq_len}   steps: {args.steps}   "
          f"lr: {args.lr}")
    print("=" * 70, flush=True)

    if args.expected_backend and jax.default_backend() != args.expected_backend:
        raise RuntimeError(
            f"expected backend {args.expected_backend!r}, got "
            f"{jax.default_backend()!r}; pick the right accelerator and rerun")
    if args.expected_devices and n_devices != args.expected_devices:
        raise RuntimeError(
            f"expected {args.expected_devices} device(s), got {n_devices}; "
            f"pick the right accelerator and rerun")
    if args.min_devices and n_devices < args.min_devices:
        raise RuntimeError(
            f"expected at least {args.min_devices} device(s), got {n_devices}")

    # --- 1. config + parameter budget -------------------------------------
    cfg = build_config(seq_len=args.seq_len, decode_len=args.decode_len)
    print(f"layers={cfg.num_layers} visits={cfg.num_visits} "
          f"(encode {cfg.prefill_visits} + decode "
          f"{cfg.num_visits - cfg.prefill_visits})")
    print(f"sparse pass : {cfg.sparse_layers}")
    print(f"mHC         : streams={cfg.mhc_streams} "
          f"dynamic={cfg.mhc_dynamic} "
          f"sinkhorn_iters={cfg.mhc_sinkhorn_iters} "
          f"magic_norm={cfg.magic_norm}")
    print(f"MoE         : {cfg.num_experts} routed, top-{cfg.top_k_experts}, "
          f"{cfg.num_shared_experts} shared, I={cfg.intermediate_size}",
          flush=True)

    params = init_model(cfg, jax.random.PRNGKey(args.seed))
    total = count_params(params)
    embed = int(params["embed"].size)
    head = int(params["head"].size)
    body = total - embed - head
    active_body = count_active_body(cfg, params)
    active_with_head = active_body + head + cfg.hidden_size

    mhc = np.asarray(sinkhorn(params["mhc_res"], cfg.mhc_sinkhorn_iters))
    row_err = float(np.max(np.abs(mhc.sum(axis=-1) - 1.0)))
    col_err = float(np.max(np.abs(mhc.sum(axis=-2) - 1.0)))

    sel = selection_budget(cfg, args.seq_len)
    print(f"selection budget : {sel}", flush=True)
    if not sel["is_sparse"]:
        raise RuntimeError(
            f"seq_len={args.seq_len} selects all {sel['n_candidates']} "
            f"candidates (window+top_k >= candidates), so the indexer cannot "
            f"influence the REUSE layers; raise --seq-len to make the path "
            f"genuinely sparse")

    print(f"total parameters          : {total:>14,}  ({total / 1e6:8.3f} M)")
    print(f"  transformer body        : {body:>14,}  ({body / 1e6:8.3f} M)")
    print(f"  GPT-2 embedding         : {embed:>14,}  ({embed / 1e6:8.3f} M)")
    print(f"  GPT-2 output head       : {head:>14,}  ({head / 1e6:8.3f} M)")
    print(f"active body per token     : {active_body:>14,}  "
          f"({active_body / 1e6:8.3f} M)")
    print(f"active incl. head + row   : {active_with_head:>14,}  "
          f"({active_with_head / 1e6:8.3f} M)")
    print(f"body sparsity             : {active_body / body:.2%} active")
    print(f"Sinkhorn max row/col err  : {max(row_err, col_err):.3e}", flush=True)

    assert 95e6 <= body <= 105e6, f"body budget off: {body}"
    assert 5e6 <= active_body <= 6e6, f"active budget off: {active_body}"
    assert max(row_err, col_err) < 1e-4, \
        "Sinkhorn projection is not doubly stochastic"

    with moe_dispatch(cfg) as dispatch:
        # --- 3. mesh gradient check ---------------------------------------
        grad_check = mesh_grad_check(n_devices)

        # --- 4. data + training -------------------------------------------
        batches, eval_tokens = make_batches(args, cfg, global_batch)
        params, history, warmup, ramp = run_training(
            cfg, params, args, global_batch, per_device_batch, batches)

        # --- 5. evaluation -------------------------------------------------
        eval_results, eval_spread, generation = evaluate(
            cfg, params, eval_tokens, args.run_generation, args.max_new,
            args.active_vocab, synthetic=(args.data == "synthetic"))

    data_info = {"source": args.data, "seed": args.seed,
                 "tokenizer": args.tokenizer, "split": args.split}
    if args.data == "hf":
        data_info.update({
            "datasets": _parse_list(args.datasets, "datasets"),
            "ratios": ([float(r) for r in _parse_list(args.ratios, "ratios")]
                       or None),
            "max_examples": args.max_examples,
            "local_data": args.local_data,
        })

    report = {
        "kind": ("loced_100m_body_12l_dynamic_mhc_magicnorm_gpt2_synth"
                 if args.data == "synthetic" else
                 "loced_100m_body_12l_dynamic_mhc_magicnorm_gpt2_hf"),
        "accelerator": os.environ.get("LOCED_ACCELERATOR", "auto"),
        "backend": jax.default_backend(),
        "devices": n_devices,
        "jax_version": jax.__version__,
        "python_version": sys.version.split()[0],
        "vocab_size": cfg.vocab_size,
        "active_vocab": args.active_vocab,
        "repo_commit": _git_commit(os.environ.get("LOCED_ROOT", ".")),
        "data": data_info,
        "config": {
            "layers": cfg.num_layers,
            "visits": cfg.num_visits,
            "d_model": cfg.hidden_size,
            "heads": cfg.num_attention_heads,
            "head_dim": cfg.head_dim,
            "kv_dim": cfg.kv_dim,
            "seq_len": cfg.max_seq_len,
            "decode_len": cfg.max_decode_len,
            "magic_norm": cfg.magic_norm,
            "mhc_streams": cfg.mhc_streams,
            "mhc_dynamic": cfg.mhc_dynamic,
            "mhc_sinkhorn_iters": cfg.mhc_sinkhorn_iters,
            "num_experts": cfg.num_experts,
            "top_k_experts": cfg.top_k_experts,
            "shared_experts": cfg.num_shared_experts,
            "intermediate_size": cfg.intermediate_size,
            "sparse_layers": list(cfg.sparse_layers),
        },
        "params": {
            "total": int(total),
            "body": int(body),
            "embedding": int(embed),
            "head": int(head),
            "active_body_per_token": int(active_body),
            "active_with_head_and_embed_row": int(active_with_head),
            "body_active_fraction": float(active_body / body),
            "sinkhorn_max_error": float(max(row_err, col_err)),
            "selection": sel,
        },
        "moe_dispatch": dispatch["dispatch"],
        "moe_dispatch_parity": dispatch["parity"],
        "mesh_gradient_check": grad_check,
        "training": {
            "steps": args.steps,
            "global_batch": global_batch,
            "per_device_batch": per_device_batch,
            "seq_len": args.seq_len,
            "learning_rate": args.lr,
            "optimizer": "muon",
            "reuse_warmup_steps": warmup,
            "reuse_ramp_steps": ramp,
            "history": history,
            "mean_ms_per_step_after_first": (
                float(np.mean([r["ms_step"] for r in history[1:]]))
                if len(history) > 1 else
                (float(history[0]["ms_step"]) if history else 0.0)),
        },
        "eval": eval_results,
        "eval_oracle_vs_indexer_spread": float(eval_spread),
        "generation": generation,
        "notes": [
            "100M target is the transformer body; GPT-2 embedding and head are "
            "on top.",
            "active_body excludes the embedding lookup and the vocabulary head.",
            "eval reuse_p sweeps the oracle -> indexer selection mix; a "
            "non-zero spread proves the sparse path is exercised.",
            "The MoE load-balance auxiliary loss uses global (all-reduced) "
            "batch statistics in both the sparse and the dense path.",
            "No plot is written; the report is the only artifact.",
        ],
    }
    if args.data == "hf":
        report["notes"].append(
            "Real Hugging Face text (mixed datasets), tokenised with the GPT-2 "
            "tokenizer; not a quality claim.")

    out_path, summary_lines = write_report(report, out_path)
    print(f"\nwrote {out_path}", flush=True)
    print(f"wrote {out_path.with_suffix('.summary.txt')}", flush=True)
    print("\n================ SUMMARY ================", flush=True)
    for line in summary_lines:
        print("  ", line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
