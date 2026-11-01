"""Diagnostics for LoCED's fidelity and selection behaviour.

The harness is deliberately outside the jitted train step: it exposes the
anchor hidden, candidate view, oracle/indexer selections and per-visit
attention probabilities, then reduces them to JSON-friendly numbers. The
training path keeps its previous contract via `model.forward_train`.

Key definitions:
  * candidate layout is always `pool | delta | ring`;
  * segment mass is measured over the *selected* rows, with the mask broadcast
    as `[B,T,1,L]` against `[B,T,H,L]` probabilities;
  * recall@L compares the indexer's top-L set with the oracle's top-L set;
  * train/decode KL runs a teacher-forced decode through the real store and
    aligns each decode logit with the training logit at the same position.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .attention import select
from .config import LoCEDConfig

NEG = -1e4


@dataclass(frozen=True)
class SegmentLayout:
    """Canonical candidate-segment sizes in store order."""

    n_pool: int
    n_delta: int
    n_win: int

    @property
    def total(self) -> int:
        return self.n_pool + self.n_delta + self.n_win

    def bounds(self) -> tuple[int, int, int]:
        return (self.n_pool, self.n_pool + self.n_delta, self.total)


def segment_layout(view: dict) -> SegmentLayout:
    """Infer the canonical layout from a train or decode candidate view.

    Train views currently expose `n_pool`/`n_win` and no delta; decode views
    expose `n_own` for the delta pool. The row axis is the source of truth if
    neither key is present.
    """
    n_pool = int(view.get("n_pool", 0))
    n_win = int(view.get("n_win", 0))
    total = (int(view["n_candidates"]) if "n_candidates" in view
             else int(view["rows_k"].shape[2]))
    if "n_delta" in view:
        n_delta = int(view["n_delta"])
    elif "n_own" in view:
        n_delta = int(view["n_own"])
    else:
        n_delta = total - n_pool - n_win
    if n_pool < 0 or n_delta < 0 or n_win < 0:
        raise ValueError(f"negative segment size in {view.keys()}")
    if n_pool + n_delta + n_win != total:
        raise ValueError(
            f"segment sizes {(n_pool, n_delta, n_win)} do not sum to "
            f"candidate rows {total}")
    return SegmentLayout(n_pool, n_delta, n_win)


def segment_ids(idx: jax.Array, layout: SegmentLayout) -> jax.Array:
    """Map selected candidate indices to `0=pool, 1=delta, 2=ring`."""
    pool_end = layout.n_pool
    delta_end = layout.n_pool + layout.n_delta
    return jnp.where(
        idx < pool_end, 0,
        jnp.where(idx < delta_end, 1, 2))


def segment_counts(idx: jax.Array, layout: SegmentLayout) -> dict:
    """Fraction of selected rows drawn from each segment."""
    ids = segment_ids(idx, layout)
    L = idx.shape[-1]
    return {
        "pool": float((ids == 0).sum(-1).mean() / L),
        "delta": float((ids == 1).sum(-1).mean() / L),
        "ring": float((ids == 2).sum(-1).mean() / L),
    }


def segment_mass(probs: jax.Array, idx: jax.Array, layout: SegmentLayout,
                 reduce_heads: bool = True) -> dict:
    """Attention mass per segment over the selected candidate rows.

    probs: [B,T,H,L] softmax over selected rows.
    idx:   [B,T,L] selected candidate indices.

    The mask is broadcast as [B,T,1,L] so every head sees the same segment
    assignment. Returns [B,T] per segment when `reduce_heads=True`.
    """
    if probs.ndim != 4:
        raise ValueError(f"probs must be [B,T,H,L], got {probs.shape}")
    if idx.shape != probs[:, :, 0, :].shape:
        raise ValueError(
            f"idx {idx.shape} must match probs [B,T,L] "
            f"{probs[:, :, 0, :].shape}")
    ids = segment_ids(idx, layout)
    out = {}
    for name, sid in (("pool", 0), ("delta", 1), ("ring", 2)):
        mask = (ids == sid)[:, :, None, :].astype(probs.dtype)
        mass = (probs * mask).sum(-1)              # [B,T,H]
        out[name] = mass.mean(-1) if reduce_heads else mass
    return out


def selection_indices(cfg: LoCEDConfig, oracle: jax.Array, logits: jax.Array,
                      valid: jax.Array, boost: jax.Array,
                      L: int | None = None) -> tuple[jax.Array, jax.Array]:
    """Oracle and indexer top-L selections (forced window included)."""
    if L is None:
        L = min(oracle.shape[-1], cfg.window + cfg.top_k)
    return (select(oracle, valid, boost, L),
            select(logits, valid, boost, L))


def selection_summary(idx_oracle: jax.Array, idx_indexer: jax.Array,
                      layout: SegmentLayout) -> dict:
    """Recall@L and set agreement between oracle and indexer selections."""
    L = idx_oracle.shape[-1]
    # [B,T,L,1] == [B,T,1,L] -> [B,T,L,L]; a row is shared if any match.
    eq = idx_oracle[..., :, None] == idx_indexer[..., None, :]
    shared = eq.any(-1).sum(-1).astype(jnp.float32)   # [B,T]
    recall = shared / L
    union = 2 * L - shared
    jaccard = shared / jnp.maximum(union, 1.0)
    return {
        "recall_at_L": float(recall.mean()),
        "jaccard": float(jaccard.mean()),
        "oracle": segment_counts(idx_oracle, layout),
        "indexer": segment_counts(idx_indexer, layout),
    }


def pool_norm_ratio(pool_k: jax.Array, per_token_k: jax.Array,
                    pool_m: int, eps: float = 1e-6) -> jax.Array:
    """||mean(group)|| / mean(||constituent||) per pool entry, [B,P].

    Values near 1.0 mean pooling preserves the constituent scale; values near
    `1/sqrt(pool_m)` are the unnormalized mean-pool shrinkage (D1).
    """
    B, P, _ = pool_k.shape
    groups = per_token_k[:, : P * pool_m].reshape(B, P, pool_m, -1)
    target = jnp.linalg.norm(groups, axis=-1).mean(-1)
    actual = jnp.linalg.norm(pool_k, axis=-1)
    return actual / (target + eps)


def _decode_step_jit(cfg, params):
    return jax.jit(lambda store, token, pos: _decode_step(cfg, params, store,
                                                          token, pos))


def _prefill_jit(cfg, params):
    return jax.jit(lambda tokens: _prefill(cfg, params, tokens))


def _prefill(cfg, params, tokens):
    from .model import prefill
    return prefill(cfg, params, tokens)


def _decode_step(cfg, params, store, token, pos):
    from .model import decode_step
    return decode_step(cfg, params, store, token, pos)


def teacher_forced_decode_logits(cfg: LoCEDConfig, params: dict,
                                 tokens: jax.Array,
                                 prefill_len: int | None = None):
    """Decode through the real store, returning logits aligned to training.

    For positions `t = prefill_len-1 .. T-2`, both the training pass and the
    decode path produce the distribution for token `t+1`:
      * `prefill` at the last prompt position -> position `prefill_len-1`;
      * `decode_step` called with token `t` -> position `t`.
    Returns `(decode_logits [B,T-prefill_len,V], train_positions)`.
    """
    B, T = tokens.shape
    if prefill_len is None:
        prefill_len = (T // 2 // cfg.pool_m) * cfg.pool_m
    if prefill_len <= 0 or prefill_len >= T:
        raise ValueError(f"prefill_len={prefill_len} must be in [1, {T - 1}]")
    if prefill_len % cfg.pool_m != 0:
        raise ValueError(
            f"prefill_len={prefill_len} must be a multiple of pool_m={cfg.pool_m}")
    jpre = _prefill_jit(cfg, params)
    jdec = _decode_step_jit(cfg, params)
    store, logits = jpre(tokens[:, :prefill_len])
    outs = [logits[:, 0]]
    for t in range(prefill_len, T - 1):
        store, logits = jdec(store, tokens[:, t], t)
        outs.append(logits[:, 0])
    dec = jnp.stack(outs, axis=1)                     # [B,T-prefill_len,V]
    train_positions = jnp.arange(prefill_len - 1, T - 1)
    return dec, train_positions


def train_decode_kl_report(cfg: LoCEDConfig, params: dict, tokens: jax.Array,
                           prefill_len: int | None = None,
                           diagnostics: dict | None = None) -> dict:
    """KL(training pass || teacher-forced decode) on the generated span."""
    from .model import forward_train_diagnostics
    if diagnostics is None:
        diagnostics = forward_train_diagnostics(cfg, params, tokens,
                                                prefill_len=prefill_len)
    train_logits = diagnostics["logits"]
    dec_logits, pos = teacher_forced_decode_logits(cfg, params, tokens,
                                                   prefill_len)
    tl = train_logits[:, pos]                         # [B,N,V]
    logp_t = jax.nn.log_softmax(tl, axis=-1)
    logp_d = jax.nn.log_softmax(dec_logits, axis=-1)
    kl = (jnp.exp(logp_t) * (logp_t - logp_d)).sum(-1)
    agree = (jnp.argmax(tl, -1) == jnp.argmax(dec_logits, -1)).mean()
    return {
        "kl": float(kl.mean()),
        "top1_agreement": float(agree),
        "n_positions": int(pos.shape[0]),
        "prefill_len": int(pos[0] + 1),
    }


def build_report(cfg: LoCEDConfig, params: dict, tokens: jax.Array,
                 prefill_len: int | None = None,
                 reuse_p: float = 0.0) -> dict:
    """Full fidelity report for one batch (tiny config by default)."""
    from .model import forward_train_diagnostics

    out = forward_train_diagnostics(cfg, params, tokens,
                                    prefill_len=prefill_len,
                                    reuse_p=reuse_p)
    layout = segment_layout(out["view"])
    idx_o, idx_i = out["idx_oracle"], out["idx_indexer"]

    visits = []
    for info in out["visit_infos"]:
        mass = segment_mass(info["probs"], info["idx"], layout)
        visits.append({
            "visit": int(info["visit"]),
            "mode": str(info["mode"]),
            "segment_mass": {k: float(v.mean()) for k, v in mass.items()},
            "segment_counts": segment_counts(info["idx"], layout),
        })

    report = {
        "config": {
            "hidden_size": cfg.hidden_size,
            "num_layers": cfg.num_layers,
            "pool_m": cfg.pool_m,
            "window": cfg.window,
            "top_k": cfg.top_k,
            "seq_len": int(tokens.shape[1]),
        },
        "layout": asdict(layout),
        "pool_norm_ratio": float(
            pool_norm_ratio(out["pool_k"], out["per_token_k"],
                            cfg.pool_m).mean()),
        "selection": selection_summary(idx_o, idx_i, layout),
        "visits": visits,
        "train_decode": train_decode_kl_report(
            cfg, params, tokens, prefill_len=prefill_len, diagnostics=out),
    }
    return report


def run_baseline(steps: int = 150, batch: int = 4, seed: int = 0,
                 optims: tuple[str, ...] = ("adamw", "muon"),
                 cfg: LoCEDConfig | None = None) -> dict:
    """Train the tiny config with each optimizer on identical batches.

    This is the Phase 0 baseline artifact: the same seed and batch stream for
    AdamW and Muon, plus the fidelity report on the final batch.
    """
    from .model import init_model
    from .train import make_train_step, synthetic_batch, tiny_config

    cfg = tiny_config() if cfg is None else cfg
    runs = {}
    for optim in optims:
        params = init_model(cfg, jax.random.PRNGKey(seed))
        step, opt_state = make_train_step(cfg, params, optim=optim)
        rng = np.random.default_rng(seed)
        history = []
        t0 = time.time()
        tokens = None
        reuse_p = 0.0
        warmup = int(steps * cfg.reuse_warmup_frac) if cfg.reuse_onpolicy else steps
        ramp = max(1, int(steps * cfg.reuse_ramp_frac))
        for i in range(steps):
            tokens = synthetic_batch(rng, batch, cfg.max_seq_len, cfg.vocab_size)
            reuse_p = 0.0 if not cfg.reuse_onpolicy else \
                min(max((i - warmup) / ramp, 0.0), 1.0)
            params, opt_state, metrics = step(params, opt_state, tokens,
                                              reuse_p=reuse_p)
            history.append({k: float(v) for k, v in metrics.items()})
        elapsed = time.time() - t0
        report = build_report(cfg, params, tokens, reuse_p=reuse_p)
        runs[optim] = {
            "steps": steps,
            "batch": batch,
            "seed": seed,
            "seconds": elapsed,
            "history": history,
            "report": report,
        }
    return {
        "kind": "loced_phase0_baseline",
        "date": "2026-11-01",
        "config": {
            "vocab_size": cfg.vocab_size,
            "hidden_size": cfg.hidden_size,
            "max_seq_len": cfg.max_seq_len,
            "pool_m": cfg.pool_m,
            "window": cfg.window,
            "top_k": cfg.top_k,
            "pool_norm_preserve": cfg.pool_norm_preserve,
            "lazy_view": cfg.lazy_view,
            "train_delta": cfg.train_delta,
            "reuse_onpolicy": cfg.reuse_onpolicy,
        },
        "runs": runs,
    }


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="LoCED Phase 0 diagnostics")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--optims", nargs="+", default=["adamw", "muon"])
    ap.add_argument("--out", default="validation/diagnostics_baseline.json")
    args = ap.parse_args(argv)
    report = run_baseline(steps=args.steps, batch=args.batch, seed=args.seed,
                          optims=tuple(args.optims))
    write_report(args.out, report)
    for name, run in report["runs"].items():
        last = run["history"][-1]
        print(f"{name}: loss {last['loss']:.4f} ce {last['ce']:.4f} "
              f"kl {last['kl']:.5f} ({run['seconds']:.1f}s)")
        print(f"  pool_norm_ratio {run['report']['pool_norm_ratio']:.4f} "
              f"recall@L {run['report']['selection']['recall_at_L']:.4f} "
              f"train/decode KL {run['report']['train_decode']['kl']:.5f}")
    print(f"wrote {args.out}")


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (jax.Array, np.ndarray)):
        return np.asarray(value).item()
    if isinstance(value, (float, int, bool, str)) or value is None:
        return value
    return float(value)


def write_report(path: str | Path, report: dict) -> None:
    """Write a JSON report, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(_to_jsonable(report), f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
