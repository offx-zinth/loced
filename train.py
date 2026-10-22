"""Training loop for LoCED (pure JAX + optax, MaxText-style jitted step).

  * `tiny_config()` - CPU-scale config used by tests and the smoke run.
  * `synthetic_batch()` - a learnable synthetic LM task: next token is
    (prev + 1) mod V with probability `rule_p`, else uniform noise.
  * `make_train_step()` - jitted value_and_grad step; returns (params,
    opt_state, metrics) where metrics includes the loss components
    (ce / ce_aux / indexer KL / MoE aux).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .config import LoCEDConfig
from .model import count_params, forward_train, init_model
from .muon import make_muon_mask, muon_optimizer


def tiny_config() -> LoCEDConfig:
    """CPU smoke-test scale (jittable, seconds per train run)."""
    return LoCEDConfig(
        vocab_size=64, hidden_size=64, max_seq_len=64, max_decode_len=32,
        num_attention_heads=4, head_dim=16, kv_dim=16, pool_m=2,
        window=16, top_k=16, indexer_heads=2, indexer_dim=32,
        indexer_block=4, hier_blocks=4,
        num_experts=4, num_shared_experts=1, top_k_experts=2,
        intermediate_size=128,
    )


def synthetic_batch(rng: np.random.Generator, batch: int, seq: int,
                    vocab: int, rule_p: float = 0.85) -> jnp.ndarray:
    """Tokens with a learnable local rule: P(next = prev + 1) = rule_p."""
    tok = np.zeros((batch, seq), dtype=np.int32)
    tok[:, 0] = rng.integers(0, vocab, size=batch)
    for t in range(1, seq):
        step = (tok[:, t - 1] + 1) % vocab
        noise = rng.integers(0, vocab, size=batch)
        take_rule = rng.random(batch) < rule_p
        tok[:, t] = np.where(take_rule, step, noise)
    return jnp.asarray(tok)


# Per-optimizer defaults: Muon's orthogonalized step is ~3x larger per unit
# LR than AdamW's on this model, so the matched LR is lower (measured on the
# tiny synthetic task: muon@1e-3 ~ adamw@3e-3; muon@6e-3 diverges).
DEFAULT_LR = {"muon": 1e-3, "adamw": 3e-3}


def make_optimizer(cfg: LoCEDConfig, params: dict, optim: str = "muon",
                   lr: float | None = None, weight_decay: float = 0.01):
    """Optimizer factory.

    optim="muon"  -> Muon on matrix weights (orthogonalized momentum) +
                      AdamW on embeddings/head/router/1-D params.
    optim="adamw" -> the previous baseline (AdamW everywhere).
    """
    lr = DEFAULT_LR[optim] if lr is None else lr
    if optim == "muon":
        mask = make_muon_mask(params)
        return muon_optimizer(mask, learning_rate=lr, weight_decay=weight_decay)
    if optim == "adamw":
        return optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=lr, weight_decay=weight_decay),
        )
    raise ValueError(f"unknown optim {optim!r} (use 'muon' or 'adamw')")


def make_train_step(cfg: LoCEDConfig, params: dict, lr: float | None = None,
                    optim: str = "muon"):
    """Returns (step_fn, params, opt_state). step_fn(params, opt_state, tokens)
    -> (params, opt_state, metrics dict)."""
    optimizer = make_optimizer(cfg, params, optim=optim, lr=lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, tokens):
        def loss_fn(p):
            out = forward_train(cfg, p, tokens)
            metrics = {k: out[k] for k in ("loss", "ce", "ce_aux", "kl",
                                           "moe_aux")}
            return out["loss"], metrics

        (loss, metrics), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(params)
        updates, opt_state2 = optimizer.update(grads, opt_state, params)
        params2 = optax.apply_updates(params, updates)
        return params2, opt_state2, metrics

    return step, opt_state


def train_smoke(steps: int = 200, batch: int = 4, seed: int = 0,
                verbose: bool = False, optim: str = "muon",
                lr: float | None = None):
    """Tiny end-to-end run used by tests and `python -m loced.train`."""
    cfg = tiny_config()
    params = init_model(cfg, jax.random.PRNGKey(seed))
    step, opt_state = make_train_step(cfg, params, lr=lr, optim=optim)
    rng = np.random.default_rng(seed)
    history = []
    for i in range(steps):
        tokens = synthetic_batch(rng, batch, cfg.max_seq_len, cfg.vocab_size)
        params, opt_state, metrics = step(params, opt_state, tokens)
        history.append(metrics)
        if verbose and (i % 25 == 0 or i == steps - 1):
            print(f"step {i:4d}  loss {float(metrics['loss']):.4f}  "
                  f"ce {float(metrics['ce']):.4f}  "
                  f"kl {float(metrics['kl']):.5f}")
    return cfg, params, history


def main():
    import argparse
    ap = argparse.ArgumentParser(description="LoCED tiny smoke train")
    ap.add_argument("--optim", default="muon", choices=["muon", "adamw"])
    ap.add_argument("--lr", type=float, default=None,
                    help="defaults: muon 1e-3, adamw 3e-3")
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()

    cfg = tiny_config()
    params = init_model(cfg, jax.random.PRNGKey(0))
    lr = args.lr if args.lr is not None else DEFAULT_LR[args.optim]
    print(f"LoCED tiny config: {count_params(params)} parameters")
    print(f"optimiser: {args.optim}  lr: {lr}")
    _, _, history = train_smoke(steps=args.steps, verbose=True,
                                 optim=args.optim, lr=lr)
    first, last = history[0], history[-1]
    print(f"uniform loss = ln({cfg.vocab_size}) = "
          f"{np.log(cfg.vocab_size):.4f}")
    print(f"loss {float(first['loss']):.4f} -> {float(last['loss']):.4f}")
    print(f"indexer KL {float(first['kl']):.4f} -> {float(last['kl']):.4f}")


if __name__ == "__main__":
    main()
