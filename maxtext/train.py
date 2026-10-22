"""LoCED training on MaxText v0.2.5.

Framework components are deliberately *MaxText's*, not reimplementations:

* ``pyconfig`` + :class:`LoCEDMaxTextConfig` for configuration;
* ``maxtext_utils.get_mesh_from_config`` and MaxText's logical-axis rules
  for the device mesh;
* ``maxtext_sharding.nnx_construct_named_sharding`` for parameter placement;
* ``maxtext_utils.create_learning_rate_schedule`` for the LR schedule;
* ``maxtext.optimizers.optimizers.get_optimizer`` with
  ``opt_type="muon"`` and ``muon_type="maxtext_muon"`` for the optimizer;
* Orbax (the checkpoint backend MaxText itself uses) for checkpoints.

The only LoCED-owned pieces are the architecture and its loss
(:mod:`loced.model` / :mod:`loced.maxtext.modeling`).  The trainer never
imports ``loced.muon`` and rejects non-MaxText Muon configurations.

Usage::

    python -m loced.maxtext.train \
        loced/maxtext/configs/loced_tiny.yml \
        steps=20 run_name=loced-tiny

MaxText's paper-scale data pipeline is intentionally not wired here: the
entry point accepts any iterable of ``[batch, seq]`` int32 token arrays via
``train(..., batches=...)``, so a Grain/MaxText iterator can be plugged in
without touching the model or optimizer code.  ``main`` falls back to the
same learnable synthetic task used by the CPU smoke tests.
"""

from __future__ import annotations

import os
import sys
from typing import Iterable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx
from flax.core.spmd import logical_axis_rules

from maxtext.configs import pyconfig
from maxtext.optimizers.optimizers import get_optimizer
from maxtext.utils import max_logging
from maxtext.utils import maxtext_utils
from maxtext.utils import sharding as maxtext_sharding

from .config import LoCEDMaxTextConfig, loced_config_from_maxtext
from .modeling import LoCEDModel, build_model


# ---------------------------------------------------------------------------
# model / mesh
# ---------------------------------------------------------------------------


def create_model_and_mesh(config) -> tuple[LoCEDModel, jax.sharding.Mesh]:
    """Build LoCED and place its parameters with MaxText's sharding helper."""
    loced_cfg = loced_config_from_maxtext(config)
    mesh = maxtext_utils.get_mesh_from_config(config)
    model = build_model(loced_cfg, seed=int(getattr(config, "init_weights_seed", 0)))

    # MaxText resolves logical-axis metadata into physical NamedShardings.
    # Do it under the mesh so Flax's eager-sharding checks see the same mesh.
    with jax.set_mesh(mesh), logical_axis_rules(config.logical_axis_rules):
        state = nnx.state(model)
        shardings = maxtext_sharding.nnx_construct_named_sharding(state, mesh)
        pure = jax.tree.map(
            lambda value, sharding: jax.device_put(value, sharding),
            nnx.to_pure_dict(state),
            nnx.to_pure_dict(shardings),
        )
        nnx.update(model, nnx.State(pure))
    return model, mesh


# ---------------------------------------------------------------------------
# optimizer (MaxText Muon only)
# ---------------------------------------------------------------------------


def create_optimizer(config, model: LoCEDModel, mesh):
    """Create the MaxText LR schedule + Muon optimizer.

    ``maxtext_muon`` is required: it understands the ``out_sharding`` metadata
    on every parameter and performs the sharded Newton-Schulz iteration.  The
    portable ``optax_muon`` path is intentionally rejected so a LoCED run
    cannot silently fall back to a different optimizer implementation.
    """
    if config.opt_type != "muon":
        raise ValueError(
            "LoCED MaxText training requires opt_type='muon' (MaxText Muon), "
            f"got {config.opt_type!r}."
        )
    muon_type = getattr(config, "muon_type", None)
    if muon_type not in ("maxtext", "maxtext_muon"):
        raise ValueError(
            "LoCED MaxText training requires muon_type='maxtext_muon' "
            f"(MaxText's sharding-aware Muon), got {muon_type!r}. "
            "optax_muon is not supported by this trainer."
        )

    schedule = maxtext_utils.create_learning_rate_schedule(config)
    # MaxText's Muon reads logical-axis rules + physical shardings from the
    # model, so create it under the same mesh context as the parameters.
    with jax.set_mesh(mesh), logical_axis_rules(config.logical_axis_rules):
        tx = get_optimizer(config, schedule, model, mesh)
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    return schedule, optimizer


# ---------------------------------------------------------------------------
# train step
# ---------------------------------------------------------------------------


def loss_fn(model: LoCEDModel, tokens: jax.Array):
    """Module-level loss so ``nnx.value_and_grad`` sees an unbound function."""
    return model.train_loss(tokens)


@nnx.jit
def train_step(model: LoCEDModel, optimizer: nnx.Optimizer, tokens: jax.Array):
    """One jitted value-and-grad step using MaxText's Muon update."""
    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, tokens)
    optimizer.update(model, grads)
    return loss, metrics


# ---------------------------------------------------------------------------
# Orbax checkpoints (same backend MaxText uses)
# ---------------------------------------------------------------------------


def _checkpoint_state(model: LoCEDModel, optimizer: nnx.Optimizer) -> dict:
    return {
        "model": nnx.to_pure_dict(nnx.state(model)),
        "optimizer": nnx.to_pure_dict(nnx.state(optimizer)),
    }


def save_checkpoint(path: str, model: LoCEDModel, optimizer: nnx.Optimizer) -> None:
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(path, _checkpoint_state(model, optimizer), force=True)
    checkpointer.wait_until_finished()


def load_checkpoint(
    path: str, model: LoCEDModel, optimizer: nnx.Optimizer, mesh
) -> int | None:
    """Restore model/optimizer in place; return the number of completed steps."""
    if not os.path.exists(path):
        return None
    current = _checkpoint_state(model, optimizer)
    target = jax.tree.map(
        lambda value: jax.ShapeDtypeStruct(
            np.shape(value),
            np.asarray(value).dtype,
            sharding=getattr(value, "sharding", None),
        ),
        current,
    )
    restored = ocp.StandardCheckpointer().restore(path, target=target)
    with jax.set_mesh(mesh):
        nnx.update(model, nnx.State(restored["model"]))
        nnx.update(optimizer, nnx.State(restored["optimizer"]))
    return int(optimizer.step.get_value())


def checkpoint_path(config) -> str:
    """Return MaxText's resolved ``checkpoint_dir``.

    MaxText fills ``checkpoint_dir`` from ``base_output_directory`` /
    ``run_name`` when checkpointing is enabled, so the explicit branch is only
    used for configs that bypass that resolution.
    """
    explicit = getattr(config, "checkpoint_dir", "") or ""
    if explicit:
        return explicit
    root = getattr(config, "base_output_directory", "") or "maxtext_output"
    return os.path.join(root, getattr(config, "run_name", "loced"), "checkpoints")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def synthetic_batches(
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    steps: int,
    seed: int = 0,
    rule_p: float = 0.85,
) -> Iterable[jax.Array]:
    """Learnable next-token task (CPU smoke only; replace with MaxText data)."""
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        tok = np.zeros((batch_size, seq_len), dtype=np.int32)
        tok[:, 0] = rng.integers(0, vocab_size, size=batch_size)
        for t in range(1, seq_len):
            step = (tok[:, t - 1] + 1) % vocab_size
            noise = rng.integers(0, vocab_size, size=batch_size)
            tok[:, t] = np.where(rng.random(batch_size) < rule_p, step, noise)
        yield jnp.asarray(tok)


# ---------------------------------------------------------------------------
# loop
# ---------------------------------------------------------------------------


def train(
    config,
    batches: Iterable[jax.Array] | None = None,
    log_every: int = 1,
) -> tuple[LoCEDModel, nnx.Optimizer, dict]:
    """Run ``config.steps`` LoCED steps with MaxText Muon.

    Args:
      config: initialized MaxText config (``LoCEDMaxTextConfig``).
      batches: iterable of int32 ``[batch, seq]`` token arrays.  ``None`` uses
        the synthetic smoke task.
      log_every: log interval in steps.

    Returns:
      ``(model, optimizer, last_metrics)``.
    """
    loced_cfg = loced_config_from_maxtext(config)
    model, mesh = create_model_and_mesh(config)
    schedule, optimizer = create_optimizer(config, model, mesh)

    steps = int(config.steps)
    ckpt_dir = checkpoint_path(config) if getattr(config, "enable_checkpointing", False) else None
    start_step = 0
    if ckpt_dir:
        restored_step = load_checkpoint(ckpt_dir, model, optimizer, mesh)
        if restored_step is not None:
            # load_checkpoint returns optimizer.step, i.e. the number of
            # completed updates, which is exactly the next step index.
            start_step = restored_step
            max_logging.log(f"resumed LoCED from {ckpt_dir} after {restored_step} steps")

    if batches is None:
        batch_size = max(1, int(config.per_device_batch_size))
        batches = synthetic_batches(
            loced_cfg.vocab_size,
            loced_cfg.max_seq_len,
            batch_size,
            max(0, steps - start_step),
            seed=int(getattr(config, "init_weights_seed", 0)),
        )

    last_metrics: dict = {}
    for step, tokens in zip(range(start_step, steps), batches):
        loss, metrics = train_step(model, optimizer, tokens)
        last_metrics = {k: float(v) for k, v in metrics.items()}
        if log_every > 0 and (step % log_every == 0 or step == steps - 1):
            lr = float(schedule(step))
            max_logging.log(
                f"step {step:6d} loss {float(loss):.6f} "
                f"ce {last_metrics['ce']:.6f} kl {last_metrics['kl']:.6f} "
                f"moe_aux {last_metrics['moe_aux']:.6f} lr {lr:.3e}"
            )
        if ckpt_dir and (step + 1) % max(1, int(config.checkpoint_period)) == 0:
            save_checkpoint(ckpt_dir, model, optimizer)

    if ckpt_dir and steps > start_step:
        save_checkpoint(ckpt_dir, model, optimizer)
    return model, optimizer, last_metrics


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv if argv is None else argv)
    config = pyconfig.initialize(argv, config_class=LoCEDMaxTextConfig)
    _, _, metrics = train(config)
    max_logging.log(f"finished LoCED run: {metrics}")


if __name__ == "__main__":
    main()
