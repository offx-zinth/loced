"""LoCED as a Flax NNX module (MaxText v0.2.5 style).

Why an NNX shell around the reference JAX core?
------------------------------------------------
MaxText v0.2.5 is NNX-first: models are ``flax.nnx.Module`` graphs registered
as parameters, sharded through logical-axis metadata, optimized by
``nnx.Optimizer`` and checkpointed as NNX state.  LoCED's *algorithm* is
already pure JAX (``loced.model`` / ``loced.attention`` / ``loced.cache``), so
the migration keeps that code untouched and adds the framework-facing shell:

* every parameter is its own ``nnx.Param`` (not one opaque dict), so Muon,
  AdamW, sharding, checkpointing and parameter counting see real leaves;
* logical-axis metadata (``out_sharding``) is attached to every parameter for
  MaxText's ``nnx_construct_named_sharding``;
* ``__call__`` follows the MaxText ``Transformer.__call__`` signature and
  returns logits; the training auxiliaries are sown as
  ``nnx.Intermediate`` variables;
* the custom store / prefill / decode entry points stay explicit methods
  because MaxText's standard KV cache cannot represent LoCED's shared
  pool+delta+ring store.

Everything algorithmic is still the exact reference implementation: the NNX
module only reassembles the reference pytree and calls ``forward_train`` /
``prefill`` / ``decode_step``.  This is deliberate - it gives bit-for-bit
parity with the CPU reference and keeps the custom research code in JAX, as
requested.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

try:  # flax is an optional dependency of the reference CPU package
    from flax import nnx
except ImportError as exc:  # pragma: no cover - exercised only without flax
    raise ImportError(
        "loced.maxtext.modeling requires Flax NNX (pip install flax>=0.12.6)."
    ) from exc

try:  # repo-relative import (normal use)
    from ..config import LoCEDConfig
    from ..model import decode_step, forward_train, generate, init_model, prefill
    from ..model import prefill_extend
    from ..cache import new_store
except ImportError:  # when this file is copied into a MaxText checkout
    from loced.config import LoCEDConfig
    from loced.model import decode_step, forward_train, generate, init_model
    from loced.model import prefill, prefill_extend
    from loced.cache import new_store


# MaxText model modes (kept as strings so this module does not import MaxText).
MODEL_MODE_TRAIN = "train"
MODEL_MODE_PREFILL = "prefill"
MODEL_MODE_AUTOREGRESSIVE = "autoregressive"


def _p(value, axes: tuple | None = None) -> nnx.Param:
    """Create a parameter and attach MaxText logical-axis metadata.

    The metadata is attached *after* construction on purpose: Flax 0.12 eager
    sharding tries to resolve ``out_sharding`` at construction time, which
    requires a live mesh.  MaxText resolves shardings later via
    ``nnx_construct_named_sharding``, which reads this metadata.
    """
    param = nnx.Param(value)
    if axes is not None:
        param.set_metadata("out_sharding", axes)
    return param


def _rand(rngs: nnx.Rngs, shape, std: float, dtype=jnp.float32) -> jax.Array:
    return jax.random.normal(rngs.params(), shape, dtype) * std


class LoCEDAttention(nnx.Module):
    """Per-layer attention projections (Q, exact K/V, output)."""

    def __init__(self, cfg: LoCEDConfig, rngs: nnx.Rngs):
        d, kv = cfg.hidden_size, cfg.kv_dim
        kd = cfg.num_attention_heads * cfg.head_dim
        std = cfg.init_std
        self.W_q = _p(_rand(rngs, (d, kd), std), ("embed", "heads"))
        self.W_fx = _p(_rand(rngs, (d, kv), std), ("embed", "kv"))
        self.W_fv = _p(_rand(rngs, (d, kv), std), ("embed", "kv"))
        self.W_o = _p(_rand(rngs, (kd, d), std), ("heads", "embed"))


class _SharedExpert(nnx.Module):
    """Always-on SwiGLU expert (AdamW-friendly 2-D weights)."""

    def __init__(self, cfg: LoCEDConfig, rngs: nnx.Rngs):
        d, I = cfg.hidden_size, cfg.intermediate_size
        std = cfg.init_std
        self.w1 = _p(_rand(rngs, (d, I), std), ("embed", "mlp"))
        self.w3 = _p(_rand(rngs, (d, I), std), ("embed", "mlp"))
        self.w2 = _p(_rand(rngs, (I, d), std), ("mlp", "embed"))

    def reference_dict(self) -> dict:
        return {
            "Ws1": self.w1.get_value(),
            "Ws3": self.w3.get_value(),
            "Ws2": self.w2.get_value(),
        }


class _RoutedExperts(nnx.Module):
    """Top-k routed SwiGLU experts.

    Names follow MaxText's MoE/Muon conventions (``wi_0`` / ``wi_1`` / ``wo`` /
    ``gate``), so ``maxtext.utils.muon_utils.transform_logic`` assigns
    ``reduction_axis=(-2,), output_axis=(-1,)`` per expert instead of
    orthogonalizing across the expert axis.
    """

    def __init__(self, cfg: LoCEDConfig, rngs: nnx.Rngs):
        d, I, E = cfg.hidden_size, cfg.intermediate_size, cfg.num_experts
        std = cfg.init_std
        self.gate = _p(_rand(rngs, (d, E), std), ("embed_router", "exp"))
        self.wi_0 = _p(_rand(rngs, (E, d, I), std), ("exp", "embed_moe", "mlp_moe"))
        self.wi_1 = _p(_rand(rngs, (E, d, I), std), ("exp", "embed_moe", "mlp_moe"))
        self.wo = _p(_rand(rngs, (E, I, d), std), ("exp", "mlp_moe", "embed_moe"))

    def reference_dict(self) -> dict:
        return {
            "W_gate": self.gate.get_value(),
            "We1": self.wi_0.get_value(),
            "We3": self.wi_1.get_value(),
            "We2": self.wo.get_value(),
        }


class LoCEDMoE(nnx.Module):
    """DeepSeek-style shared + routed SwiGLU experts for one physical layer."""

    def __init__(self, cfg: LoCEDConfig, rngs: nnx.Rngs):
        self.routed_moe = _RoutedExperts(cfg, rngs)
        self.shared_experts = _SharedExpert(cfg, rngs)

    def reference_dict(self) -> dict:
        return {
            **self.routed_moe.reference_dict(),
            **self.shared_experts.reference_dict(),
        }


class LoCEDLayer(nnx.Module):
    """One physical layer: attention + MoE + two pre-norms.

    The same module is run at visit ``k`` (encode / pass A) and visit
    ``V + k`` (pass B); the visit index selects ``cond`` / ``gate_a`` /
    ``gate_m`` on the parent model.  Weights are tied across roles, exactly as
    in the reference.
    """

    def __init__(self, cfg: LoCEDConfig, rngs: nnx.Rngs):
        d = cfg.hidden_size
        self.norm_a = _p(jnp.ones((d,), jnp.float32), ("norm",))
        self.norm_m = _p(jnp.ones((d,), jnp.float32), ("norm",))
        self.attn = LoCEDAttention(cfg, rngs)
        self.moe = LoCEDMoE(cfg, rngs)

    def reference_dict(self) -> dict:
        return {
            "norm_a": self.norm_a.get_value(),
            "norm_m": self.norm_m.get_value(),
            "W_q": self.attn.W_q.get_value(),
            "W_fx": self.attn.W_fx.get_value(),
            "W_fv": self.attn.W_fv.get_value(),
            "W_o": self.attn.W_o.get_value(),
            **self.moe.reference_dict(),
        }


class LoCEDModel(nnx.Module):
    """LoCED model graph: 4 physical layers + model-level shared state.

    Parameters
    ----------
    cfg:
        The reference :class:`loced.config.LoCEDConfig`.
    rngs:
        NNX RNG stream (``nnx.Rngs(seed)``).
    mesh:
        Optional JAX mesh.  Kept for parity with MaxText model constructors;
        sharding is resolved from the ``out_sharding`` metadata, so the mesh
        is not required to build the module.
    """

    def __init__(self, cfg: LoCEDConfig, rngs: nnx.Rngs):
        self.cfg = cfg
        d, kv, idim = cfg.hidden_size, cfg.kv_dim, cfg.indexer_dim
        std = cfg.init_std

        # Names intentionally match MaxText's Muon exclusions: `embedding` and
        # `logits_dense` stay on AdamW; `visit_embedding` too.
        self.token_embedding = _p(
            _rand(rngs, (cfg.vocab_size, d), std), ("vocab", "embed")
        )
        self.logits_dense = _p(
            _rand(rngs, (d, cfg.vocab_size), std), ("embed", "vocab")
        )

        self.W_bk = _p(_rand(rngs, (d, kv), std), ("embed", "kv"))
        self.W_bv = _p(_rand(rngs, (d, kv), std), ("embed", "kv"))
        self.W_ik = _p(_rand(rngs, (kv, idim), std), ("kv", "kv_head_dim"))
        self.W_iq = _p(
            _rand(rngs, (d, cfg.indexer_heads * idim), std), ("embed", "heads")
        )

        nv = cfg.num_visits
        self.visit_embedding = _p(_rand(rngs, (nv, d), std), (None, "embed"))
        self.gate_a = _p(jnp.zeros((nv,), jnp.float32), (None,))
        self.gate_m = _p(jnp.zeros((nv,), jnp.float32), (None,))
        self.norm_f = _p(jnp.ones((d,), jnp.float32), ("norm",))

        # Use an NNX dict with string keys rather than nnx.List: MaxText's
        # Muon path classifier (muon_utils.transform_logic) expects every
        # path segment to be a string.  nnx.List lowers to integer dict keys
        # in nnx.to_pure_dict, which breaks that classifier.
        self.cells = nnx.Dict(
            {str(i): LoCEDLayer(cfg, rngs) for i in range(cfg.num_layers)}
        )

    # ------------------------------------------------------------------
    # parameter <-> reference pytree
    # ------------------------------------------------------------------
    def to_reference_params(self) -> dict:
        """Assemble the exact pytree consumed by ``loced.model``."""
        return {
            "embed": self.token_embedding.get_value(),
            "head": self.logits_dense.get_value(),
            "W_bk": self.W_bk.get_value(),
            "W_bv": self.W_bv.get_value(),
            "W_ik": self.W_ik.get_value(),
            "W_iq": self.W_iq.get_value(),
            "cond": self.visit_embedding.get_value(),
            "gate_a": self.gate_a.get_value(),
            "gate_m": self.gate_m.get_value(),
            "norm_f": self.norm_f.get_value(),
            "cells": tuple(cell.reference_dict() for cell in self.cells.values()),
        }

    def load_reference_params(self, params: dict) -> "LoCEDModel":
        """Copy a reference pytree into the NNX parameter graph."""
        self.token_embedding.set_value(params["embed"])
        self.logits_dense.set_value(params["head"])
        self.W_bk.set_value(params["W_bk"])
        self.W_bv.set_value(params["W_bv"])
        self.W_ik.set_value(params["W_ik"])
        self.W_iq.set_value(params["W_iq"])
        self.visit_embedding.set_value(params["cond"])
        self.gate_a.set_value(params["gate_a"])
        self.gate_m.set_value(params["gate_m"])
        self.norm_f.set_value(params["norm_f"])
        for cell, src in zip(self.cells.values(), params["cells"]):
            cell.norm_a.set_value(src["norm_a"])
            cell.norm_m.set_value(src["norm_m"])
            cell.attn.W_q.set_value(src["W_q"])
            cell.attn.W_fx.set_value(src["W_fx"])
            cell.attn.W_fv.set_value(src["W_fv"])
            cell.attn.W_o.set_value(src["W_o"])
            cell.moe.routed_moe.gate.set_value(src["W_gate"])
            cell.moe.shared_experts.w1.set_value(src["Ws1"])
            cell.moe.shared_experts.w3.set_value(src["Ws3"])
            cell.moe.shared_experts.w2.set_value(src["Ws2"])
            cell.moe.routed_moe.wi_0.set_value(src["We1"])
            cell.moe.routed_moe.wi_1.set_value(src["We3"])
            cell.moe.routed_moe.wo.set_value(src["We2"])
        return self

    @classmethod
    def from_reference(
        cls, cfg: LoCEDConfig, params: dict, rngs: nnx.Rngs | None = None
    ) -> "LoCEDModel":
        """Build an NNX model and load an existing reference pytree."""
        model = cls(cfg, rngs or nnx.Rngs(0))
        return model.load_reference_params(params)

    def count_params(self) -> int:
        return int(
            sum(x.size for x in jax.tree_util.tree_leaves(self.to_reference_params()))
        )

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def train_output(self, tokens: jax.Array) -> dict:
        """Reference training forward (loss components + logits)."""
        return forward_train(self.cfg, self.to_reference_params(), tokens)

    def train_loss(self, tokens: jax.Array):
        """Return ``(total_loss, metrics)`` for a custom MaxText trainer."""
        out = self.train_output(tokens)
        metrics = {k: out[k] for k in ("loss", "ce", "ce_aux", "kl", "moe_aux")}
        return out["loss"], metrics

    def __call__(
        self,
        decoder_input_tokens: jax.Array,
        decoder_positions: jax.Array | None = None,
        decoder_segment_ids: jax.Array | None = None,
        cache: Any | None = None,
        enable_dropout: bool = True,
        model_mode: str = MODEL_MODE_TRAIN,
        **kwargs,
    ):
        """MaxText ``Transformer``-compatible entry point.

        Training returns logits [B, T, V] and sows the auxiliaries as
        ``nnx.Intermediate`` variables.  ``loced_ce_aux`` is the deep
        supervision term; ``indexer_loss`` and ``moe_lb_loss`` use MaxText's
        standard suffix names so MaxText's own ``loss_fn`` can pick them up.
        """
        del decoder_segment_ids, enable_dropout, kwargs
        tokens = decoder_input_tokens
        if model_mode == MODEL_MODE_TRAIN:
            out = self.train_output(tokens)
            self.sow(nnx.Intermediate, "loced_loss", out["loss"])
            self.sow(nnx.Intermediate, "loced_ce", out["ce"])
            self.sow(nnx.Intermediate, "loced_ce_aux", out["ce_aux"])
            self.sow(nnx.Intermediate, "loced_kl", out["kl"])
            self.sow(nnx.Intermediate, "loced_moe_aux", out["moe_aux"])
            # MaxText-native auxiliary names:
            self.sow(
                nnx.Intermediate,
                "indexer_loss",
                self.cfg.indexer_kl_coef * out["kl"],
            )
            self.sow(
                nnx.Intermediate,
                "moe_lb_loss",
                self.cfg.moe_aux_weight * out["moe_aux"],
            )
            return out["logits"]

        if model_mode == MODEL_MODE_PREFILL:
            _, logits = self.prefill_tokens(tokens)
            return logits

        if model_mode == MODEL_MODE_AUTOREGRESSIVE:
            if cache is None:
                raise ValueError(
                    "LoCED autoregressive mode needs its shared store; pass "
                    "cache=model.init_cache(...) or call model.decode_token()."
                )
            pos = (
                int(decoder_positions[0, -1])
                if decoder_positions is not None
                else int(tokens.shape[1] - 1)
            )
            token = tokens[:, -1]
            _, logits = self.decode_token(cache, token, pos)
            return logits

        raise ValueError(f"unknown model_mode {model_mode!r}")

    # ------------------------------------------------------------------
    # inference: the custom shared store is explicit (not MaxText's KVCache)
    # ------------------------------------------------------------------
    def init_cache(self, batch_size: int, dtype=jnp.float32) -> dict:
        return new_store(self.cfg, batch_size, dtype)

    def prefill_tokens(self, tokens: jax.Array):
        """Prompt pass -> ``(store, logits)`` (first generated token)."""
        return prefill(self.cfg, self.to_reference_params(), tokens)

    def prefill_extend_tokens(
        self, store: dict, tokens: jax.Array, pos0: int
    ):
        """Multi-turn prompt extension -> ``(store, logits)``."""
        return prefill_extend(
            self.cfg, self.to_reference_params(), store, tokens, pos0
        )

    def decode_token(self, store: dict, token: jax.Array, pos: int):
        """One generated token -> ``(store, logits)``."""
        return decode_step(self.cfg, self.to_reference_params(), store, token, pos)

    def generate_tokens(self, prompt: jax.Array, max_new: int) -> jax.Array:
        return generate(self.cfg, self.to_reference_params(), prompt, max_new)


def build_model(
    cfg: LoCEDConfig,
    seed: int = 0,
    reference_params: dict | None = None,
) -> LoCEDModel:
    """Factory used by tests and by the MaxText trainer."""
    model = LoCEDModel(cfg, nnx.Rngs(seed))
    if reference_params is not None:
        model.load_reference_params(reference_params)
    return model


def reference_init(cfg: LoCEDConfig, seed: int = 0) -> dict:
    """Convenience for parity tests / checkpoint conversion."""
    return init_model(cfg, jax.random.PRNGKey(seed))
