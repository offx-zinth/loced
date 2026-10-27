"""MaxText v0.2.5 integration tests for LoCED.

Run in the MaxText/Flax environment::

    python -m pytest loced/maxtext/test_integration.py -q

The CPU reference environment can run the architecture/parity/config tests;
Muon and YAML tests are skipped when MaxText is not importable.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flax import nnx

from ..config import LoCEDConfig
from ..model import count_params, decode_step, forward_train, init_model, prefill
from ..train import synthetic_batch, tiny_config
from .config import loced_config_from_mapping, loced_config_from_maxtext
from .modeling import build_model

# ---------------------------------------------------------------------------
# MaxText is optional in the CPU reference environment.  Prefer the real
# MaxText utilities; only fall back to CPU stubs when the full MaxText
# dependency stack is unavailable (the local reference environment).  Set
# LOCED_REQUIRE_REAL_MAXTEXT_UTILS=1 in a GPU/TPU run to make stubbing fatal.
# ---------------------------------------------------------------------------

try:
    import maxtext  # noqa: F401

    _HAS_MAXTEXT = True
except ImportError:
    _HAS_MAXTEXT = False

REAL_MAXTEXT_UTILS = False

if _HAS_MAXTEXT:
    from maxtext.configs import pyconfig

    try:
        # These pull in the full data/checkpoint stack.  If the environment is
        # complete (MaxText container / Kaggle GPU install), use the real code.
        import maxtext.utils.maxtext_utils as _maxtext_utils  # noqa: F401
        import maxtext.utils.model_creation_utils  # noqa: F401

        REAL_MAXTEXT_UTILS = True
    except Exception as _exc:  # pragma: no cover - CPU fallback only
        print(
            f"[loced.maxtext.tests] real MaxText utilities unavailable "
            f"({_exc!r}); using CPU stubs"
        )
        for _name in ("maxtext.utils.maxtext_utils", "maxtext.utils.model_creation_utils"):
            sys.modules.pop(_name, None)
            sys.modules[_name] = types.ModuleType(_name)
        import optax

        _maxtext_utils = sys.modules["maxtext.utils.maxtext_utils"]
        _maxtext_utils.get_mesh_from_config = lambda config, devices=None: jax.sharding.Mesh(
            np.array(jax.devices()).reshape((1,) * len(config.mesh_axes)),
            tuple(config.mesh_axes),
        )
        _maxtext_utils.create_learning_rate_schedule = lambda config: optax.constant_schedule(
            config.learning_rate
        )

    # Import after the fallback so Muon's module-level dependency on
    # maxtext_utils/model_creation_utils is satisfied by the stubs too.
    from maxtext.utils.muon_utils import transform_logic

    maxtext_train = importlib.import_module("loced.maxtext.train")
else:  # pragma: no cover
    pyconfig = None
    transform_logic = None
    maxtext_train = None


CFG = tiny_config()
KEY = jax.random.PRNGKey(0)
PARITY_TOL = float(os.environ.get("LOCED_PARITY_TOL", "0"))


def _assert_maxdiff(name: str, actual, expected, tol: float | None = None) -> None:
    """Assert |actual - expected| <= tol (default 0, i.e. exact parity)."""
    diff = float(jnp.max(jnp.abs(actual - expected)))
    limit = PARITY_TOL if tol is None else tol
    assert diff <= limit, f"{name}: maxdiff {diff:g} > tolerance {limit:g}"


@pytest.fixture(scope="module")
def reference_params():
    return init_model(CFG, KEY)


@pytest.fixture(scope="module")
def nnx_model(reference_params):
    return build_model(CFG, seed=0, reference_params=reference_params)


@pytest.fixture(scope="module")
def tokens():
    return synthetic_batch(np.random.default_rng(7), 2, CFG.max_seq_len, CFG.vocab_size)


# ---------------------------------------------------------------------------
# architecture invariants
# ---------------------------------------------------------------------------


def test_architecture_is_the_same_two_pass_four_layer_model():
    """The migration must not alter LoCED's schedule or sparse attention."""
    assert CFG.num_layers == 4
    assert CFG.encode_layers == ("full",) * 4
    assert CFG.decode_layers == (
        "full",
        "full",
        "full",
        "full",
        "reuse",
        "reuse",
        "reindex",
        "reindex",
    )
    assert CFG.num_visits == 8          # 4 physical layers x 2 passes
    assert CFG.prefill_visits == 4      # prefill = one encode pass
    assert CFG.sparse_layers == ("reuse", "reuse", "reindex", "reindex")
    assert [CFG.layer_of_visit(v) for v in range(CFG.num_visits)] == [0, 1, 2, 3] * 2

    # compressed / sparse attention knobs are unchanged
    assert CFG.pool_m == 2
    assert CFG.window == 16
    assert CFG.top_k == 16
    assert CFG.indexer_heads == 2
    assert CFG.indexer_dim == 32
    assert CFG.indexer_block == 4
    assert CFG.hier_blocks == 4


def test_nnx_parameter_inventory_matches_reference(reference_params, nnx_model):
    assert nnx_model.count_params() == count_params(reference_params)
    nnx_params = nnx_model.to_reference_params()
    assert jax.tree.structure(nnx_params) == jax.tree.structure(reference_params)
    ref_leaves = jax.tree.leaves(reference_params)
    nnx_leaves = jax.tree.leaves(nnx_params)
    assert len(ref_leaves) == len(nnx_leaves) == 62
    for a, b in zip(ref_leaves, nnx_leaves):
        assert np.array_equal(np.asarray(a), np.asarray(b))


# ---------------------------------------------------------------------------
# exact parity with the reference implementation
# ---------------------------------------------------------------------------


def test_forward_and_loss_are_bit_identical(reference_params, nnx_model, tokens):
    ref = forward_train(CFG, reference_params, tokens)
    got = nnx_model.train_output(tokens)
    _assert_maxdiff("loss", ref["loss"], got["loss"])
    _assert_maxdiff("logits", ref["logits"], got["logits"])
    for key in ("ce", "ce_aux", "kl", "moe_aux"):
        _assert_maxdiff(key, ref[key], got[key])


def test_gradients_are_bit_identical(reference_params, nnx_model, tokens):
    ref_grad = jax.grad(lambda p: forward_train(CFG, p, tokens)["loss"])(reference_params)
    nnx_grad = nnx.grad(lambda m: m.train_loss(tokens)[0])(nnx_model)
    mapped = _reference_gradient_dict(nnx.to_pure_dict(nnx_grad), CFG)
    assert jax.tree.structure(mapped) == jax.tree.structure(ref_grad)
    for a, b in zip(jax.tree.leaves(ref_grad), jax.tree.leaves(mapped)):
        _assert_maxdiff("gradient", a, b)


def test_prefill_and_decode_are_bit_identical(reference_params, nnx_model, tokens):
    ref_store, ref_logits = prefill(CFG, reference_params, tokens)
    got_store, got_logits = nnx_model.prefill_tokens(tokens)
    _assert_maxdiff("prefill logits", ref_logits, got_logits)
    for a, b in zip(jax.tree.leaves(ref_store), jax.tree.leaves(got_store)):
        _assert_maxdiff("prefill store", a, b)

    for i in range(3):
        token = jnp.asarray([3, 7], jnp.int32)
        pos = CFG.max_seq_len + i
        ref_store, ref_step_logits = decode_step(CFG, reference_params, ref_store, token, pos)
        got_store, got_step_logits = nnx_model.decode_token(got_store, token, pos)
        _assert_maxdiff(f"decode step {i} logits", ref_step_logits, got_step_logits)


def test_real_maxtext_utilities_are_available():
    """GPU/TPU runs set LOCED_REQUIRE_REAL_MAXTEXT_UTILS=1 to forbid stubs."""
    if os.environ.get("LOCED_REQUIRE_REAL_MAXTEXT_UTILS") == "1":
        assert REAL_MAXTEXT_UTILS, (
            "MaxText utilities fell back to CPU stubs. Install the full MaxText "
            "dependencies (maxtext[cuda12] or maxtext[tpu])."
        )


def _reference_gradient_dict(pure: dict, cfg: LoCEDConfig) -> dict:
    """Map NNX pure-dict gradients back to the reference pytree keys."""
    cells = tuple(
        {
            "norm_a": pure["cells"][str(i)]["norm_a"],
            "norm_m": pure["cells"][str(i)]["norm_m"],
            "W_q": pure["cells"][str(i)]["attn"]["W_q"],
            "W_fx": pure["cells"][str(i)]["attn"]["W_fx"],
            "W_fv": pure["cells"][str(i)]["attn"]["W_fv"],
            "W_o": pure["cells"][str(i)]["attn"]["W_o"],
            "W_gate": pure["cells"][str(i)]["moe"]["routed_moe"]["gate"],
            "Ws1": pure["cells"][str(i)]["moe"]["shared_experts"]["w1"],
            "Ws3": pure["cells"][str(i)]["moe"]["shared_experts"]["w3"],
            "Ws2": pure["cells"][str(i)]["moe"]["shared_experts"]["w2"],
            "We1": pure["cells"][str(i)]["moe"]["routed_moe"]["wi_0"],
            "We3": pure["cells"][str(i)]["moe"]["routed_moe"]["wi_1"],
            "We2": pure["cells"][str(i)]["moe"]["routed_moe"]["wo"],
        }
        for i in range(cfg.num_layers)
    )
    return {
        "embed": pure["token_embedding"],
        "head": pure["logits_dense"],
        "W_bk": pure["W_bk"],
        "W_bv": pure["W_bv"],
        "W_ik": pure["W_ik"],
        "W_iq": pure["W_iq"],
        "cond": pure["visit_embedding"],
        "gate_a": pure["gate_a"],
        "gate_m": pure["gate_m"],
        "norm_f": pure["norm_f"],
        "cells": cells,
    }


# ---------------------------------------------------------------------------
# Muon classification: MaxText's own transform_logic sees LoCED correctly
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_MAXTEXT, reason="MaxText is not importable")
def test_maxtext_muon_treats_experts_as_batch_axis(nnx_model):
    pure = nnx.to_pure_dict(nnx.state(nnx_model))
    found = set()
    for path, leaf in _walk_params(pure):
        dims = transform_logic(path, shape=tuple(leaf.shape))
        name = path[-1]
        if name in ("token_embedding", "logits_dense", "visit_embedding"):
            assert dims is None, f"{path} must stay on AdamW, got {dims}"
        if name in ("gate_a", "gate_m") or leaf.ndim == 1:
            assert dims is None, f"{path} is not a matrix, got {dims}"
        if "routed_moe" in path and name in ("wi_0", "wi_1", "wo"):
            found.add(name)
            # [num_experts, in, out] -> orthogonalize each expert, not the
            # expert axis (which would mix experts together).
            assert dims.reduction_axis == (-2,)
            assert dims.output_axis == (-1,)
        if "routed_moe" in path and name == "gate":
            found.add("gate")
            assert dims.reduction_axis == (0,)
            assert dims.output_axis == (-1,)
        if "shared_experts" in path:
            found.add("shared")
            assert dims.reduction_axis == (0,)
            assert dims.output_axis == (-1,)
        if name in ("W_q", "W_fx", "W_fv", "W_o", "W_bk", "W_bv", "W_ik", "W_iq"):
            found.add(name)
            assert dims.reduction_axis == (0,)
            assert dims.output_axis == (-1,)
    assert {"wi_0", "wi_1", "wo", "gate", "shared"} <= found


def _walk_params(tree, path=()):
    if isinstance(tree, dict):
        for key, value in tree.items():
            yield from _walk_params(value, path + (str(key),))
    else:
        yield path, tree


# ---------------------------------------------------------------------------
# config mapping
# ---------------------------------------------------------------------------


def _maxtext_like_config(**overrides):
    values = dict(
        vocab_size=64,
        base_emb_dim=64,
        max_target_length=64,
        base_num_query_heads=4,
        base_num_kv_heads=1,
        head_dim=16,
        base_num_decoder_layers=4,
        num_experts=4,
        num_experts_per_tok=2,
        shared_experts=1,
        base_moe_mlp_dim=128,
        base_mlp_dim=128,
        weight_dtype="float32",
        dtype="float32",
        loced_pool_m=2,
        loced_window=16,
        loced_top_k=16,
        loced_indexer_heads=2,
        loced_indexer_dim=32,
        loced_indexer_block=4,
        loced_hier_blocks=4,
        loced_indexer_mlp=0,
        loced_intermediate_size=-1,
        loced_max_decode_len=32,
        loced_encode_layers=["full", "full", "full", "full"],
        loced_decode_layers=["full", "full", "full", "full", "reuse", "reuse", "reindex", "reindex"],
        loced_aux_deep_weight=0.10,
        loced_indexer_kl_coef=0.50,
        loced_moe_aux_weight=0.01,
        loced_rms_norm_eps=1e-6,
        loced_init_std=0.02,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_maxtext_config_maps_to_reference_config():
    cfg = loced_config_from_maxtext(_maxtext_like_config())
    assert cfg.vocab_size == 64
    assert cfg.hidden_size == 64
    assert cfg.max_seq_len == 64
    assert cfg.num_attention_heads == 4
    assert cfg.head_dim == cfg.kv_dim == 16
    assert cfg.pool_m == 2
    assert cfg.window == 16 and cfg.top_k == 16
    assert cfg.num_layers == 4
    assert cfg.num_visits == 8
    assert cfg.intermediate_size == 128
    assert cfg.top_k_experts == 2
    assert cfg.encode_layers == ("full",) * 4
    assert cfg.decode_layers[4:] == ("reuse", "reuse", "reindex", "reindex")

    # explicit LoCED width wins over base_moe_mlp_dim
    cfg2 = loced_config_from_maxtext(_maxtext_like_config(loced_intermediate_size=96))
    assert cfg2.intermediate_size == 96


def test_config_mapping_rejects_incompatible_shapes():
    with pytest.raises(ValueError, match="one shared KV head"):
        loced_config_from_maxtext(_maxtext_like_config(base_num_kv_heads=2))
    with pytest.raises(ValueError, match="physical depth"):
        loced_config_from_maxtext(_maxtext_like_config(base_num_decoder_layers=8))
    with pytest.raises(ValueError, match="shared SwiGLU expert"):
        loced_config_from_maxtext(_maxtext_like_config(shared_experts=2))
    with pytest.raises(ValueError, match="divisible by pool_m"):
        loced_config_from_maxtext(_maxtext_like_config(max_target_length=63))


def test_mapping_fallback_uses_the_same_keys():
    cfg = loced_config_from_mapping(
        {
            "vocab_size": 64,
            "base_emb_dim": 64,
            "max_target_length": 64,
            "base_num_query_heads": 4,
            "head_dim": 16,
            "base_moe_mlp_dim": 128,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "loced_pool_m": 2,
            "loced_window": 16,
            "loced_top_k": 16,
            "loced_indexer_heads": 2,
            "loced_indexer_dim": 32,
            "loced_indexer_block": 4,
            "loced_hier_blocks": 4,
            "loced_encode_layers": ["full"] * 4,
            "loced_decode_layers": ["full"] * 4 + ["reuse", "reuse", "reindex", "reindex"],
        }
    )
    assert cfg.intermediate_size == 128
    assert cfg.num_visits == 8


@pytest.mark.skipif(not _HAS_MAXTEXT, reason="MaxText is not importable")
def test_shipped_yaml_initializes_with_custom_config_class():
    config = pyconfig.initialize(
        ["test", "loced/maxtext/configs/loced_tiny.yml"],
        config_class=__import__(
            "loced.maxtext.config", fromlist=["LoCEDMaxTextConfig"]
        ).LoCEDMaxTextConfig,
    )
    cfg = loced_config_from_maxtext(config)
    assert config.model_name == "loced-tiny"
    assert cfg.num_layers == 4
    assert cfg.max_seq_len == config.max_target_length == 64
    assert cfg.intermediate_size == config.base_moe_mlp_dim == 128
    assert config.opt_type == "muon"
    assert config.muon_type == "maxtext_muon"


# ---------------------------------------------------------------------------
# MaxText Muon: real optimizer, one update
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_MAXTEXT, reason="MaxText is not importable")
@pytest.mark.parametrize(
    ("config_name", "expected_seq_len"),
    [
        ("loced/maxtext/configs/loced_500m_t4x2.yml", 128),
        ("loced/maxtext/configs/loced_500m_tpu8.yml", 128),
        ("loced/maxtext/configs/loced_500m_tpu.yml", 64),
    ],
)
def test_shipped_500m_yaml_keeps_run_sequence_length(
    config_name: str, expected_seq_len: int
):
    """The 500M model YAML must not silently override the run sequence length.

    MaxText merges the model YAML after the run YAML.  A ``max_target_length``
    in the model file therefore wins over the run setting, and an explicit CLI
    value triggers MaxText's model-vs-CLI conflict error.  Sequence length is a
    run setting; this test covers the T4, TPU v5e-8, and Colab smoke configs.
    """
    config = pyconfig.initialize(
        ["test", config_name, "steps=1"],
        config_class=__import__(
            "loced.maxtext.config", fromlist=["LoCEDMaxTextConfig"]
        ).LoCEDMaxTextConfig,
    )
    assert config.max_target_length == expected_seq_len
    assert config.base_emb_dim == 1024
    assert config.base_num_decoder_layers == 4
    assert config.num_experts == 8
    if expected_seq_len == 128:
        # Kaggle is single-host; the T4 and TPU v5e-8 runs must not try to
        # initialize JAX's multi-host distributed runtime during config setup.
        assert config.skip_jax_distributed_system


@pytest.mark.skipif(not _HAS_MAXTEXT, reason="MaxText is not importable")
def test_maxtext_muon_one_training_step():
    config = pyconfig.initialize(
        [
            "test",
            "loced/maxtext/configs/loced_tiny.yml",
            "steps=1",
            "enable_checkpointing=false",
            "ici_fsdp_parallelism=-1",
            # MaxText's cosine schedule starts at LR 0 during warmup.  This
            # test is about the Muon update, so start at the peak LR.
            "warmup_steps_fraction=0.0",
            "learning_rate_final_fraction=1.0",
        ],
        config_class=__import__(
            "loced.maxtext.config", fromlist=["LoCEDMaxTextConfig"]
        ).LoCEDMaxTextConfig,
    )
    model, mesh = maxtext_train.create_model_and_mesh(config)
    _, optimizer = maxtext_train.create_optimizer(config, model, mesh)
    tokens = synthetic_batch(np.random.default_rng(0), 2, CFG.max_seq_len, CFG.vocab_size)

    before = jax.tree.leaves(nnx.to_pure_dict(nnx.state(model)))
    loss, metrics = maxtext_train.train_step(model, optimizer, tokens)
    after = jax.tree.leaves(nnx.to_pure_dict(nnx.state(model)))

    assert np.isfinite(float(loss))
    assert set(metrics) == {"loss", "ce", "ce_aux", "kl", "moe_aux"}
    assert int(optimizer.step.get_value()) == 1
    # MaxText Muon must have changed the parameters (not a no-op).
    assert any(not np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(before, after))


@pytest.mark.skipif(not _HAS_MAXTEXT, reason="MaxText is not importable")
def test_empty_checkpoint_directory_is_not_restored(tmp_path):
    """A pre-created but empty checkpoint dir must start a fresh run."""
    empty = tmp_path / "empty-checkpoints"
    empty.mkdir()
    assert maxtext_train.load_checkpoint(str(empty), None, None, None) is None
    assert maxtext_train.load_checkpoint(str(tmp_path / "missing"), None, None, None) is None


@pytest.mark.skipif(not _HAS_MAXTEXT, reason="MaxText is not importable")
def test_orbax_checkpoint_roundtrip(tmp_path):
    config = pyconfig.initialize(
        [
            "test",
            "loced/maxtext/configs/loced_tiny.yml",
            "steps=1",
            "enable_checkpointing=true",
            "ici_fsdp_parallelism=-1",
            f"base_output_directory={tmp_path}",
            "run_name=loced-ckpt-test",
            "checkpoint_period=1",
            "skip_jax_distributed_system=true",
        ],
        config_class=__import__(
            "loced.maxtext.config", fromlist=["LoCEDMaxTextConfig"]
        ).LoCEDMaxTextConfig,
    )
    model, mesh = maxtext_train.create_model_and_mesh(config)
    _, optimizer = maxtext_train.create_optimizer(config, model, mesh)
    path = maxtext_train.checkpoint_path(config)

    tokens = synthetic_batch(np.random.default_rng(0), 2, CFG.max_seq_len, CFG.vocab_size)
    maxtext_train.train_step(model, optimizer, tokens)
    assert int(optimizer.step.get_value()) == 1
    maxtext_train.save_checkpoint(path, model, optimizer)

    model2, mesh2 = maxtext_train.create_model_and_mesh(config)
    _, optimizer2 = maxtext_train.create_optimizer(config, model2, mesh2)
    assert maxtext_train.load_checkpoint(path, model2, optimizer2, mesh2) == 1
    assert int(optimizer2.step.get_value()) == 1

    before = jax.tree.leaves(nnx.to_pure_dict(nnx.state(model)))
    after = jax.tree.leaves(nnx.to_pure_dict(nnx.state(model2)))
    assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(before, after))


@pytest.mark.skipif(not _HAS_MAXTEXT, reason="MaxText is not importable")
def test_trainer_resumes_from_checkpoint(tmp_path):
    common = [
        "test",
        "loced/maxtext/configs/loced_tiny.yml",
        "enable_checkpointing=true",
        "ici_fsdp_parallelism=-1",
        f"base_output_directory={tmp_path}",
        "run_name=loced-resume-test",
        "checkpoint_period=1",
        "per_device_batch_size=1",
        "skip_jax_distributed_system=true",
    ]
    config_class = __import__(
        "loced.maxtext.config", fromlist=["LoCEDMaxTextConfig"]
    ).LoCEDMaxTextConfig
    config = pyconfig.initialize([*common, "steps=2"], config_class=config_class)
    _, optimizer, _ = maxtext_train.train(config, log_every=0)
    assert int(optimizer.step.get_value()) == 2

    resumed = pyconfig.initialize([*common, "steps=3"], config_class=config_class)
    _, optimizer2, _ = maxtext_train.train(resumed, log_every=0)
    assert int(optimizer2.step.get_value()) == 3


@pytest.mark.skipif(not _HAS_MAXTEXT, reason="MaxText is not importable")
def test_trainer_rejects_non_maxtext_muon():
    def init_config(*overrides):
        return pyconfig.initialize(
            [
                "test",
                "loced/maxtext/configs/loced_tiny.yml",
                "ici_fsdp_parallelism=-1",
                *overrides,
            ],
            config_class=__import__(
                "loced.maxtext.config", fromlist=["LoCEDMaxTextConfig"]
            ).LoCEDMaxTextConfig,
        )

    bad_opt = init_config("opt_type=adamw")
    model, mesh = maxtext_train.create_model_and_mesh(bad_opt)
    with pytest.raises(ValueError, match="opt_type='muon'"):
        maxtext_train.create_optimizer(bad_opt, model, mesh)

    bad_muon = init_config("muon_type=optax_muon")
    model, mesh = maxtext_train.create_model_and_mesh(bad_muon)
    with pytest.raises(ValueError, match="maxtext_muon"):
        maxtext_train.create_optimizer(bad_muon, model, mesh)
