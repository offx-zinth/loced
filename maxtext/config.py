"""MaxText v0.2.5 configuration for LoCED.

LoCED's architecture is not one of MaxText's decoder blocks, so the
integration adds a small :class:`LoCEDMaxTextConfig` extension with the
``loced_*`` fields that MaxText does not model.  Everything MaxText *does*
model stays in MaxText:

* ``base_emb_dim`` / ``head_dim`` / ``base_num_decoder_layers`` / ``vocab_size``
  / ``max_target_length`` -- the standard model shape;
* ``num_experts`` / ``num_experts_per_tok`` / ``shared_experts`` /
  ``base_moe_mlp_dim`` -- the standard MoE shape;
* ``opt_type`` / ``muon_type`` / ``learning_rate`` / mesh axes / checkpoint
  settings -- the standard MaxText training and parallelism knobs.

``loced_config_from_maxtext`` translates that MaxText config into the
reference :class:`loced.config.LoCEDConfig`.  The translation is one-way: the
reference dataclass remains the algorithm contract and is never used to
override a MaxText framework setting.

The module imports and works without MaxText installed (a dataclass fallback
is used) so the CPU reference environment can still exercise the mapping.  In
a MaxText container ``LoCEDMaxTextConfig`` is passed to
``pyconfig.initialize(..., config_class=LoCEDMaxTextConfig)`` which keeps
``extra='forbid'`` validation intact instead of patching MaxText's own types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

try:  # MaxText is optional in the reference CPU environment
    from maxtext.configs.types import MaxTextConfig
except ImportError:  # pragma: no cover - exercised only without MaxText
    MaxTextConfig = None  # type: ignore[assignment,misc]

try:
    from pydantic import Field
except ImportError:  # pragma: no cover - pydantic ships with MaxText
    Field = None  # type: ignore[assignment]


# Defaults accepted by both the pydantic (MaxText) and dataclass (CPU) forms.
_LOCED_DEFAULTS: dict[str, Any] = {
    "loced_pool_m": 2,
    "loced_window": 32,
    "loced_top_k": 32,
    "loced_indexer_heads": 4,
    "loced_indexer_dim": 64,
    "loced_indexer_block": 8,
    "loced_hier_blocks": 8,
    "loced_indexer_mlp": 0,
    "loced_intermediate_size": -1,
    "loced_max_decode_len": 128,
    "loced_encode_layers": ["full", "full", "full", "full"],
    "loced_decode_layers": [
        "full",
        "full",
        "full",
        "full",
        "reuse",
        "reuse",
        "reindex",
        "reindex",
    ],
    "loced_aux_deep_weight": 0.10,
    "loced_indexer_kl_coef": 0.50,
    "loced_moe_aux_weight": 0.01,
    "loced_rms_norm_eps": 1e-6,
    "loced_init_std": 0.02,
}


if MaxTextConfig is not None and Field is not None:

    class LoCEDMaxTextConfig(MaxTextConfig):
        """MaxText config plus the knobs MaxText has no equivalent for.

        Only LoCED-specific attention / schedule / auxiliary-loss values live
        here.  Model width, depth, vocabulary, MoE size, optimizer, mesh and
        checkpointing are inherited from :class:`MaxTextConfig` and therefore
        validated by MaxText itself.
        """

        # MaxText restricts model_name to its built-in model zoo.  LoCED ships
        # its own configs/models/*.yml, so widen the type while keeping the
        # exact same YAML lookup (`model_name` -> configs/models/<name>.yml).
        model_name: str = Field("default", description="Model config name; resolved to configs/models/<name>.yml.")

        loced_pool_m: int = Field(2, ge=1, description="KV Bridging pool_m:1 compression ratio.")
        loced_window: int = Field(32, ge=0, description="Forced local attention window.")
        loced_top_k: int = Field(32, ge=0, description="Token-level sparse attention budget.")
        loced_indexer_heads: int = Field(4, ge=1, description="Number of indexer query heads.")
        loced_indexer_dim: int = Field(64, ge=1, description="Per-indexer-head dimension.")
        loced_indexer_block: int = Field(8, ge=1, description="Entries per hierarchical candidate block.")
        loced_hier_blocks: int = Field(8, ge=1, description="Candidate blocks kept by a REINDEX pass.")
        loced_indexer_mlp: int = Field(
            0, ge=0, description="Indexer scorer hidden width; 0 keeps the linear scorer."
        )
        loced_intermediate_size: int = Field(
            -1,
            description="Per-expert SwiGLU width; <= 0 derives it from MaxText's MoE/MLP width.",
        )
        loced_max_decode_len: int = Field(128, ge=1, description="Generated-token budget per session.")
        loced_encode_layers: list[str] = Field(
            default_factory=lambda: list(_LOCED_DEFAULTS["loced_encode_layers"]),
            description="Ordered FULL-mode visits used by prefill and pass A of decode.",
        )
        loced_decode_layers: list[str] = Field(
            default_factory=lambda: list(_LOCED_DEFAULTS["loced_decode_layers"]),
            description="Ordered visits used by pass A + pass B of decode.",
        )
        loced_aux_deep_weight: float = Field(
            0.10, ge=0.0, description="Weight of the deep-supervision cross entropy."
        )
        loced_indexer_kl_coef: float = Field(
            0.50, ge=0.0, description="Weight of the oracle-to-indexer KL distillation."
        )
        loced_moe_aux_weight: float = Field(
            0.01, ge=0.0, description="Weight of the Switch-style load-balance auxiliary loss."
        )
        loced_rms_norm_eps: float = Field(1e-6, gt=0.0, description="RMSNorm epsilon.")
        loced_init_std: float = Field(0.02, gt=0.0, description="Parameter initialization standard deviation.")

else:  # pragma: no cover - CPU-only fallback

    @dataclass
    class LoCEDMaxTextConfig:  # type: ignore[no-redef]
        """CPU fallback carrying the same ``loced_*`` fields as the MaxText form."""

        loced_pool_m: int = 2
        loced_window: int = 32
        loced_top_k: int = 32
        loced_indexer_heads: int = 4
        loced_indexer_dim: int = 64
        loced_indexer_block: int = 8
        loced_hier_blocks: int = 8
        loced_indexer_mlp: int = 0
        loced_intermediate_size: int = -1
        loced_max_decode_len: int = 128
        loced_encode_layers: list[str] = None  # type: ignore[assignment]
        loced_decode_layers: list[str] = None  # type: ignore[assignment]
        loced_aux_deep_weight: float = 0.10
        loced_indexer_kl_coef: float = 0.50
        loced_moe_aux_weight: float = 0.01
        loced_rms_norm_eps: float = 1e-6
        loced_init_std: float = 0.02

        def __post_init__(self):
            if self.loced_encode_layers is None:
                self.loced_encode_layers = list(_LOCED_DEFAULTS["loced_encode_layers"])
            if self.loced_decode_layers is None:
                self.loced_decode_layers = list(_LOCED_DEFAULTS["loced_decode_layers"])


def _dtype_name(value: Any) -> str:
    """Return the canonical string for a MaxText ``DType``/string/None."""
    if value is None:
        return "float32"
    if hasattr(value, "value"):
        value = value.value
    return str(value).lower()


def _loced_get(config: Any, name: str) -> Any:
    """Read a ``loced_*`` field, falling back to the documented default."""
    return getattr(config, name, _LOCED_DEFAULTS[name])


def _intermediate_size(config: Any) -> int:
    """Choose the per-expert width from LoCED or MaxText model fields."""
    configured = int(_loced_get(config, "loced_intermediate_size"))
    if configured > 0:
        return configured
    moe_dim = int(getattr(config, "base_moe_mlp_dim", -1) or -1)
    if moe_dim > 0:
        return moe_dim
    return int(getattr(config, "base_mlp_dim", 0))


def loced_config_from_maxtext(config: Any):
    """Translate a MaxText config object into the reference ``LoCEDConfig``.

    Raises:
      ValueError: if the MaxText shape is incompatible with LoCED's invariants
        (single KV head, depth matching the visit schedule, ``pool_m`` dividing
        the sequence length, ...).
    """
    from ..config import LoCEDConfig

    encode_layers = tuple(str(x) for x in _loced_get(config, "loced_encode_layers"))
    decode_layers = tuple(str(x) for x in _loced_get(config, "loced_decode_layers"))
    if not encode_layers:
        raise ValueError("loced_encode_layers must not be empty")
    if not decode_layers:
        raise ValueError("loced_decode_layers must not be empty")

    kv_heads = int(getattr(config, "base_num_kv_heads", 1))
    if kv_heads != 1:
        raise ValueError(
            "LoCED uses MQA (one shared KV head); set base_num_kv_heads=1, "
            f"got {kv_heads}."
        )

    num_layers = int(getattr(config, "base_num_decoder_layers", len(encode_layers)))
    if num_layers != len(encode_layers):
        raise ValueError(
            "LoCED's physical depth is the length of loced_encode_layers; set "
            f"base_num_decoder_layers={len(encode_layers)}, got {num_layers}."
        )

    shared_experts = int(getattr(config, "shared_experts", 1))
    if shared_experts != 1:
        raise ValueError(
            "LoCED has exactly one always-on shared SwiGLU expert; set "
            f"shared_experts=1, got {shared_experts}."
        )

    cfg = LoCEDConfig(
        vocab_size=int(config.vocab_size),
        hidden_size=int(config.base_emb_dim),
        max_seq_len=int(config.max_target_length),
        max_decode_len=int(_loced_get(config, "loced_max_decode_len")),
        num_attention_heads=int(config.base_num_query_heads),
        head_dim=int(config.head_dim),
        kv_dim=int(config.head_dim),
        pool_m=int(_loced_get(config, "loced_pool_m")),
        window=int(_loced_get(config, "loced_window")),
        top_k=int(_loced_get(config, "loced_top_k")),
        indexer_heads=int(_loced_get(config, "loced_indexer_heads")),
        indexer_dim=int(_loced_get(config, "loced_indexer_dim")),
        indexer_block=int(_loced_get(config, "loced_indexer_block")),
        hier_blocks=int(_loced_get(config, "loced_hier_blocks")),
        indexer_mlp=int(_loced_get(config, "loced_indexer_mlp")),
        param_dtype=_dtype_name(getattr(config, "weight_dtype", "float32")),
        compute_dtype=_dtype_name(getattr(config, "dtype", "float32")),
        encode_layers=encode_layers,
        decode_layers=decode_layers,
        num_experts=int(getattr(config, "num_experts", 1)),
        num_shared_experts=shared_experts,
        top_k_experts=int(getattr(config, "num_experts_per_tok", 1)),
        intermediate_size=_intermediate_size(config),
        aux_deep_weight=float(_loced_get(config, "loced_aux_deep_weight")),
        indexer_kl_coef=float(_loced_get(config, "loced_indexer_kl_coef")),
        moe_aux_weight=float(_loced_get(config, "loced_moe_aux_weight")),
        rms_norm_eps=float(_loced_get(config, "loced_rms_norm_eps")),
        init_std=float(_loced_get(config, "loced_init_std")),
    )
    validate_loced_config(cfg)
    return cfg


def validate_loced_config(cfg: Any) -> None:
    """Assert the invariants the reference implementation relies on.

    The MaxText port is deliberately locked to the published LoCED
    architecture: 4 physical layers and two decode passes over the same stack.
    """
    if cfg.kv_dim != cfg.head_dim:
        raise ValueError(f"LoCED is MQA: kv_dim ({cfg.kv_dim}) must equal head_dim ({cfg.head_dim}).")
    if cfg.pool_m < 1:
        raise ValueError("pool_m must be >= 1")
    if cfg.max_seq_len % cfg.pool_m != 0:
        raise ValueError(
            f"max_seq_len ({cfg.max_seq_len}) must be divisible by pool_m ({cfg.pool_m})."
        )
    expected_encode = ("full",) * 4
    expected_decode = expected_encode + ("reuse", "reuse", "reindex", "reindex")
    if cfg.encode_layers != expected_encode:
        raise ValueError(
            "LoCED's encode stack is locked to 4 FULL layers; got "
            f"{cfg.encode_layers!r}."
        )
    if cfg.decode_layers != expected_decode:
        raise ValueError(
            "LoCED's decode schedule is locked to 4 FULL layers followed by "
            f"reuse,reuse,reindex,reindex; got {cfg.decode_layers!r}."
        )
    if cfg.num_layers != 4:
        raise ValueError(f"LoCED has 4 physical layers; got {cfg.num_layers}.")
    if cfg.num_experts < 1:
        raise ValueError("num_experts must be >= 1")
    if not 1 <= cfg.top_k_experts <= cfg.num_experts:
        raise ValueError(
            f"top_k_experts ({cfg.top_k_experts}) must be in [1, num_experts ({cfg.num_experts})]."
        )
    if cfg.indexer_block < 1:
        raise ValueError("indexer_block must be >= 1")
    if cfg.indexer_heads < 1 or cfg.indexer_dim < 1:
        raise ValueError("indexer_heads and indexer_dim must be >= 1")


def loced_config_from_mapping(mapping: Mapping[str, Any]):
    """CPU/YAML fallback used when MaxText is not installed.

    The mapping uses MaxText field names so the same YAML works in both
    environments; missing keys fall back to ``LoCEDConfig`` defaults where a
    sensible default exists.
    """
    from ..config import LoCEDConfig

    base = LoCEDConfig()
    get = mapping.get
    intermediate = int(get("loced_intermediate_size", -1))
    if intermediate <= 0:
        moe_dim = int(get("base_moe_mlp_dim", -1) or -1)
        intermediate = moe_dim if moe_dim > 0 else int(get("base_mlp_dim", base.intermediate_size))
    cfg = LoCEDConfig(
        vocab_size=int(get("vocab_size", base.vocab_size)),
        hidden_size=int(get("base_emb_dim", base.hidden_size)),
        max_seq_len=int(get("max_target_length", base.max_seq_len)),
        max_decode_len=int(get("loced_max_decode_len", base.max_decode_len)),
        num_attention_heads=int(get("base_num_query_heads", base.num_attention_heads)),
        head_dim=int(get("head_dim", base.head_dim)),
        kv_dim=int(get("head_dim", base.kv_dim)),
        pool_m=int(get("loced_pool_m", base.pool_m)),
        window=int(get("loced_window", base.window)),
        top_k=int(get("loced_top_k", base.top_k)),
        indexer_heads=int(get("loced_indexer_heads", base.indexer_heads)),
        indexer_dim=int(get("loced_indexer_dim", base.indexer_dim)),
        indexer_block=int(get("loced_indexer_block", base.indexer_block)),
        hier_blocks=int(get("loced_hier_blocks", base.hier_blocks)),
        indexer_mlp=int(get("loced_indexer_mlp", base.indexer_mlp)),
        param_dtype=_dtype_name(get("weight_dtype", base.param_dtype)),
        compute_dtype=_dtype_name(get("dtype", base.compute_dtype)),
        encode_layers=tuple(get("loced_encode_layers", base.encode_layers)),
        decode_layers=tuple(get("loced_decode_layers", base.decode_layers)),
        num_experts=int(get("num_experts", base.num_experts)),
        num_shared_experts=int(get("shared_experts", base.num_shared_experts)),
        top_k_experts=int(get("num_experts_per_tok", base.top_k_experts)),
        intermediate_size=intermediate,
        aux_deep_weight=float(get("loced_aux_deep_weight", base.aux_deep_weight)),
        indexer_kl_coef=float(get("loced_indexer_kl_coef", base.indexer_kl_coef)),
        moe_aux_weight=float(get("loced_moe_aux_weight", base.moe_aux_weight)),
        rms_norm_eps=float(get("loced_rms_norm_eps", base.rms_norm_eps)),
        init_std=float(get("loced_init_std", base.init_std)),
    )
    validate_loced_config(cfg)
    return cfg
