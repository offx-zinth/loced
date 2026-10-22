"""LoCED configuration (pure-JAX dataclass, MaxText hparam style)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LoCEDConfig:
    """LoCED (Looped Causal Encoder-Decoder) hyper-parameters.

    Architecture parameters follow the design in ARCHITECTURE.md:
      * 4 PHYSICAL layers (each with its own attention + MoE), driven by
        explicit layer stacks:
            encode = (full, full, full, full)                    # 1 pass
            decode = encode + (reuse, reuse, reindex, reindex)   # 2 passes
        The decoder owns no weights: pass A IS the encode stack; pass B
        revisits the SAME 4 layers with sparse modes. 4 layers x 2 passes
        = 8 layer visits per generated token (no loop of the decode block).
      * attention = token-level sparse top-k + forced local window (HySparse2)
                    + FULL / REUSE / REINDEX cache modes (CSA2)
      * decoder-side global KV is projected once from the encoder exit
        hidden state (KV Bridging, YOCO/CED) and compressed pool_m:1
    """

    # --- core ---
    vocab_size: int = 32000
    hidden_size: int = 256
    max_seq_len: int = 512            # prefill length budget (prompt)
    max_decode_len: int = 128         # generated-token budget per session

    # --- attention ---
    num_attention_heads: int = 8
    head_dim: int = 32               # per-Q-head dim (NoPE throughout)
    kv_dim: int = 32                 # MQA: 1 KV head; invariant kv_dim == head_dim
    pool_m: int = 2                  # KV Bridging compression ratio (prompt entries)
    window: int = 32                 # forced local window (recent tokens always selected)
    top_k: int = 32                  # token-level sparse budget on top of the window
    indexer_heads: int = 4
    indexer_dim: int = 64
    indexer_block: int = 8           # entries per hierarchical candidate block
    hier_blocks: int = 8             # blocks kept by a REINDEX pass (candidate pool)
    indexer_mlp: int = 0             # 0 = linear indexer scorer, >0 = 1-hidden-MLP width
    param_dtype: str = "float32"     # store/param dtype (FP4 is a deployment-time swap)
    compute_dtype: str = "float32"

    # --- schedule: explicit, ordered layer stacks (the single source of truth) ---
    # The model is a stack of 4 PHYSICAL layers. prefill (encode) runs the
    # stack ONCE (all FULL, no loop). decode runs exactly TWO forward passes
    # over the SAME 4 physical layers (no decoder-side layer weights):
    #   pass A = encode_layers       (the ENC itself, reused as decoder)
    #   pass B = decode_layers[4:]   (sparse: REUSE, REUSE, REINDEX, REINDEX)
    # = 4 layers x 2 passes = 8 layer visits per generated token.
    encode_layers: tuple = ("full", "full", "full", "full")
    decode_layers: tuple = ("full", "full", "full", "full",
                            "reuse", "reuse", "reindex", "reindex")

    # --- MoE FFN ---
    num_experts: int = 8             # routed experts
    num_shared_experts: int = 1
    top_k_experts: int = 2
    intermediate_size: int = 512     # per-expert SwiGLU hidden

    # --- training ---
    aux_deep_weight: float = 0.10    # CE on the pass-1 exit (deep supervision)
    indexer_kl_coef: float = 0.50    # KL distill: oracle (exact attn) -> indexer
    moe_aux_weight: float = 0.01     # Switch-style load-balance auxiliary loss
    rms_norm_eps: float = 1e-6
    init_std: float = 0.02

    @property
    def num_layers(self) -> int:
        """Physical layers in the stack (4). Encode depth = pass-A depth."""
        return len(self.encode_layers)

    @property
    def num_visits(self) -> int:
        """Cell visits in one full decode step (2 passes x 4 layers = 8)."""
        return len(self.decode_layers)

    @property
    def prefill_visits(self) -> int:
        """Cell visits per prompt token (encode pass only, early exit = 4)."""
        return len(self.encode_layers)

    @property
    def sparse_layers(self) -> tuple:
        """Pass B: the decode-only modes, one per physical layer."""
        return tuple(self.decode_layers[self.num_layers:])

    def layer_of_visit(self, visit: int) -> int:
        """Physical layer a visit runs on.

        Pass A visits 0..V-1 are layers 0..V-1; pass B visits V..2V-1 revisit
        layers 0..V-1 (weight-tied across passes).
        """
        return visit % self.num_layers
