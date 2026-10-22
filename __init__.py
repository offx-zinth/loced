"""LoCED: Looped Causal Encoder-Decoder (pure-JAX reference + MaxText port).

A stack of **4 physical layers** (each = attention + MoE) whose weights are
tied ACROSS ROLES, not depth - the decoder owns no weights:
  * encode (prompt / embedding): one pass over layers 0..3, all FULL ->
    early exit, bridge the decode KV store (pool + exact window ring)
  * decode (generated token):    exactly TWO passes over the SAME 4 layers
        pass A = layers 0..3 FULL   (the ENC itself; writes own KV + anchor)
        pass B = layers 0..3 mode   (REUSE, REUSE, REINDEX, REINDEX)
    = 4 layers x 2 passes = 8 visits per token (no loop)

Attention = HySparse2 (token-level top-k + forced local window, oracle FULL
layers) + CSA2 (FULL/REINDEX/REUSE shared store, one indexer projected from
the main KV, hierarchical candidate blocks). Decode-side global KV is
CED/YOCO-style bridged from the encoder exit residual; per-visit identity is
`cond[v]`, `gate_a[v]`, `gate_m[v]`.

Load with `python3.11 -m pytest loced/smoke_test.py` for the reference unit
tests; `python3.11 -m loced.train` runs the tiny end-to-end train smoke. The
MaxText v0.2.5 port lives in `loced/maxtext/` (see `MAXTEXT.md`).
"""

from .config import LoCEDConfig
from .model import (count_params, decode_step, forward_train, generate,
                    init_model, prefill, prefill_extend)

__all__ = [
    "LoCEDConfig", "init_model", "count_params", "forward_train",
    "prefill", "prefill_extend", "decode_step", "generate",
]
