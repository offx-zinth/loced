# LoCED — Looped Causal Encoder–Decoder

A causal encoder–decoder built from a stack of **4 physical layers** (each
attention + MoE) tied across roles, whose attention combines the good parts of
**HySparse2**
(arXiv:2609.26368) and **CSA2 / DeepSeek-V4.1-Flash** (arXiv:2609.19969),
with **YOCO/CED-style KV bridging** (arXiv:2405.05254 and V4.1's §2.2).

- Prefill / encode = **one pass** over the **4 physical layers**, all FULL
  (early exit, bridges the decode KV store).
- Decoding = **exactly two forward passes** over the *same* 4 layers:
  the FULL stack (= the ENC itself) + `[REUSE, REUSE, REINDEX, REINDEX]`
  = 4 layers x 2 passes = 8 visits. The decoder owns no weights of its own.
- Weight tying is **across roles, not depth**: layer `k` is the same
  parameters in encode, pass A and pass B (`visit v -> layer v % 4`).
- Beyond the 4 layers, only the model-level pieces exist: embedding, head,
  final norm, one KV bridge + one indexer (the store is shared), and the
  per-visit `cond[v]` / `gate_a[v]` / `gate_m[v]` identities.
- The decoder side has its **own cache**, managed through
  **FULL / REUSE / REINDEX** modes exactly in the CSA2 style.

Pure JAX (`jax.jit`-able, plain pytree params), with a MaxText v0.2.5 port
in `loced/maxtext/` and a pytest suite (`loced/smoke_test.py`).

---

## 1. Layer schedule

The stack is **4 physical layers** = 4 x {attention + MoE}; the schedule is
explicit and ordered — it is the single source of truth in `config.py`:

```
encode = (full, full, full, full)                    # layers 0..3, one pass
decode = (full, full, full, full,                    # pass A = the ENC itself
          reuse, reuse, reindex, reindex)            # pass B = same 4 layers
```

- **Prefill / encode (one pass, no loop)** — every prompt token runs layers
  0..3 in `full` mode with exact causal attention (the oracle indexer), then
  the stack *exits early*: `H_out` is bridged into the decode KV store. Only
  the last prompt position additionally runs pass B to emit the first
  generated token's logits.
- **Decode (exactly two forward passes = 4 x 2 = 8 visits)** — every
  generated token:
  1. **pass A** = layers 0..3 in `full` mode — the encoder reused as decoder
     (the decoder owns no weights of its own). It writes the token's own
     store entry (delta m=1 + ring shift) and the *anchor scan* scores the
     whole causally visible store with the indexer (FULL: no block
     restriction), takes top-k + forced window.
  2. **pass B** = the SAME layers 0..3 revisited with sparse modes:
     - visit 4 on layer 0 **REUSE** — inherits the anchor's rows + indices,
       own Q, no scoring (CSA2 Reuse);
     - visit 5 on layer 1 **REUSE** — same rows/indices, fresh Q again;
     - visit 6 on layer 2 **REINDEX** — fresh indexer Q scores a
       block-restricted candidate pool against the main KV (CSA2 Reindex:
       rescore → fresh top-k), attends;
     - visit 7 on layer 3 **REINDEX** — rescore again with its own Q (final
       exit).

Tying is across roles, not depth: visit `v` runs on layer `v % 4`, so the
layer weights are shared by encode, pass A and pass B. There is no
`slots_per_pass`/loop multiplier — depth edits the stacks.

## 2. Attention: combining HySparse2 and CSA2

One MQA KV head (width = `head_dim`), no positional encodings anywhere
(HySparse2 uses NoPE for full/sparse layers), one output gate per visit.

**From HySparse2**
- *Token-level* selection (not block-level): ablations show token-level top-k
  beats block-level (+6.6 RULER-v2, +8.1 MRCR, +5.6 GraphWalks).
- A **forced local window** (here `window = 128`) is always selected from the
  *same* shared KV — no separate SWA branch, no separate SWA cache.
- Budget rule: `L = window + top_k`; window rows get a `+1e6` selection boost,
  the remaining `top_k` come from the indexer ranking.
- The FULL layers double as the **oracle indexer during training**: their
  exact attention scores decide the selection consumed by REUSE layers.

**From CSA2 / V4.1**
- Operating modes FULL / REINDEX / REUSE as above; every visit computes its
  own Q, so reuse is only about *which rows* are read.
- **Indexer K is projected from the main KV entries** (`ik = entry @ W_ik`),
  never from a separate representation — one projection basis for pool, delta
  and ring entries.
- **Hierarchical candidate pool**: pool entries are grouped into blocks of
  `indexer_block` (8) entries with masked-mean block keys. A REINDEX pass
  first picks the best `hier_blocks` blocks (V4.1: 2048 blocks × 8 positions
  = 16384 candidates) and may only consider their entries; the anchor scans
  everything. The grid is `ceil(P / indexer_block)` blocks (the last one may
  be partial), so every pool entry belongs to exactly one block.
- **Indexer training is distillation-only**: KL(oracle ‖ indexer-softmax)
  where the oracle distribution is the exact attention mass aggregated
  block-max over pool groups plus the window probabilities, with positions
  covered by the forced window removed from the pool groups (non-overlapping
  union). Gradients are isolated to the indexer projections via
  `stop_gradient` on hidden states/entries — the CE loss never touches the
  indexer, matching V4.1 ("gradients flow only through the indexer").
  Distillation is applied at **all three** scoring hiddens (the pass-A exit
  hidden `h0` and the two REINDEX hiddens), so train matches every inference
  scoring event.

## 3. KV bridging (encoder → decoder cache)

After prefill, the encode-stack exit residual `H_out` is projected once into
the shared store (CED: `C_l = H_{L/2} W_l^{KV}`, here one shared bridge pair
`W_bk/W_bv` for all visits):

| store segment | written by | contents | compression |
|---|---|---|---|
| `pool` (prompt) | prefill bridge from `H_out` | mean of `pool_m = 2` consecutive `H_out` projections | m = 2 |
| `delta` pool (generated) | decode pass-A exit hidden (same `W_bk/W_bv` bridge) | m = 1 per token | m = 1 |
| `ring` (exact window) | ring protocol | last `window` exact entries (prompt tail + generated) | m = 1 |

Ring protocol on append: if the ring is full, its **oldest entry is archived
into the delta pool**, everything shifts, the new entry enters at the tail —
so every exact entry eventually reaches the delta pool exactly once, and the
most recent `window` tokens are always exactly available.

Causal validity: pool entry valid for a query at position `t` iff
`0 ≤ pool_last ≤ t-1`; delta entries iff `0 ≤ delta_last ≤ t`; ring rows iff
`0 ≤ ring_pos ≤ t` — and only valid ring rows carry the forced-window boost.

**Cache economy** (scale configuration, fp8-ish K+V, 512-d model):
V4.1's CSA2 keeps ~890 B/token of global KV (plus per-layer local state).
LoCED prompt tokens cost `2 × kv_dim ≈ 64 B × (1/m=2)` for the pool plus the
fixed `window`-ring, and generated tokens one m=1 entry (~64 B) each with the
ring covering the exact recent window — ≈ 80–100 B/token total, with *no*
per-layer global KV at all (all 8 visits share the same store).

## 4. Training recipe (implemented in `forward_train`)

Single teacher-forced sequence, exactly mirroring the two inference passes:

1. **pass A** — the 4 `full` layers with exact causal attention (= oracle).
2. **KV bridge from `H_out`** — pool groups (`pool_m`) + per-position window
   rows; the candidate view carries the *same shape of evidence* decode will
   have (store rows: pool / window / delta). Positions covered by the window
   are excluded from pool-group oracle mass (non-overlapping union).
3. **pass B** — `[REUSE, REUSE, REINDEX, REINDEX]` over that view. The two
   REUSE layers inherit the rows the oracle selected (teacher forcing); the
   two REINDEX layers use the learned indexer with the hierarchical block
   restriction, so the trained selection rule == the deployed selection rule.
4. **Indexer distillation at all three scoring hiddens** (the pass-A exit
   hidden and the two REINDEX hiddens), stop-gradient isolated.

Losses:

```
L = CE(final) + 0.10 · CE(pass-A exit)          # deep supervision
      + 0.50 · mean_{h_anchor, h_reindex¹, h_reindex²} KL(oracle ‖ indexer)
      + 0.01 · MoE load-balance
```

Indexer KL never receives CE gradients; the main-KV bridge path receives no
indexer gradients (asserted in `test_indexer_gradients`).

## 5. Configuration

`loced/config.py` (tiny test scale defaults vs the scale-up suggestions):

| knob | tiny (tests) | scale-up (V4.1/HySparse2-adjacent) |
|---|---|---|
| hidden / heads × head_dim | 64 / 4×16 | 2048–3072 / 32–64 × 64–128 |
| kv_dim (MQA) | 16 | = head_dim |
| pool_m | 2 | 2 (prompt), 1 (delta) |
| window | 16 | 128 |
| top_k | 16 | 512 |
| indexer | 2 heads × 32 | 32 heads × 128 |
| indexer_block / hier_blocks | 4 / 4 | 8 / 2048 |
| physical layers (encode / decode visits) | 4 (4 / 8 visits) | 4 (4 / 8) — depth = edit the stacks |
| MoE | 4 routed, top-2, 1 shared | 256 routed, top-8, 1 shared |

## 6. Files

```
loced/
  config.py        # LoCEDConfig
  cache.py         # store: pool / delta / ring, bridging, block arrays
  attention.py     # exact FULL attention, candidate views, indexer, hierarchies, KL
  fnn.py           # DeepSeek-style MoE (shared + routed SwiGLU)
  block.py         # the physical layer stack (visit = norm + attn + gate + MoE + gate)
  model.py         # init / forward_train / prefill / prefill_extend / decode_step
  train.py         # optax jitted train step + tiny-config smoke run
  smoke_test.py    # pytest suite (shapes, causality, jit, cache, distill, train)
```

Run tests: `python3.11 -m pytest loced/smoke_test.py -q`
Run MaxText integration tests: `python -m pytest loced/maxtext/test_integration.py -q`
Run smoke train: `python3.11 -m loced.train`

## 7. Known approximations & next steps

- **Delta-content drift**: generated tokens write m=1 entries bridged from
  their pass-A exit hidden (same projections as the prompt pool/ring); the
  training view represents prompt tokens from `H_out`. This is the analogue of
  YOCO's train/decode mismatch for generated KV and is bounded by the shared
  ring (the drift only affects entries older than the window). V4.1 handles
  the related SWA-replay issue with "SWA Bounded Replay" during post-training
  — the same fine-tuning recipe applies here.
- **Tail asymmetry**: the first generated token's logits come from pass B on
  the last prompt position (its pass A was the exact prefill itself), whereas
  later tokens exit pass B following a fresh pass-A anchor. One token per
  prompt is affected.
- **Multi-turn extend** approximates cross-turn exact attention: new turns
  attend the store (pool + recent exact rows) plus their own chunk, not the
  previous turn's full exact KV.
- **Production kernels**: candidate rows are materialized per query. The
  scale-up path is fused score→top-k→gather Pallas kernels over query chunks
  (like MaxText's attention kernels) plus MXFP4 for main KV and FP4 indexer
  keys. FP4/mHC/Engram from V4.1 are left as deployment-time extensions.
- **MaxText lift**: the parameter pytree and visit functions are pure
  functions of `(cfg, params, …)`, so the MaxText port is a thin NNX shell
  rather than an algorithm rewrite. `loced/maxtext/` now contains the NNX
  model (`modeling.py`), the `MaxTextConfig` extension and YAML configs
  (`config.py`, `configs/`), and a trainer that uses MaxText's mesh, LR
  schedule and **MaxText Muon** (`train.py`). The reference training,
  prefill and decode functions are still called directly, and parity tests
  compare weights, loss, logits, gradients and decode against the reference.
  See `MAXTEXT.md` for the parameter/Muon mapping and the exact scope split.
  The MaxText path is GPU-validated on Kaggle T4 x2: 15/15 integration tests,
  36.0 ms/step on the 549,456-parameter config, and Orbax save/restore/resume.
  It is also TPU-validated on Colab v5e-1: 15/15 tests, 7.1 ms/step, and Orbax
  save/restore/resume. The 8-device Kaggle TPU v5e-8 job is the remaining
  multi-device check; it was still queued for capacity when the single-chip
  run completed.
