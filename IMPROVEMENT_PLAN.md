# LoCED — Improvement Plan

Status snapshot: 2026-10-21. Scope: `loced/` (pure-JAX reference implementation).
Phase 0.5 (Muon) is **implemented and green**; everything else is planned.

---

## 0. Diagnosis (measured, not assumed)

All numbers from the tiny config (`hidden=64`, 4 layers, `pool_m=2`, `window=16`,
`top_k=16`, `L=32`, `P=32`, random weights) via `attention.py` / `cache.py`:

| # | Finding | Evidence | Consequence |
|---|---|---|---|
| D1 | **Mean-pooled entries shrink**: `‖k_pool‖/‖k_t‖ = 0.813` (theory `1/√m = 0.707`) | `cache.py:88-89` | Pooled rows get systematically smaller logits than exact rows → the softmax demotes the compressed store by construction |
| D2 | **Selection inherits the bias**: pool share of top-k = `0.559` vs unbiased `0.667`; identical for oracle and indexer | `attention.py:124`, `pool_ik = pool_k @ W_ik` | The indexer (derived from the pooled K) also under-selects compressed history |
| D3 | **`delta` segment never appears in training** | `assemble_train_view` builds pool + window only; `assemble_store_view` adds delta + ring | Training store layout ≠ decode store layout; the model never learns to read m=1 archived entries |
| D4 | **REUSE rows are oracle-selected in training, indexer-selected at inference** | `model.py:252` vs `model.py:215` | Exposure bias on 2 of 4 pass-B visits (`ARCHITECTURE.md:140-142` claims these match — they do not) |
| D5 | **Pass A attention is a no-op at decode** | `full_exact` with `T=1` attends only to itself (`model.py:340`) | 4 of 8 visits per token pay attention cost for zero context; also means the encoder has no local context |
| D6 | **Candidate views are materialized** `[B,T,C,w]` | `attention.py:238-248` | ~14 GB per tensor at B=1, T=4096, C=2176, w=128 → blocks any real context length |

Open: the **per-segment attention-mass** measurement (`pr` is `[B,T,H,L]`, mask
must broadcast as `[B,T,1,L]`) is not yet correct — Phase 0 item.

---

## Phase 0 — Instrumentation (blocks Phase 1 decisions)

- [ ] Fix segment attention-mass measurement (pool / window / delta).
- [ ] Metrics harness logged per step: segment mass, indexer **recall@L vs oracle**,
      selection counts per segment, oracle↔indexer agreement.
- [ ] **Train-vs-decode agreement**: teacher-forced decode through the real store
      on the same sequence; report logit KL vs the training pass.
- [ ] Baseline artifacts: loss curves for AdamW and Muon on the same seed/batches.

Acceptance: every Phase 1 claim is backed by a before/after number from this harness.

---

## Phase 0.5 — Optimizer: AdamW → Muon  ✅ DONE

New file `loced/muon.py`; wired through `loced/train.py` (`--optim muon|adamw`).

- `make_muon_mask(params)` → per-leaf tags `{muon, muon_expert, adamw}`
  - **Muon**: hidden 2-D matrices (`W_q`, `W_fx`, `W_fv`, `W_o`, `Ws1/2/3`,
    `W_bk`, `W_bv`, `W_ik`, `W_iq`).
  - **Muon per-expert** (vmapped over the expert axis): `We1`, `We3`, `We2`.
  - **AdamW**: `embed`, `head`, router `W_gate`, `cond`, and every 1-D param.
- Newton–Schulz quintic `(3.4445, -4.7750, 2.0315)`, Frobenius pre-normalization,
  transpose for tall matrices, `ns_steps=3` default.
- `consistent_rms=True` (default): scale `sqrt(max(rows, cols))` so a Muon step
  has AdamW-like per-element RMS (~1) — swapping optimizers does not silently
  change the effective LR. `consistent_rms=False` gives the Keller-Jordan rule.
- Nesterov momentum uses the **sum-form** buffer (`buf = β·buf + g`), hence **no**
  bias correction (a `1/(1-βᵗ)` factor amplifies step 1 by 20× and diverges —
  caught during bring-up).

Measured on the tiny synthetic task (100 steps, batch 4, seed 0; final loss):

| optimizer | lr | final | note |
|---|---|---|---|
| AdamW | 3e-3 | 1.34 | baseline |
| Muon | 1e-3 | 1.34 | matches baseline at **3× lower LR** |
| Muon | 3e-3 | 1.59 | over-large step |
| Muon | 6e-3 | 947 | diverges |

`DEFAULT_LR = {"muon": 1e-3, "adamw": 3e-3}` encodes this.

Tests: `19 passed` (`python3.11 -m pytest loced/smoke_test.py -q`), including
orthogonality of the NS output, mask partitioning, RMS matching, jitted train
step, and end-to-end loss decrease with Muon.

Caveat: the synthetic task is trivial and the model tiny — this is a
*wiring + stability* check, not evidence of Muon's scaling advantage. The real
A/B belongs in Phase 6 (TinyStories, as in the MoRE runs).

---

## Phase 1 — Fidelity fixes (cheap, high impact)

| Item | Change | Files | Acceptance |
|---|---|---|---|
| 1.1 Norm-preserving pooling | Rescale pooled K/V to constituent mean norm, or QK-norm; optionally normalize keys before pooling | `cache.py:88-89` | `‖k_pool‖/‖k_t‖ ≈ 1`; pool share of top-k → ~0.667 (D1/D2) |
| 1.2 Train-time delta simulation | Add a `delta` segment (m=1 entries from pass-A exit hiddens) to the training candidate view, with the ring-archive protocol | `attention.py:222`, `cache.py:118` | Training view contains pool+delta+ring exactly like `assemble_store_view` (D3) |
| 1.3 On-policy REUSE | Mix/λ-schedule oracle vs indexer rows for the two REUSE visits; switch fully after warmup | `model.py:252`, `model.py:215` | Train/decode REUSE selection agreement > 95% by end of training (D4) |
| 1.4 Pass-A local context | Give pass A SWA over the ring (HySparse2 self-decoder) or delete its attention for the ablation | `model.py:340` | Decode FLOPs −25% (delete) or perplexity non-regression (SWA) (D5) |

---

## Phase 2 — Complexity: O(T) training and prefill

Today `full_exact` is O(T²) in **both** training and prefill (`model.py:281`);
compression/sparsity only pays at decode. Three escalating steps:

1. **Oracle subsampling** (cheapest): exact attention only at S sampled query
   positions → `O(S·T)`; exact scores then feed only the oracle target, aux CE
   and distill. Schedule S: 1/64 → 1/8 → full.
2. **Causal sparse encode** ("train what you deploy"): encode stack reads
   window + pool causally, `O(T·(w+k))` per layer. Fixes prefill *and* training.
3. **Optional:** swap one FULL layer for a linear-attention recurrence; a
   vectorized **KDA** scan already exists in `jax_mokre/` to port.

Acceptance: same final CE at 2k context for ≤ 1/5 the train-step wall clock;
prefill FLOPs/token flat in T.

---

## Phase 3 — Gates and normalization

- [ ] **Segment-gated attention read**: separate pool / window / delta outputs,
      learn per-segment gates (lets the model distrust compressed rows).
      Today the gates are scalar per visit (`block.py:76,79`, init 0 ⇒ 0.5).
- [ ] Per-channel or per-head gates; z-loss on gate saturation.
- [ ] **Bridge norm**: normalize `H_out` before `_bridge` so store statistics are
      stationary (the store's scale currently drifts with pass-A depth).
- [ ] **Pass-boundary norm** (MagicNorm transplant).
- [ ] QK-norm (also removes the D1 logit bias).
- [ ] **MagicNorm — honest scope**: LoCED has no TBPTT and no recurrence, so
      HRM's gradient-instability argument does not transfer. The analogues are
      (a) each layer's weights applied twice per token, (b) residual accumulation
      across two passes. Only the norm placements above are recommended.
- [ ] **Warmup credit assignment** (HRM's K=2→5 analog): ramp active visits
      (`full×4` → add reuse/reindex), ramp KL coefficient, ramp on-policy λ.

Acceptance: no loss regression; variance of hidden-state norm across visits is
bounded (log it in Phase 0).

---

## Phase 4 — New mechanisms (research upside)

1. **Compression-fidelity loss**: penalize the attention-output gap between exact
   per-token KV and pooled KV over each group — trains the compressor directly.
2. **Learned compression weights** per pool group (V4.1-style) instead of a plain
   mean; escape hatch promoting uncertain groups to exact `delta` entries.
3. **Relevance-based ring eviction**: archive the highest-salience evicted row into
   the *exact* delta pool instead of FIFO.
4. **Two-level indexer**: cheap block scorer → token rescoring (cuts indexer cost
   where `hier_blocks` currently only restricts REINDEX).
5. **Attention-sink segment**: first few prompt positions kept exact and always
   boosted, so pooled summaries cannot dilute BOS.

---

## Phase 5 — Scale-up engineering (gates everything at 4k+)

- [ ] Chunk the query dimension in `assemble_train_view` / `oracle_scores`
      (D6: ~14 GB per candidate tensor at 4k; `oracle_scores` also builds `[B,T,T]`).
- [ ] Replace the 3-D `gather_by_idx` one-hot einsum (`O(B·T·L·C)`) with
      `take_along_axis` over chunked shared candidates.
- [ ] Fuse score → top-k → gather (Pallas), as `attention.py:32-35` already states.
- [ ] MXFP4 main KV + FP4 indexer K (deployment-time format swap).

---

## Phase 6 — Evaluation protocol

Perplexity vs context length · indexer recall@L vs oracle · segment mass ·
train/decode logit KL · compression fidelity · prefill/decode latency · KV bytes/token.
Fixed ablation matrix, same seed/batch stream, both optimizers.

---

## Sequencing

```
Phase 0 (harness)  ──► Phase 1 (fidelity)  ──► Phase 3 (gates/norms)
        │                                            │
        └──► Phase 0.5 (Muon) ✅ ──► Phase 6 (real A/B)
                                                      │
Phase 5 (chunking) ──► Phase 2 (O(T)) ──► Phase 4 (new mechanisms)
```

## Open decisions

1. Priority: quality/fidelity (Phase 1) vs throughput (Phase 2)?
2. Target scale (context length, model size) — decides whether Phase 5 gates Phase 2.
3. Is the learned indexer the *only* inference-time selector? If yes, 1.3 is mandatory.
