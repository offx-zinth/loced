# LoCED — Improvement Plan

Status snapshot: 2026-11-01. Scope: `loced/` (pure-JAX reference implementation).
Phase 0.5 (Muon) is **implemented and green**; the MaxText 100M/500M full-pass
paths are implemented. The next target is **4k context**, with fidelity and
long-context engineering running in parallel.

## Execution plan (2026-10-31)

Decisions locked:

1. **Parallel workstreams on one branch** (`improvement-plan`):
   - **Track F (fidelity):** Phase 0 -> Phase 1 -> Phase 3.
   - **Track T (long context):** Phase 0 -> shared view API -> Phase 5 ->
     Phase 2.
2. **Target 4k context now.** The 100M/T=128 full-pass run stays frozen until
   the 4k gate passes.
3. **Primary 4k gate: TPU v5e-8.** T4 x2 is the secondary fallback.
4. **The learned indexer is the only inference selector**, so 1.3
   (on-policy REUSE) is mandatory.

Milestone order:

| ID | Deliverable | Gate |
|---|---|---|
| M0 | Diagnostics harness + AdamW/Muon baseline artifacts | Every later claim has before/after numbers |
| M0.5 | Canonical `pool \| delta \| ring` view + query-chunk iterator | Chunked path matches current output |
| M1 | Norm-preserving pool, delta train view, on-policy REUSE, pass-A spike | D1-D4 fixed/measured; no loss regression |
| M2 | Segment gates, bridge/QK/pass-boundary norms, warmup ramps | No regression; norm/gate variance bounded |
| M3 | Lazy segment scoring, selected-row gather, 4k TPU v5e-8 benchmark | 4k fits TPU v5e-8; chunked == unchunked |
| M4 | Causal sparse encode + sampled exact oracle | Same CE at 2k-4k for <= 1/5 wall clock |
| M5 | Compression loss/weights, relevance eviction, two-level indexer, sink | Ablation wins on M0 metrics |
| M6 | Continuous fixed eval matrix | Reproducible report per change |

Immediate work: **M0 + M0.5**, then 1.1/1.3 (Track F) and M3.1 (Track T) in
parallel. M0.5 must land before either track edits the view/scoring layout, so
the parallel tracks do not invent competing APIs.

## Architecture scope update (2026-11-01)

The complexity/compression track (Phase 2: oracle subsampling, causal sparse
encode, KDA) is **deferred**. Current work is limited to bug patches and
architecture improvements, specifically:

- **MagicNorm** — module-level RMSNorm at every visit exit. Implemented behind
  `magic_norm` (opt-in).
- **mHC** — multi-stream residual state with a Sinkhorn-projected Birkhoff
  mixing matrix `H_res`, plus `H_pre`/`H_post` projections. Implemented behind
  `mhc_streams > 1` (opt-in), with optional dynamic coefficient prediction.
- **Bug patches** — MaxText `__call__` now honors the `reuse_p` schedule;
  architecture fields are mapped through MaxText config.

Both architecture changes are opt-in so the frozen 100M/T=128 checkpoint path
and existing parameter layout remain unchanged. The reference suite is now
**31 passed**, including MagicNorm, static mHC, dynamic-coefficient mHC, and
Sinkhorn doubly-stochastic checks. MaxText/Flax remains unvalidated in this
environment.

## Progress on `improvement-plan`

Implemented and tested in the pure-JAX suite:

- [x] **M0** `loced/diagnostics.py`: segment mass, recall@L/Jaccard, oracle vs
  indexer segment shares, pool norm ratio, teacher-forced train/decode KL,
  JSON reports, AdamW/Muon baseline CLI.
- [x] **M0.5 (API half)** canonical `pool | delta | ring` view metadata,
  `iter_query_chunks`, `slice_view`, selected-row gather helper, and
  query-chunked pass B. Chunked and unchunked logits/loss match.
- [x] **1.1** norm-preserving pooling in the shared `cache.pool_kv` helper.
  Tiny-config `‖pool‖/mean‖constituent‖` moved from ~0.81 to ~1.00.
- [x] **1.2** training-time delta simulation. Training validity/source
  positions now match the real prefill+decode store at every generated query.
- [x] **1.3** on-policy REUSE (`reuse_p` schedule, deterministic per-query
  mix). Reference trainer and MaxText trainer both ramp oracle -> indexer.
- [x] **MaxText plumbing** for `pool_norm_preserve`, `train_delta`,
  `train_prefill_frac`, `reuse_onpolicy`, warmup/ramp config fields.
- [x] **M3.1 (core)** lazy segment scoring + selected-row gather. Training
  and multi-turn prefill keep pool/delta shared and only materialize the ring
  plus `[B,T,L,w]` selected rows; `lazy_view=True` matches the materialized
  view exactly, and query-chunked pass B matches to `~1e-7`.
- [ ] **4k TPU v5e-8 gate**: run
  `maxtext/configs/loced_100m_4k_tpu8.yml` (100M shape,
  `max_target_length=4096`, window/top-k 128, batch 1/chip). The exact pass-A
  attention is still O(T²), so this gate also measures whether Phase 2 is
  needed before promising 4k.
- [ ] Rerun MaxText parity suite in a MaxText environment (this workstation
  lacks Flax/MaxText).

D2 needs a corrected baseline: with a forced window of `W`, `L = W + top_k`,
and a full window available, the expected pool share is `top_k / L = 0.5`,
not `0.667`. The measured `0.559` is above that because early queries have
shorter valid windows. The diagnostics harness now reports the raw counts, so
D2 should be judged on indexer-vs-oracle agreement and recall, not against
`0.667`.

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

Status: **D1 fixed** (norm-preserving pool), **D3 fixed** (training delta
simulation), **D4 fixed** (on-policy REUSE schedule), **D6 mitigated** (lazy
segment scoring/gather; no `[B,T,C,w]` view). **D2** is measured but the
`0.667` target was not the right null; see Progress. **D5 remains open** and
the O(T²) exact pass-A attention is the remaining 4k cost.

The segment-mass measurement is now correct: `attend` returns `[B,T,H,L]`
and the diagnostics mask broadcasts as `[B,T,1,L]`.

---

## Phase 0 — Instrumentation (blocks Phase 1 decisions)

- [x] Fix segment attention-mass measurement (pool / window / delta).
- [x] Metrics harness: segment mass, indexer **recall@L vs oracle**, selection
      counts per segment, oracle↔indexer agreement (`loced/diagnostics.py`).
- [x] **Train-vs-decode agreement**: teacher-forced decode through the real store
      on the same sequence; report logit KL vs the training pass.
- [x] Baseline artifacts: loss curves for AdamW and Muon on the same
      seed/batches (`loced/validation/diagnostics_baseline_*.json`).

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
| 1.1 Norm-preserving pooling ✅ | Rescale pooled K/V to constituent mean norm, or QK-norm; optionally normalize keys before pooling | `cache.py:pool_kv` | `‖k_pool‖/mean‖k_t‖ ≈ 1.00` (was ~0.81). D2 needs the forced-window baseline, see Progress. |
| 1.2 Train-time delta simulation ✅ | Add a `delta` segment (m=1 entries from pass-A exit hiddens) to the training candidate view, with the ring-archive protocol | `attention.py:assemble_train_view`, `oracle_scores` | Training view contains pool+delta+ring and matches store validity/source positions (D3 test). |
| 1.3 On-policy REUSE ✅ | Mix/λ-schedule oracle vs indexer rows for the two REUSE visits; switch fully after warmup | `model.py`, `train.py`, `maxtext/train.py` | `reuse_p=1` REUSE indices equal indexer indices; schedule is wired in reference + MaxText loops (D4). |
| 1.4 Pass-A local context | Give pass A SWA over the ring (HySparse2 self-decoder) or delete its attention for the ablation | `model.py:340` | Decode FLOPs −25% (delete) or perplexity non-regression (SWA) (D5) |

---

## Phase 2 — Complexity: O(T) training and prefill  ⏸ DEFERRED

Per the 2026-11-01 scope update, this track is deferred. It remains below as a
reference, but no Phase 2 optimization work is active.

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
