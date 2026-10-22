# LoCED — Layer & Loop Diagrams

ASCII diagrams of the architecture, the layer schedule, the cache protocol, and
the training/inference flows. Companion to `ARCHITECTURE.md`.

---

## 0. Master flow (one session)

```
tokens ──► embed ──►╔════════════ PRE-FILL / ENCODE — 4 physical layers, 1 pass ═══════╗
                     ║  L0 FULL ─► L1 FULL ─► L2 FULL ─► L3 FULL      H_out = exit x   ║
                     ║  (exact causal attention at EVERY layer = the oracle)  (early    ║
                     ║        │                                                exit)   ║
                     ║   probs_mean (oracle)                                     │       ║
                     ║        └──────────► KV BRIDGE: pool + window ring ◄───────┘       ║
                     ╚═══════════════════════════════════════════════╪═══════════════════╝
                                                                     ▼
                last prompt position only ──► PASS B: v4/L0 REUSE ─► v5/L1 REUSE ─►
                                               v6/L2 REINDEX ─► v7/L3 REINDEX ─► head ─► logits₀
      ┌──────────────────────────────────────────────────────────────────────────────┘
      ▼
╔══════════ DECODE — 2 passes over the SAME 4 layers (8 visits, no loop) ═══════════════╗
║  PASS A — the ENC itself (v0..v3 on L0..L3, FULL)  │  PASS B — sparse (v4..v7 on L0..L3)║
║   • write own KV entry (delta + ring)              │   REUSE  : inherit anchor rows+idx ║
║   • anchor scan + fresh top-k                      │   REINDEX: block-rescore → top-k   ║
║     (indexer over pool ∪ delta ∪ ring)             │                  (own Q, no write) ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝
      │
      └──► head ──► logitsₜ₊₁  ──► (greedy/sampling) ──► next token ──► loop
```

---

## 1. A physical layer (one visit) — layer diagram

```
                       ┌────────────── PHYSICAL LAYER k  (k = v % 4; same params for encode / pass A / pass B) ─────────────┐
                       │                                                                                                       │
 x ──► ⊕ ──► RMSNorm ──► HYBRID ATTENTION ──► @W_o ──► × σ(gate_a[v]) ──► ⊕ ──► RMSNorm ──► MoE FFN ──► × σ(gate_m[v]) ──► ⊕ ──► x'
        ▲       ▲              ▲                                                          │                 ▲
        │       │              │                                                          ▲                 │
    cond[v]   norm_a      mode-specific candidates                                     norm_m          gate_m[v]
  (loop cond.)            (FULL / REUSE / REINDEX view)                                         (shared + routed SwiGLU)
```

Attention internals (a layer's projections are shared by all of its modes):

```
 h ──► W_q ──► q [B,T,H,d_h] ──────────────────────────────┐
                                                            ▼
 rows ─────────────────────────► sparse softmax(q · Kᵀ/√d) ──► @V ──► out
   ▲                                                       (MQA: ONE shared KV head,
   │                                                        kv_dim == head_dim)
   │
   ├── FULL   : rows = exact causal prefix (W_fx/W_fv), also emits probs_mean (oracle)
   ├── REUSE  : rows = whatever the anchor scan picked (no new scores)
   └── REINDEX: rows = pool entries restricted to top-hier_blocks blocks ∪ delta ∪ ring,
                rescored by the indexer

 indexer:  q_i = h @ W_iq  (heads×dim, mean-pooled)
           k_i = entry @ W_ik        (indexer K is projected FROM main KV entries)
           scores = mean_h(q_i · k_i)
```

---

## 2. The two passes — visit state machine

```
             ┌──────────────────── DECODE — PASS A (the ENC itself) ────────────────┐
             │                                                                      │
   embed ──► ● L0/v0 FULL ─► ● L1/v1 FULL ─► ● L2/v2 FULL ─► ● L3/v3 FULL ─► H_exit│
             └────────────────────────────────┬─────────────────────────────────────┘
                                              │ writes delta + ring shift; anchor scan
                                              │ (indexer top-k + forced window)
                                              ▼
             ┌──────────────────── DECODE — PASS B (sparse modes) ──────────────────┐
             │ ● L0/v4 REUSE ─► ● L1/v5 REUSE ─► ● L2/v6 REINDEX ─► ● L3/v7 REINDEX │
             └──────────────────────────────────────────────────────────────────────┘

   PRE-FILL (ONE pass = visits 0..3): every prompt token runs layers L0..L3 FULL,
                                      then EXIT with H_out (KV bridge)
   last prompt position:              PASS B (visits 4..7)  →  first generated logits
```

Per-visit identity is ONLY: `cond[v]`, `gate_a[v]`, `gate_m[v]`. The attention
and MoE weights belong to the physical layer: visit `v` runs on layer `v % 4`,
so layers are shared by encode, pass A and pass B (tied across roles).

Visit schedule table (writes = cache mutations, score = indexer events):

| visit | layer | mode | pass | Q | KV write | indexer scan | rows read |
|---|---|---|---|---|---|---|---|
| 0–3 | 0–3 | FULL | A / prefill | own | L3 only (decode) | anchor scan (full range) | pool ∪ delta ∪ ring |
| 4 | 0 | REUSE | B | own | no | none | inherited from pass A |
| 5 | 1 | REUSE | B | own | no | none | inherited from pass A |
| 6 | 2 | REINDEX | B | own | no | top-blocks rescore | pool(blocked) ∪ delta ∪ ring |
| 7 | 3 | REINDEX | B | own | no | top-blocks rescore | pool(blocked) ∪ delta ∪ ring |

---

## 3. Decode step — detailed sequence

```
 Gaussian/sampled token ──► embed [B,1,d]
        │
        ▼
┌─ PASS A — v0..v3 on layers 0..3 : FULL (the ENC itself) ──────────────────────┐
│ x = L0.FULL(x; cond[0]) ─► L1.FULL(x; cond[1]) ─► L2 ─► L3.FULL(x; cond[3])  │
│ (each layer: h_k = RMSNorm(x + cond[k]); exact attn over the single token)    │
│ k = x@W_bk ; v = x@W_bv ; ik = k@W_ik          ──► append_delta(store, pos)   │
│                                                      (ring shift + archive)   │
│ view = assemble_store_view(store, pos)   [pool | delta | ring]                │
│ lg   = indexer_logits(x, view.ik)        (full causal scan, no blocks)        │
│ idx  = top_L( lg + mask + 1e6·ring_boost )                                    │
│ Kg,Vg,vg = gather(view, idx)                                                  │
└───────────────────────────────────────────────────────────────────────────────┘
        ▼
┌─ PASS B — v4..v7 on layers 0..3 : sparse modes ───────────────────────────────┐
│ v4 REUSE  (layer 0): out = attend(q(RMSNorm(x+cond[4])), SAME rows)           │
│ v5 REUSE  (layer 1): out = attend(q(RMSNorm(x+cond[5])), SAME rows)           │
│ v6 REINDEX(layer 2): h6 = RMSNorm(x+cond[6])                                  │
│             blocks = top_BB( indexer_logits(stopgrad(h6), block_key) )        │
│             valid2 = view.valid ∧ entry_mask(blocks)   (pool part only)       │
│             lg2 = indexer_logits(stopgrad(h6), view.ik)                       │
│             idx2 = top_L(lg2, valid2, ring boost) ; out = attend(q(h6), ...)  │
│ v7 REINDEX(layer 3): same procedure with cond[7] and a fresh top-k (exit)     │
└───────────────────────────────────────────────────────────────────────────────┘
        ▼
   head(RMSNorm(x, norm_f)) ──► logits [B,1,V] ──► next token
```

---

## 4. KV store layout and ring protocol

```
 SHARED STORE (one per sequence, ALL visits read/write it)

 pool:   [ e0 e1 e2 e3 ... e{P-1} ]   P = max_seq_len // pool_m      m = 2
          ▲ mean of pool_m consecutive H_out @ W_bk               (prompt only)
          │  causal stamp: pool_last (last covered position)
          │
 delta:  [ d0 d1 d2 ... d{D-1} ]      D = max_decode_len            m = 1
          ▲ one entry per generated token (pass-A exit, bridged)
          │
 ring:   [ ......... window ......... ]                             m = 1
          ▲ last `window` EXACT entries (prompt tail + newest tokens)
          │  forced window: boosted +1e6 during selection
          │
 blocks: block_key/block_last = masked-mean indexer-K and max stamp per
         8-entry block of pool                                 (hierarchical pool)

 RING PROTOCOL (every decode append):

   before:  ring = [ r0  r1  r2 ... r_{W-1} ]        r0 valid?
   append:      │
                ├── r0 valid & delta not full:
                │      archive r0 ──► delta[delta_n++]   (exact entry kept forever)
                │
                └── shift left; new entry enters tail
   after:   ring = [ r1  r2 ... r_{W-1}  new ]
```

Timeline example (window = 4, prompt tail p-3..p):

```
 step:    append    archive→delta     ring contents (positions)
 ─────────────────────────────────────────────────────────────
 prefill  –         –                 [p-3 p-2 p-1 p]
 gen 1    p+1       p-3 → delta[0]    [p-2 p-1 p   p+1]
 gen 2    p+2       p-2 → delta[1]    [p-1 p   p+1 p+2]
 gen 3    p+3       p-1 → delta[2]    [p   p+1 p+2 p+3]
```

---

## 5. Candidate view — how rows are assembled and selected

```
 one query at position t ──►  C = [ POOL (P) │ DELTA (D) │ RING (W) ]
                                  │            │            │
                 valid:  0 ≤ pool_last ≤ t-1  0 ≤ delta_last ≤ t  0 ≤ ring_pos ≤ t
                 boost:        0                0                 +1e6  (forced window)
                                  │            │            │
                                  ▼            ▼            ▼
                         rows_k/rows_v/rows_ik  [B, Tq, C, w]
                                  │
             indexer (or oracle) scores ──► +mask −1e4 ──► +boost ──► top_L
                                                                       │  L = window + top_k
                                                                       ▼
                                            gather rows ──► sparse softmax attention
```

(Only *valid* ring rows carry the boost, so a multi-position view — e.g. a
whole extended turn — is causally correct before/after ring reseeding.)

Train view has the same contract, with segments = POOL + trailing WINDOW rows
(unfolded from the per-position bridge buffer) — the same evidence decode sees.

---

## 6. Indexer training — distillation loop with gradient isolation

```
      EXACT attention (last encode layer)                    INDEXER network
   ┌───────────────────────────────────┐               ┌──────────────────────────┐
   │ probs_mean [B,T,T]                │               │ q_i = stopgrad(h) @ W_iq │
   │ block-max over pool_m groups      │               │ k_i = stopgrad(entry)@W_ik│
   │ ✗ positions inside forced window  │               │ scores = mean_h(q_i·k_i) │
   │ + exact probs at window rows      │               └────────────┬─────────────┘
   └──────────────┬────────────────────┘                            │
                  ▼                                                 ▼
          target distribution ─────────────► KL( target ‖ softmax(scores) ) ◄── at the
                  (stop-gradient)                             anchor hidden + both
                                                              reindex hiddens
                                                     │
                                                     ▼ gradients flow ONLY into W_iq, W_ik
```

---

## 7. Hierarchy (REINDEX block restriction)

```
 pool entries grouped in blocks of indexer_block = 8:

   block:   b0        b1        b2        b3        b4     ...     b{NB-1}
   entries: e0..e7    e8..e15   e16..e23  e24..e31  ...
   key:     mean(ik)  mean(ik)  ...                       ← block_key  [B, NB, idim]
   stamp:   max(pool_last in block)                       ← block_last [B, NB]

 REINDEX:  q_i(h6) · block_key ──► top_BB (hier_blocks, e.g. 2048@scale) ──► entry mask
           └─ only masked pool entries compete for the final top_L selection

 grid:     NB = ceil(P / indexer_block) blocks (the LAST one may be partial) and
           block_id(j) = j // indexer_block, so every pool entry maps to exactly
           one block (no out-of-range id -> no silent jnp.take clipping).
```

---

## 8. Parameter sharing map (4 physical layers, tied across roles)

```
                ┌────────────────── PER-LAYER  (x4, shared across passes) ────────────────┐
                │  attention:  W_q  W_fx  W_fv  W_o                                       │
                │  MoE:        W_gate  Ws1  Ws3  Ws2  We1  We3  We2                       │
                │  norms:      norm_a  norm_m                                             │
                └─────────────────────────────────────────────────────────────────────────┘
                     used identically by:  encode layers 0..3 = pass A layers 0..3
                                           = pass B layers 0..3
                     (visit v runs on layer v % 4 -- tied ACROSS ROLES, not depth)

  model-level:  embed, head, norm_f, W_bk, W_bv, W_ik, W_iq
                (one shared store -> one bridge + one indexer)
  per-visit:    cond[v]  (loop embedding)      gate_a[v]  gate_m[v]  (sigmoid gates)
                v in [0, 8)
```

---

## 9. Training forward — one diagram

```
 tokens [B,T]
    │
 embed ──► PASS A: layers 0..3 (v0..v3) FULL (exact causal) ──► H_out
               │                                    │
        probs_mean (oracle)               KV BRIDGE (pool means +
               │                          per-position window rows)
               │                                    │
               │      train VIEW = pool ∪ trailing window rows (decode-shaped evidence)
               │                                    │
               │   PASS B: v4/L0 REUSE ─► v5/L1 REUSE ─► v6/L2 REINDEX ─► v7/L3 REINDEX
               │              (oracle-selected rows)          │  ↑ KL distill at the
               │                                            │  │  anchor h + both
               ▼                                            ▼  │  reindex hiddens
        KL loss (indexer)          CE(final) + 0.1·CE(H_out exit) + 0.01·MoE aux
```
