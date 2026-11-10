# LoCED 100M body — 12 layers, dynamic mHC + MagicNorm, GPT-2 vocab

**Colab TPU v5e-1 runbook** for `LoCED_100M_12L_Dynamic_mHC_GPT2_Synth_Colab_TPU_v5e1.ipynb`.

---

## What it builds

| Piece | Params |
|---|---:|
| 12-layer transformer body | **99.75M** |
| GPT-2 embedding | 12.87M |
| GPT-2 output head | 12.87M |
| **Full model** | **125.48M** |
| **Active body per token** | **5.30M** (5.32% of body) |

Per your choice, the 100M target is the **transformer body**; the GPT-2
embedding and head sit on top. "Active body" counts attention, norms, mHC,
the router, the shared expert, and the top-1 routed expert for one token,
across all 12 physical layers. It excludes the embedding lookup and the
vocabulary head (including the head gives 18.17M active).

**Model**: `d=256`, 12 physical layers, 24 visits (`12 × full`, then
`6 × reuse + 6 × reindex`), 4 mHC streams with dynamic coefficients,
MagicNorm at every visit exit, 62 routed experts (top-1) + 1 shared expert,
`I=168`, GPT-2 vocabulary (50,257).

**Data**: synthetic. Next token is `(prev + 1) % 512` with probability 0.9
over the first 512 GPT-2 token IDs, else uniform noise. The model keeps the
full 50,257-way head.

**Schedule**: `reuse_p` stays 0 for the first 20% of steps (exact FULL
attention as the oracle), ramps to 1 over the next 30%, then stays at 1 so
the sparse indexer-selection path trains.

---

## How to run it

1. Open the notebook in Colab.
2. **Runtime → Change runtime type → TPU.** Free Colab exposes one v5e chip.
3. Run all cells in order. The first cell installs MaxText's TPU extra into
   an isolated Python 3.12 venv and takes roughly **5–15 minutes**.
4. Keep the tab open for the whole run — free Colab may disconnect idle
   sessions.

The notebook clones `https://github.com/offx-zinth/loced.git` at `master`
into `/content/loced` and puts its parent on `PYTHONPATH`.

---

## What it checks

The single training script does five things, in order:

1. **Parameter audit** — asserts the body is 95–105M, the active body is
   5–6M, and `H_res` is doubly stochastic after Sinkhorn.
2. **Sparse top-1 MoE dispatch** — installs a `ragged_dot`-based top-1
   dispatch so a token really reads only its selected expert, then checks it
   against the dense reference kernel for values, aux loss, and gradients.
   Falls back to the dense kernel if the backend lacks `ragged_dot`.
3. **Mesh gradient check** — with ≥2 devices, compares the sharded gradient
   against the single-device gradient of the same global batch and fails
   loudly if they diverge. On the single Colab chip this reports
   `single device`.
4. **Training** — Muon on matrix weights, AdamW on embeddings/head/router,
   12 steps on the v5e-1 chip.
5. **Evaluation + generation** — sweeps `reuse_p ∈ {0.0, 0.5, 1.0}` and
   reports next-token accuracy, then greedily generates 16 tokens.

It writes `loced_100m_body_12l_mhc_gpt2_synth_report.json` plus
`loced_summary/` (report, loss history, `summary.txt`) into `/content`, so
they sit next to the notebook and are easy to download.

---

## Two bugs this notebook is designed to catch

These were found while building it, and the checks are now built in:

1. **Gradient reduction off by the device count.** Inside `shard_map`, the
   gradient of a *replicated* parameter is already all-reduced across the
   data axis. Applying an extra `pmean` is a no-op, so the applied gradient
   ends up `N_devices`× too large. The fix is to divide the local loss by
   `N_DEVICES`. The mesh gradient check verifies this on the real device.

2. **MoE aux loss cannot be sharded.** The load-balance auxiliary loss is a
   *product of two batch-level means*, so per-shard values cannot be
   recombined into the global one. It must use all-reduced global statistics
   (what production MoE training does). Otherwise the sharded gradient
   differs from the single-device gradient by ~1e-3 relative.

A third issue was configuration rather than math: at `seq_len=64` the
candidate count equals the selection budget (`window + top_k`), so *every*
candidate is picked regardless of the indexer — the sparse path is silently
untested and `reuse_p` has no effect. The notebook now uses `seq_len=128`
(selection ratio 0.8) and asserts the selection budget is strictly below the
candidate count.

---

## Expected output

```
accelerator : tpu  backend: tpu  devices: 1
total parameters          :    125,481,328  ( 125.481 M)
  transformer body        :     99,749,744  (  99.750 M)
  GPT-2 embedding         :     12,865,792  (  12.866 M)
  GPT-2 output head       :     12,865,792  (  12.866 M)
active body per token     :      5,304,176  (   5.304 M)
body sparsity             : 5.32% active
selection budget : {'n_candidates': 80, 'selection_budget': 64, 'is_sparse': True, ...}
top-1 dense-vs-sparse parity: value=... aux=... grad=...
```

On a single chip the mesh gradient check reports
`{'checked': False, 'reason': 'single device'}`. That is expected — it needs
≥2 devices to compare a sharded gradient against a single-device one. Use
the Kaggle v5e-8 notebook for the 8-device mesh check.

---

## Tuning knobs

Set these before running (edit the cell or use `os.environ`):

| Variable | Default | Meaning |
|---|---|---|
| `LOCED_SEQ_LEN` | `128` | Sequence length. Must be large enough that `window + top_k < candidates`, or the run fails with an explicit error. |
| `LOCED_STEPS` | `12` (Colab) / `20` (multi-device) | Training steps. |
| `LOCED_BATCH` | `2 × devices` | Global batch size. |
| `LOCED_LR` | `1e-3` | Muon/AdamW learning rate. |
| `LOCED_RUN_GENERATION` | `1` | Set `0` to skip greedy generation (it is the slowest part). |
| `LOCED_TIMEOUT` | `7200` | Seconds allowed for the training subprocess. |

---

## Notes

- This is a **correctness/plumbing run on synthetic data**, not a quality or
  throughput claim.
- The sparse MoE dispatch and the global-aux loss are installed in the
  notebook process only; the repository is unchanged.
- The loss plot needs `matplotlib`; it is written as
  `loced_100m_body_loss.png`.
