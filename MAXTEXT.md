# LoCED on MaxText v0.2.5

This document describes the MaxText port in `loced/maxtext/`. The goal is
narrow: **keep LoCED's architecture exactly as it is, and use MaxText for the
framework parts MaxText already provides.**

The reference implementation in `loced/model.py`, `attention.py`, `cache.py`,
`fnn.py`, and `block.py` remains the single source of truth. The MaxText
package is a thin framework shell around it.

---

## 1. What is reused from MaxText / Optax / Orbax

| Concern | Component used | Where |
|---|---|---|
| Config schema + YAML loading | `maxtext.configs.types.MaxTextConfig`, `pyconfig.initialize` | `loced/maxtext/config.py` |
| Device mesh + logical-axis rules | `maxtext.utils.maxtext_utils.get_mesh_from_config`, `config.logical_axis_rules` | `loced/maxtext/train.py` |
| Parameter placement | `maxtext.utils.sharding.nnx_construct_named_sharding` | `loced/maxtext/train.py` |
| LR schedule | `maxtext.utils.maxtext_utils.create_learning_rate_schedule` | `loced/maxtext/train.py` |
| Optimizer | `maxtext.optimizers.optimizers.get_optimizer` -> **MaxText Muon** | `loced/maxtext/train.py` |
| Checkpoints | Orbax `StandardCheckpointer` (Orbax is also MaxText's checkpoint backend) | `loced/maxtext/train.py` |
| NNX parameter graph | Flax NNX `Param` / `Intermediate` | `loced/maxtext/modeling.py` |

`loced/muon.py` is **not imported** by the MaxText package and is not used by
the trainer. The MaxText trainer requires:

```yaml
opt_type: muon
muon_type: maxtext_muon
```

`create_optimizer()` rejects `opt_type="adamw"` and rejects
`muon_type="optax_muon"` so a LoCED run cannot silently switch to a different
optimizer implementation. MaxText's `maxtext_muon` is the production path: it
is sharding-aware, understands the `out_sharding` metadata on every parameter,
and performs the sharded Newton-Schulz iteration.

Why MaxText Muon instead of Optax Muon? For a MaxText target, MaxText Muon is
the maintained and sharding-aware implementation. Optax Muon is portable and
fine for a single-device reference, but it does not understand MaxText's
physical shardings. The CPU integration tests exercise MaxText's Muon code
path directly; the reference CPU package never needs an optimizer at all.

---

## 2. What stays LoCED-owned

Only the architecture and its algorithm remain custom JAX:

* `loced/model.py` -- train / prefill / decode entry points;
* `loced/attention.py` -- sparse attention, oracle/indexer, KL distillation;
* `loced/cache.py` -- shared pool + delta + ring store (not MaxText `KVCache`);
* `loced/fnn.py` -- shared + routed SwiGLU MoE;
* `loced/block.py` -- visit conditioning, norms, gates;
* `loced/maxtext/modeling.py` -- the NNX shell that reassembles those
  parameters and calls the reference functions.

This split is intentional: MaxText has no decoder block with LoCED's
compressed pool / REUSE / REINDEX semantics, so rewriting the algorithm in
MaxText's `Decoder` would change the model. Instead, the NNX module gives
MaxText real parameter leaves, sharding metadata, intermediates and optimizer
hooks while calling the reference implementation bit-for-bit.

### Architecture invariants (unchanged)

```
encode = (full, full, full, full)                  # one pass, 4 physical layers
decode = (full, full, full, full,                  # pass A = the encoder
          reuse, reuse, reindex, reindex)          # pass B = same 4 layers
```

* 4 physical layers, each with its own attention + MoE;
* exactly two passes per generated token = 8 visits;
* weight tying is across roles, not depth: `visit v -> layer v % 4`;
* attention is token-level sparse top-k + forced local window, with
  `pool_m:1` KV bridging and an indexer;
* decoder cache modes are FULL / REUSE / REINDEX;
* only the last prompt position runs pass B to emit the first generated token.

The migration tests assert these invariants and compare the MaxText model
against the reference for weights, loss, logits, gradients, prefill and
decode.

---

## 3. Parameter names and MaxText Muon

MaxText's Muon classifier (`maxtext.utils.muon_utils.transform_logic`) decides
per leaf whether to orthogonalize and along which axes. The NNX shell names
parameters so the classifier produces the intended LoCED semantics:

| NNX parameter | Shape | MaxText treatment |
|---|---|---|
| `token_embedding` | `[V, d]` | AdamW (`embedding` exclusion) |
| `logits_dense` | `[d, V]` | AdamW (`logits_dense` exclusion) |
| `visit_embedding` | `[visits, d]` | AdamW (`embedding` exclusion) |
| `W_bk`, `W_bv`, `W_ik`, `W_iq` | 2-D | Muon, standard 2-D `(0,) -> (-1,)` |
| `cells.<i>.attn.W_q/W_fx/W_fv/W_o` | 2-D | Muon, standard 2-D |
| `cells.<i>.moe.routed_moe.wi_0/wi_1/wo` | `[E, d, I]` / `[E, I, d]` | Muon per expert: reduce `(-2,)`, output `(-1,)` |
| `cells.<i>.moe.routed_moe.gate` | `[d, E]` | Muon, standard 2-D |
| `cells.<i>.moe.shared_experts.w1/w3/w2` | 2-D | Muon, standard 2-D |
| `norm_*`, `gate_a`, `gate_m`, scalars | 1-D | AdamW |

The expert axis is a **batch axis** for Muon: `wi_0/wi_1/wo` are
orthogonalized independently per expert. This is exactly MaxText's MoE
convention and is covered by an integration test.

One NNX detail mattered here: `cells` is an `nnx.Dict` keyed by `"0".."3"`,
not an `nnx.List`. MaxText's `transform_logic` expects every path segment to
be a string; `nnx.List` lowers to integer dict keys and crashes the classifier.
The dict is only a container change; layer order and the reference pytree are
unchanged.

---

## 4. Configuration

The custom class is `loced.maxtext.config.LoCEDMaxTextConfig`, a subclass of
`MaxTextConfig`. It adds only the fields MaxText has no equivalent for
(`loced_pool_m`, `loced_window`, `loced_top_k`, indexer dimensions, the visit
schedule, auxiliary-loss weights). It also widens `model_name` from MaxText's
built-in model enum to a string so `models/loced-tiny.yml` can be resolved by
the normal MaxText YAML lookup.

Shipped configs:

```
loced/maxtext/configs/loced_tiny.yml          # run/optimizer/mesh settings
loced/maxtext/configs/models/loced-tiny.yml   # architecture
```

Initialize exactly like MaxText:

```python
from maxtext.configs import pyconfig
from loced.maxtext.config import LoCEDMaxTextConfig

config = pyconfig.initialize(
    ["train", "loced/maxtext/configs/loced_tiny.yml", "steps=100"],
    config_class=LoCEDMaxTextConfig,
)
```

Do not patch MaxText's `types.py`; pass `config_class=` instead.

---

## 5. Training

```bash
python -m loced.maxtext.train \
    loced/maxtext/configs/loced_tiny.yml \
    steps=20 run_name=loced-tiny
```

The entry point uses MaxText's mesh, LR schedule and Muon optimizer, then runs
a jitted `nnx.value_and_grad` step on `LoCEDModel.train_loss`. Checkpoints use
Orbax and are written under `checkpoint_dir` (MaxText derives it from
`base_output_directory` / `run_name` when checkpointing is enabled). The
optimizer's internal `step` is the resume cursor: a run with the same
`base_output_directory` / `run_name` restores the model and optimizer state
and continues from the next step.

For a real dataset, pass any iterable of `[batch, seq]` int32 token arrays:

```python
from loced.maxtext.train import train

train(config, batches=my_maxtext_or_grain_iterator)
```

`main` falls back to a small learnable synthetic task so the CPU smoke run
needs no tokenizer or data pipeline. The model, loss and optimizer are the
same in both cases.

---

## 6. Validation

The MaxText integration tests live in `loced/maxtext/test_integration.py`:

```bash
# CPU reference suite (pure JAX; no Flax required)
python3.11 -m pytest loced/smoke_test.py -q

# MaxText/Flax environment
python -m pytest loced/maxtext/test_integration.py -q
```

Current results on the tiny config (549,456 parameters):

| Check | Result |
|---|---|
| parameter inventory / names / shapes | exact match, 62 leaves |
| weights | max diff `0.0` |
| training loss + logits | max diff `0.0` |
| gradients | 62 leaves, max diff `0.0` |
| prefill logits + store | max diff `0.0` |
| decode logits (3 steps) | max diff `0.0` |
| MaxText Muon classification | embeddings/head/1-D -> AdamW; experts -> `(-2,), (-1,)` |
| MaxText Muon one update | finite loss, optimizer step 1, parameters changed |
| Orbax save/restore | model + optimizer state + `optimizer.step` exact |
| trainer resume | run 2 steps, restore, continue to step 3 |
| real MaxText utilities | enforced with `LOCED_REQUIRE_REAL_MAXTEXT_UTILS=1` |
| reference smoke suite | 19 passed |

The MaxText integration suite currently has 15 tests. The YAML test initializes
the shipped `loced-tiny` config through MaxText's real `pyconfig` and maps it to
`LoCEDConfig`.

### Cloud validation notebooks

Three notebooks cover the accelerator backends:

* `LoCED_MaxText_GPU_Kaggle.ipynb` — Kaggle GPU (T4 x2 / L4),
  `maxtext[cuda12]`;
* `LoCED_MaxText_TPU_Kaggle.ipynb` — Kaggle TPU v5e-8, `maxtext[tpu]`;
* `LoCED_MaxText_Colab_TPU.ipynb` — Colab TPU (free tier: v5e-1),
  `maxtext[tpu]`.

Each creates a Python 3.12 venv, installs MaxText v0.2.5 from the
`maxtext-v0.2.5` tag, clones the LoCED repository if needed, and sets
`LOCED_REQUIRE_REAL_MAXTEXT_UTILS=1` so the run fails instead of silently
falling back to CPU stubs. They run the architecture check, the 15 integration
tests, a 20-step MaxText Muon benchmark, and an Orbax save/restore/resume
check. Cloud parity uses `LOCED_PARITY_TOL=1e-5`; the CPU default remains exact
`0.0`. Reports are written as `kaggle_gpu_report.json`,
`kaggle_tpu_report.json`, and `colab_tpu_report.json`, respectively.

All three validation notebooks use the **0.55M-parameter `loced-tiny`**
shape, not a 500M model. The 4-layer/8-visit contract and the
model/optimizer/checkpoint code paths are identical; only width, MLP size,
vocabulary, and expert dimensions are reduced. These results validate
correctness and plumbing, not 500M-scale memory use or throughput.

[`LoCED_MaxText_500M_Colab_TPU.ipynb`](https://colab.research.google.com/github/offx-zinth/loced/blob/master/LoCED_MaxText_500M_Colab_TPU.ipynb)
is a separate **~500.2M-parameter scale smoke** for one TPU v5e-1 chip (two
MaxText Muon steps, no checkpoint). It checks fit, parameter count, a real
update, and peak memory; it does not replace the 15-test exact parity suite.

### Validated on Kaggle T4 x2 (2026-10-23)

Kernel: [LoCED MaxText v0.2.5 GPU Validation](https://www.kaggle.com/code/senthilvasany/loced-maxtext-v0-2-5-gpu-validation)

| Item | Result |
|---|---|
| environment | Python 3.12.3, JAX 0.11.1, MaxText 0.2.5, driver 580.178.04 |
| backend / devices | `gpu`, 2 x Tesla T4 |
| architecture | 4 physical layers, 8 visits, 549,456 params |
| integration tests | **15/15 passed** in 134.27 s |
| real MaxText utilities | enforced with `LOCED_REQUIRE_REAL_MAXTEXT_UTILS=1`; pass |
| 20-step benchmark | 36.0 ms/step, 7,110 per-device tokens/s (~14,220 global), loss 4.6542 -> 4.2957 |
| Orbax checkpoint | 17 files / 4,423,572 bytes; restore step 2; resume step 3 |
| raw report | [`validation/kaggle_t4_gpu_2026-10-23.json`](validation/kaggle_t4_gpu_2026-10-23.json) |

Two environment-specific fixes are part of the notebook:

* the venv's `site-packages/nvidia/*/lib` directories are added to
  `LD_LIBRARY_PATH`; without this the JAX CUDA plugin cannot load cuSPARSE and
  silently falls back to CPU;
* MaxText's cosine schedule starts at LR 0 during warmup, so the one-step Muon
  test disables warmup to exercise a nonzero update.

Both TPU targets are now validated: the single-chip run below, and the
8-device Kaggle `TpuV5E8` run recorded after it.

### Validated on Colab TPU v5e-1 (2026-10-23)

Notebook: [LoCED MaxText v0.2.5 — Colab TPU validation](https://colab.research.google.com/github/offx-zinth/loced/blob/master/LoCED_MaxText_Colab_TPU.ipynb)

| Item | Result |
|---|---|
| environment | Python 3.12.3, JAX 0.11.1, MaxText 0.2.5 |
| backend / devices | `tpu`, 1 x TPU v5 lite (v5e-1), 16.9 GB memory limit |
| architecture | 4 physical layers, 8 visits, 549,456 params |
| integration tests | **15/15 passed** in 53.52 s |
| real MaxText utilities | enforced with `LOCED_REQUIRE_REAL_MAXTEXT_UTILS=1`; pass |
| 20-step benchmark | 7.1 ms/step, 36,212 tokens/s (single chip), loss 4.6541 -> 4.2957 |
| Orbax checkpoint | 14 files / 4,385,506 bytes; restore step 2; resume step 3 |
| raw report | [`validation/colab_tpu_v5e1_2026-10-23.json`](validation/colab_tpu_v5e1_2026-10-23.json) |
| full PDF log | [`validation/colab_tpu_v5e1_2026-10-23.pdf`](validation/colab_tpu_v5e1_2026-10-23.pdf) |

This run used one TPU chip. It validates the TPU backend, real MaxText
utilities, MaxText Muon updates, and the Orbax save/restore/resume path, but it
does not exercise the 8-device mesh. The only warning was JAX's
transparent-hugepages advisory for TPU v5e; it does not affect correctness.

### Validated on Kaggle TPU v5e-8 (2026-10-24)

Kernel: [LoCED MaxText v0.2.5 TPU Validation](https://www.kaggle.com/code/senthilvasany/loced-maxtext-v0-2-5-tpu-validation)

| Item | Result |
|---|---|
| environment | MaxText v0.2.5 TPU image, Python 3.12 venv (same installer as v5e-1) |
| backend / devices | `tpu`, 8 x TPU v5e |
| architecture | 4 physical layers, 8 visits, 549,456 params |
| integration tests | **15/15 passed** (`integration_tests: PASS`) |
| real MaxText utilities | enforced with `LOCED_REQUIRE_REAL_MAXTEXT_UTILS=1`; pass |
| mesh | 8-way data parallel (`data=8`, `fsdp=1`) |
| 20-step benchmark | 11.3 ms/step, 22,741 per-device tokens/s (~181,900 global), loss 4.6541 -> 4.2957 |
| Orbax checkpoint | 14 files / 4,669,466 bytes; restore step 2; resume step 3 |
| raw report | [`validation/kaggle_tpu_v5e8_2026-10-24.json`](validation/kaggle_tpu_v5e8_2026-10-24.json) |

The 8-device loss trajectory is identical to the single-chip run, and the
Orbax restore/resume succeeded with the sharded checkpoint. The notebook's
`tokens_per_sec` field uses the per-device batch only; the global synthetic
batch rate is approximately 8x that number. This closes the multi-device TPU
check.

---

## 7. 500M full-pass HF data setup

The 500M training path streams all six `nampdn-ai` datasets with their
dataset-specific text columns, rather than assuming one shared schema:

| Dataset | Train rows | Text fields used | Est. tokens |
|---|---:|---|---:|
| `tiny-textbooks` | 399,000 | `textbook` + `text` | 0.40B |
| `tiny-strange-textbooks` | 1,000,000 | `text` | 0.70B |
| `tiny-codes` | 1,632,309 | `prompt` + `response` | 0.98B |
| `tiny-math-textbooks` | 635,497 | `response` | 1.41B |
| `tiny-orca-textbooks` | 146,732 | `textbook` + `question` + `response` | 0.28B |
| `tiny-webtext` | 2,276,083 | `human` + `bot` + `part_of_speech` | 0.95B |
| **total** | **6,089,621** | | **~4.72B** |

The token estimates come from tokenizing the datasets' first-row samples with
the TinyLlama 32k tokenizer; they are planning numbers, not exact counts. At
`per_device_batch_size=2` and `max_target_length=128` on 2×T4, one pass is
about **18.5M optimizer steps**. The trainer's step cap is above that and the
stream exhaustion break ends the run, so all source tokens are consumed.

The mixer uses an equal shuffled document schedule, then a 1,024-window shuffle
buffer. Tokenization feeds a rolling token buffer across document boundaries
(with BOS/EOS separators); the only padding is the final short window, so no
source tokens are dropped. The T4 configuration is
`maxtext/configs/loced_500m_t4x2.yml`; the notebook is
[`LoCED_MaxText_500M_T4x2_HF_Kaggle.ipynb`](https://www.kaggle.com/code/senthilvasany/loced-maxtext-500m-t4x2-hf-training).
A full pass is substantially longer than one Kaggle GPU session; Orbax
checkpoints every 10 minutes let the run continue, but the 8-chip TPU path is
the practical way to finish the pass.

---

## 8. Environment notes and limitations

* MaxText v0.2.5 requires Python >= 3.12 and its pinned JAX/Flax stack.
  `maxtext_muon`, `pyconfig`, the mesh helpers and Orbax checkpoints should be
  run in the official MaxText TPU/GPU container.
* The CPU reference environment used for the parity tests does not have the
  full MaxText dependency set. The integration tests import MaxText's real
  `pyconfig`, `transform_logic`, `get_optimizer` and Muon implementation, with
  lightweight stand-ins only for the two MaxText utility modules that pull in
  the full data/checkpoint stack. Production uses the real modules.
* LoCED's cache is custom (pool + delta + ring + REUSE/REINDEX rows).
  MaxText's standard `KVCache` and paged-attention path cannot represent it,
  so inference stays behind `LoCEDModel.prefill_tokens` / `decode_token` /
  `generate_tokens`, which are thin calls into the reference implementation.
* MaxText's standard `loss_fn` automatically adds variables ending in
  `indexer_loss` and `moe_lb_loss`. LoCED also has a deep-supervision CE term,
  which MaxText's generic loss does not know about; `LoCEDModel.train_loss`
  includes it, and the custom trainer uses that loss directly.
* The next performance step is replacing the gather-based candidate attention
  with Pallas kernels over query chunks, as noted in `ARCHITECTURE.md` §7.
  That is a kernel change, not a model/optimizer change.
