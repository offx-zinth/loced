# LoCED — Looped Causal Encoder–Decoder

An experimental language-model architecture exploring shared layer reuse, sparse attention, and compressed KV memory.

LoCED explores a central research question:

> Can a small stack of physical layers be reused across encoder and decoder roles while a shared, compressed memory store reduces inference-time KV-cache costs?

This repository contains a pure-JAX reference implementation and a MaxText v0.2.5 integration. The project is under active development. Current tests validate implementation and framework integration—not competitive language-model quality or production-scale efficiency.

## Key Features

- **4 physical layers** — Each combines attention and a routed Mixture-of-Experts (MoE) feed-forward network.
- **Cross-role weight sharing** — The same four layers handle encoding and both decoding passes.
- **8 layer visits per generated token** — Four visits in pass A, followed by four in pass B.
- **Shared compressed KV memory** — Pooled prompt history, archived generated tokens, and an exact rolling window.
- **Sparse token selection** — A learned hierarchical indexer selects candidate history entries while preserving access to recent context.
- **Pure JAX implementation** — A reference model with training, prefill, and decoding paths.
- **MaxText integration** — NNX model wrapper, Muon optimizer, multi-device support, and Orbax checkpointing.

## Architecture

### Layer Schedule

LoCED has four physical layers. Repeated visits reuse these layers rather than introducing additional independent layers.

#### Encode / Prefill

    Input → L0 FULL → L1 FULL → L2 FULL → L3 FULL
                                             ↓
                                      Shared KV Store

#### Decode — Pass A

           L0 FULL → L1 FULL → L2 FULL → L3 FULL
                                             ↓
                                       Memory anchor

#### Decode — Pass B

           L0 REUSE → L1 REUSE → L2 REINDEX → L3 REINDEX
                                                  ↓
                                             Next-token logits

The decoding schedule is:

| Visit | Physical Layer | Attention Mode |
|---:|---|---|
| 1 | L0 | FULL |
| 2 | L1 | FULL |
| 3 | L2 | FULL |
| 4 | L3 | FULL |
| 5 | L0 | REUSE |
| 6 | L1 | REUSE |
| 7 | L2 | REINDEX |
| 8 | L3 | REINDEX |

Each visit computes its own query and uses visit-specific conditioning, while physical layer parameters are shared across roles.

### Attention Modes

- **FULL:** Computes exact causal attention and establishes the anchor memory selection.
- **REUSE:** Reuses selected memory positions while computing a fresh query.
- **REINDEX:** Rescores a restricted candidate pool and makes a fresh token selection.

The design combines token-level top-k selection, a forced local window, and hierarchical candidate selection.

### Shared KV Memory

Instead of maintaining a separate global KV cache for every layer, LoCED uses one shared store.

| Segment | Purpose | Representation |
|---|---|---|
| Prompt pool | Compressed prompt history | Mean-pooled groups; default `pool_m = 2` |
| Delta archive | Generated history leaving the exact window | One entry per generated token |
| Exact ring | Recent context | Rolling window of exact entries |

The local window preserves direct access to recent tokens. The indexer selects additional historical entries, while REINDEX visits narrow the candidate pool before token selection.

## Training

The reference implementation includes:

- Teacher-forced language-model training.
- Auxiliary cross-entropy supervision at the pass-A exit.
- KL distillation from exact-attention scores to the learned indexer.
- MoE load-balancing loss.
- AdamW and Muon optimizer support in the reference training path.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the detailed memory protocol, layer schedule, configuration, and objective.

## Repository Structure

    loced/
    ├── config.py                 # Model configuration
    ├── cache.py                  # Shared KV store
    ├── attention.py              # Attention and indexer
    ├── fnn.py                    # Routed/shared MoE
    ├── block.py                  # Layer blocks and conditioning
    ├── model.py                  # Training, prefill and decode
    ├── train.py                  # Reference training entry point
    ├── smoke_test.py             # Reference test suite
    └── maxtext/
        ├── modeling.py           # MaxText/NNX model wrapper
        ├── config.py             # MaxText configuration
        ├── train.py              # MaxText training entry point
        ├── configs/              # Tiny configuration files
        └── test_integration.py   # Integration tests

## Getting Started

### 1. Clone the Repository

    git clone https://github.com/offx-zinth/loced.git
    cd loced

### 2. Set Up the Environment

The reference implementation uses JAX, Optax, and pytest. Use a compatible Python environment and install the appropriate JAX build for your hardware.

See the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

### 3. Run the Reference Tests

    python3.11 -m pytest loced/smoke_test.py -q

### 4. Run the Smoke-Training Path

    python3.11 -m loced.train

This exercises the tiny reference training path. It is an implementation check, not a full language-model training recipe.

## MaxText Integration

LoCED also includes an integration with MaxText v0.2.5. It uses NNX, MaxText's Muon optimizer implementation, device-mesh utilities, and Orbax checkpointing while retaining the reference implementation as the source of truth for the architecture.

Run a short training job in an environment configured with the compatible MaxText dependencies:

    python -m loced.maxtext.train \
        loced/maxtext/configs/loced_tiny.yml \
        steps=20 run_name=loced-tiny

Run the integration tests:

    python -m pytest loced/maxtext/test_integration.py -q

See [MAXTEXT.md](MAXTEXT.md) for environment requirements, configuration, training, and checkpoint restoration.

## Validation

The recorded integration runs use a 549,456-parameter tiny configuration and cover CPU and accelerator environments, including Kaggle T4 GPUs, Colab TPU v5e-1, and Kaggle TPU v5e-8.

The tests cover implementation properties such as parameter parity, loss and logit comparisons, gradients, prefill/decode behavior, and checkpoint save/restore/resume.

These results do not establish competitive perplexity, long-context quality, or an end-to-end inference speedup over a conventional Transformer.

## Known Limitations

LoCED is an active research prototype. Important open issues include:

- Training-time REUSE selection does not yet fully match inference-time indexer selection.
- The training candidate view does not yet faithfully reproduce the generated-token delta archive used during decoding.
- Exact attention makes training and prefill quadratic in sequence length.
- Materializing candidate rows limits practical long-context scaling.
- Pass A currently provides no useful cross-token attention for a single-token decode query.

These are known gaps, not solved claims. See [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) for measured findings and proposed fixes.

## Research Goals

LoCED investigates the trade-offs between:

1. Reusing parameters across encoder and decoder roles.
2. Compressing historical context while preserving recent tokens exactly.
3. Retrieving relevant history through sparse attention.
4. Aligning learned memory selection between training and inference.
5. Measuring real efficiency through KV-cache bytes, latency, and quality at increasing context lengths.

The next major milestone is a controlled comparison against a conventional baseline using comparable data, tokenizer, parameter/FLOP budget, and evaluation methodology.

## Documentation

- [Architecture and configuration](ARCHITECTURE.md)
- [MaxText integration](MAXTEXT.md)
- [Improvement plan](IMPROVEMENT_PLAN.md)

## Project Status

**Research prototype — active development.**

LoCED is an architectural experiment. It is not yet presented as a production-ready or quality-validated language model.

## License

See the repository's license file for licensing terms.