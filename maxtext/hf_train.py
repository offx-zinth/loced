"""Full LoCED-500M training on streaming Hugging Face datasets.

This is a cloud training entry point for the 8-chip MaxText TPU path.  It:

1. Streams (no local download) the nampdn-ai textbook / code / webtext datasets,
   mixes them in a configurable ratio, and tokenizes on the fly.
2. Trains the ~500.2M LoCED model with MaxText Muon (opt_type="muon",
   muon_type="maxtext_muon") on an 8-way data-parallel TPU v5e-8 mesh.
3. Checkpoints model + optimizer with Orbax every 10 minutes of wall clock
   (once total runtime passes 29880 s / ~8.3 h), plus a final checkpoint.
4. Resumes from the latest Orbax checkpoint if one exists.

The HF token is read from the environment (``HF_TOKEN`` / ``HF_HUB_TOKEN``) --
it is never hard-coded here.  Pass it in the Kaggle/Colab notebook cell or as a
notebook secret; do not commit it.

Usage (Kaggle TPU v5e-8)::

    python -m loced.maxtext.hf_train \
        loced/maxtext/configs/loced_500m_tpu8.yml \
        steps=200000 run_name=loced-500m-tpu8 \
        hf_datasets="nampdn-ai/tiny-textbooks,nampdn-ai/tiny-strange-textbooks,nampdn-ai/tiny-codes,nampdn-ai/tiny-math-textbooks,nampdn-ai/tiny-orca-textbooks,nampdn-ai/tiny-webtext" \
        hf_ratios="1.0,1.0,1.0,1.0,1.0,1.0"

The run config sets ``steps``, ``per_device_batch_size`` and ``max_target_length``;
override them on the command line as needed.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Iterable, Iterator

import jax
import jax.numpy as jnp
import numpy as np

from maxtext.configs import pyconfig
from maxtext.utils import max_logging

from .config import LoCEDMaxTextConfig, loced_config_from_maxtext
from .modeling import LoCEDModel, build_model
from .train import (
    checkpoint_path,
    create_model_and_mesh,
    create_optimizer,
    load_checkpoint,
    save_checkpoint,
    train_step,
)

# ---------------------------------------------------------------------------
# Hugging Face streaming data
# ---------------------------------------------------------------------------

DEFAULT_DATASETS = (
    "nampdn-ai/tiny-textbooks",
    "nampdn-ai/tiny-strange-textbooks",
    "nampdn-ai/tiny-codes",
    "nampdn-ai/tiny-math-textbooks",
    "nampdn-ai/tiny-orca-textbooks",
    "nampdn-ai/tiny-webtext",
)

# Column names commonly used for raw text in these datasets.
_TEXT_COLUMNS = ("text", "content", "body", "document", "prompt", "completion")


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HF_HUB_TOKEN")


def _text_column(features) -> str | None:
    for name in _TEXT_COLUMNS:
        if name in features:
            return name
    # Fall back to the first string-ish column.
    for name, feat in features.items():
        if getattr(feat, "dtype", None) == "string" or "Value" in type(feat).__name__:
            return name
    return None


def _iter_dataset_text(
    dataset_name: str,
    split: str,
    token: str | None,
    seed: int,
    shuffle_buffer: int = 10_000,
) -> Iterator[str]:
    """Yield text examples from one streaming HF dataset."""
    from datasets import load_dataset  # imported lazily so CPU envs keep working

    ds = load_dataset(
        dataset_name,
        split=split,
        streaming=True,
        token=token,
        trust_remote_code=True,
    )
    column = _text_column(ds.features)
    if column is None:
        raise ValueError(
            f"No text column found in {dataset_name}; features={dict(ds.features)}"
        )
    ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    for example in ds:
        text = example.get(column)
        if isinstance(text, str) and text.strip():
            yield text


def _mix_iterators(
    iterators: list[Iterator[str]],
    ratios: list[float],
    rng: np.random.Generator,
) -> Iterator[str]:
    """Probability-mix several streaming text iterators by ratio."""
    active = list(range(len(iterators)))
    probs = [float(r) for r in ratios]
    while active:
        norm = sum(probs[j] for j in active)
        pick = [probs[j] / norm for j in active]
        sel_pos = int(rng.choice(len(active), p=pick))
        i = active[sel_pos]
        try:
            yield next(iterators[i])
        except StopIteration:
            active.pop(sel_pos)
            probs.pop(sel_pos)
            if not active:
                break


def hf_streaming_batches(
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    datasets: tuple[str, ...] = DEFAULT_DATASETS,
    ratios: tuple[float, ...] | None = None,
    seed: int = 0,
    split: str = "train",
    tokenizer_name: str | None = None,
    max_examples: int | None = None,
) -> Iterable[jax.Array]:
    """Yield int32 ``[batch, seq]`` token batches from streaming HF datasets.

    Tokenization: if ``tokenizer_name`` is set it is loaded with ``transformers``
    and must produce ids in ``[0, vocab_size)``.  Otherwise a deterministic
    character hash is used so the script runs without an external tokenizer; for
    real training you should train/use a 32k-vocab tokenizer that matches the
    model and pass it via ``tokenizer_name``.
    """
    rng = np.random.default_rng(seed)
    token = _hf_token()
    ratios = ratios or tuple(1.0 for _ in datasets)

    if tokenizer_name is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, token=token, trust_remote_code=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        def encode(text: str) -> list[int]:
            ids = tokenizer.encode(text, add_special_tokens=False)
            return [int(i) % vocab_size for i in ids]

    else:

        def encode(text: str) -> list[int]:
            # Deterministic per-character hash into [0, vocab_size).  Smoke-only
            # placeholder; replace with a real 32k-vocab tokenizer for quality.
            return [(hash(text) * 31 + i) % vocab_size for i in range(len(text))]

    iterators = [
        _iter_dataset_text(name, split, token, seed + i)
        for i, name in enumerate(datasets)
    ]
    mixed = _mix_iterators(iterators, list(ratios), rng)

    batch: list[int] = []
    count = 0
    for text in mixed:
        if max_examples is not None and count >= max_examples:
            break
        ids = encode(text)
        # Chunk long documents into seq_len windows (drop short tail).
        for start in range(0, len(ids) - seq_len + 1, seq_len):
            window = ids[start : start + seq_len]
            if len(window) < seq_len:
                window = window + [0] * (seq_len - len(window))
            batch.append(window)
            if len(batch) == batch_size:
                yield jnp.asarray(batch, dtype=jnp.int32)
                batch = []
        count += 1

    if batch:
        # Pad the final partial batch.
        while len(batch) < batch_size:
            batch.append([0] * seq_len)
        yield jnp.asarray(batch, dtype=jnp.int32)


# ---------------------------------------------------------------------------
# Timed training loop with 10-minute checkpointing
# ---------------------------------------------------------------------------

CKPT_INTERVAL_S = 600.0          # checkpoint every 10 minutes
LONG_RUN_THRESHOLD_S = 29_880.0  # begin time-based checkpointing after ~8.3 h


def train_timed(
    config,
    batches: Iterable[jax.Array],
    log_every: int = 1,
    ckpt_interval_s: float = CKPT_INTERVAL_S,
    long_run_threshold_s: float = LONG_RUN_THRESHOLD_S,
) -> tuple[LoCEDModel, object, dict]:
    """Train with wall-clock checkpointing.

    Checkpoints model + optimizer every ``ckpt_interval_s`` seconds once total
    elapsed time passes ``long_run_threshold_s`` (and also at the end).  This
    is independent of MaxText's step-based ``checkpoint_period``.
    """
    loced_cfg = loced_config_from_maxtext(config)
    model, mesh = create_model_and_mesh(config)
    schedule, optimizer = create_optimizer(config, model, mesh)

    steps = int(config.steps)
    ckpt_dir = checkpoint_path(config) if getattr(config, "enable_checkpointing", False) else None
    start_step = 0
    if ckpt_dir:
        restored_step = load_checkpoint(ckpt_dir, model, optimizer, mesh)
        if restored_step is not None:
            start_step = restored_step
            max_logging.log(f"resumed LoCED from {ckpt_dir} after {restored_step} steps")

    batch_iter = iter(batches)
    last_metrics: dict = {}
    t_start = time.time()
    last_ckpt = t_start

    for step in range(start_step, steps):
        try:
            tokens = next(batch_iter)
        except StopIteration:
            max_logging.log(f"dataset exhausted at step {step}")
            break

        loss, metrics = train_step(model, optimizer, tokens)
        last_metrics = {k: float(v) for k, v in metrics.items()}

        if log_every > 0 and (step % log_every == 0 or step == steps - 1):
            lr = float(schedule(step))
            elapsed = time.time() - t_start
            max_logging.log(
                f"step {step:6d} loss {float(loss):.6f} "
                f"ce {last_metrics.get('ce', 0):.6f} kl {last_metrics.get('kl', 0):.6f} "
                f"moe_aux {last_metrics.get('moe_aux', 0):.6f} lr {lr:.3e} "
                f"elapsed {elapsed/60:.1f} min"
            )

        now = time.time()
        if ckpt_dir and (now - t_start) >= long_run_threshold_s and (now - last_ckpt) >= ckpt_interval_s:
            save_checkpoint(ckpt_dir, model, optimizer)
            last_ckpt = now
            max_logging.log(f"time-based checkpoint at step {step} ({now - t_start:.0f}s)")

    # Final checkpoint.
    if ckpt_dir and steps > start_step:
        save_checkpoint(ckpt_dir, model, optimizer)
        max_logging.log(f"final checkpoint at step {step if steps > start_step else start_step}")

    return model, optimizer, last_metrics


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv if argv is None else argv)
    config = pyconfig.initialize(argv, config_class=LoCEDMaxTextConfig)

    # Map to the reference LoCED config so we know the real vocab / seq size.
    loced_cfg = loced_config_from_maxtext(config)

    hf_datasets = os.environ.get("HF_DATASETS") or getattr(config, "hf_datasets", "") or ",".join(DEFAULT_DATASETS)
    datasets = tuple(d.strip() for d in hf_datasets.split(",") if d.strip())
    ratios_env = os.environ.get("HF_RATIOS") or getattr(config, "hf_ratios", "") or ""
    ratios = tuple(float(r) for r in ratios_env.split(",") if r.strip()) or None
    tokenizer_name = os.environ.get("HF_TOKENIZER") or getattr(config, "hf_tokenizer", None) or None
    seed = int(getattr(config, "init_weights_seed", 0))

    batches = hf_streaming_batches(
        loced_cfg.vocab_size,
        int(config.max_target_length),
        int(config.per_device_batch_size),
        datasets=datasets,
        ratios=ratios,
        seed=seed,
        tokenizer_name=tokenizer_name,
    )

    _, _, metrics = train_timed(config, batches)
    max_logging.log(f"finished LoCED HF run: {metrics}")


if __name__ == "__main__":
    main()
