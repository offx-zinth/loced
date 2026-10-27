"""Full LoCED-500M training on streaming Hugging Face datasets.

This is the MaxText training entry point used by both the 8-chip TPU and the
2x T4 GPU notebooks.  It:

1. Streams (no local download) the six nampdn-ai textbook / code / webtext
   datasets, interleaves them with a shuffled ratio schedule, keeps every
   token in a rolling stream, and tokenizes on the fly with a 32k tokenizer.
2. Trains the ~500.2M LoCED model with MaxText Muon (opt_type="muon",
   muon_type="maxtext_muon").
3. Checkpoints model + optimizer with Orbax every 10 minutes of wall clock
   once total runtime passes ``LOCED_CKPT_AFTER_S`` (default 29880 s / ~8.3 h;
   pass 0 to start checkpointing immediately), plus a final checkpoint.
4. Resumes from the latest Orbax checkpoint if one exists.  A full pass is
   ~4.7B tokens; on 2x T4 it spans multiple Kaggle sessions.

The HF token is read from the environment (``HF_TOKEN`` / ``HF_HUB_TOKEN``) --
it is never hard-coded here.  Pass it in the Kaggle/Colab notebook cell or as a
notebook secret; do not commit it.

Usage (Kaggle TPU v5e-8)::

    python -m loced.maxtext.hf_train \
        loced/maxtext/configs/loced_500m_tpu8.yml \
        steps=25000000 run_name=loced-500m-tpu8 \
        hf_datasets="nampdn-ai/tiny-textbooks,nampdn-ai/tiny-strange-textbooks,nampdn-ai/tiny-codes,nampdn-ai/tiny-math-textbooks,nampdn-ai/tiny-orca-textbooks,nampdn-ai/tiny-webtext" \
        hf_ratios="1.0,1.0,1.0,1.0,1.0,1.0"

The run config sets ``steps``, ``per_device_batch_size`` and ``max_target_length``;
override them on the command line as needed.  Set ``steps`` above the full-pass
estimate so the loop stops when the streams are exhausted, not at an arbitrary
step count.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Iterable, Iterator

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from maxtext.configs import pyconfig
from maxtext.utils import max_logging

from .config import LoCEDMaxTextConfig, loced_config_from_maxtext
from .modeling import LoCEDModel
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

# TinyLlama's tokenizer has exactly 32,000 entries, matching the LoCED vocab.
DEFAULT_TOKENIZER = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Column names commonly used for raw text in these datasets.
_TEXT_COLUMNS = ("text", "content", "body", "document", "prompt", "completion")

# The nampdn-ai datasets do not share one schema: tiny-codes stores the answer in
# "response", tiny-math stores the whole textbook in "response", tiny-orca has a
# textbook plus a question/answer, and tiny-webtext has human/bot sides.  Using
# one generic column silently trains on prompts, UUIDs, or half the examples.
_DATASET_TEXT_FIELDS: dict[str, tuple[str, ...]] = {
    "tiny-textbooks": ("textbook", "text"),
    "tiny-strange-textbooks": ("text",),
    "tiny-codes": ("prompt", "response"),
    "tiny-math-textbooks": ("response",),
    "tiny-orca-textbooks": ("textbook", "question", "prompt", "response"),
    "tiny-webtext": ("human", "bot", "part_of_speech"),
}


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HF_HUB_TOKEN")


def _text_column(features) -> str | None:
    for name in _TEXT_COLUMNS:
        if name in features:
            return name
    # Fall back to the first string-ish column, but never to an id-like field.
    for name, feat in features.items():
        if name.lower() in {"id", "idx", "uuid"}:
            continue
        if getattr(feat, "dtype", None) == "string" or "Value" in type(feat).__name__:
            return name
    return None


def _extract_text(dataset_name: str, example: dict, features) -> str | None:
    """Build the training text from the dataset-specific columns."""
    short_name = dataset_name.rsplit("/", 1)[-1]
    fields = _DATASET_TEXT_FIELDS.get(short_name)
    if fields is None:
        column = _text_column(features)
        fields = (column,) if column else ()
    parts = [
        str(example[field]).strip()
        for field in fields
        if isinstance(example.get(field), str) and example[field].strip()
    ]
    return "\n\n".join(parts) if parts else None


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
        text = _extract_text(dataset_name, example, ds.features)
        if text is not None:
            yield text


def _mix_iterators(
    iterators: list[Iterator[str]],
    ratios: list[float],
    rng: np.random.Generator,
) -> Iterator[str]:
    """Weighted round-robin mix so every cycle contains all datasets.

    A probability-per-example draw can accidentally produce a long run from a
    single dataset.  This builds a deterministic ratio schedule (e.g. 21/21/12/
    5/8/34 slots per 100 examples), reshuffles each cycle, and skips datasets
    once they are exhausted.  That keeps the training stream mixed.
    """
    weights = [max(0.0, float(r)) for r in ratios]
    if len(weights) != len(iterators):
        raise ValueError(
            f"HF_RATIOS has {len(weights)} values for {len(iterators)} datasets"
        )
    if sum(weights) <= 0:
        weights = [1.0] * len(iterators)
    scale = 100.0 / sum(weights)
    counts = [
        0 if weight <= 0 else max(1, int(round(weight * scale)))
        for weight in weights
    ]
    active = {i for i, weight in enumerate(weights) if weight > 0}
    schedule = [i for i, count in enumerate(counts) for _ in range(count)]
    while active:
        rng.shuffle(schedule)
        for i in schedule:
            if i not in active:
                continue
            try:
                yield next(iterators[i])
            except StopIteration:
                active.remove(i)
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

    The tokenizer defaults to TinyLlama's 32k-vocab tokenizer, which matches the
    LoCED vocabulary.  Documents are concatenated through a rolling token buffer
    (with BOS/EOS separators), so trailing tokens are not dropped; only the
    final short window is padded.  A shuffle buffer mixes windows from all
    datasets before batching.
    """
    rng = np.random.default_rng(seed)
    token = _hf_token()
    ratios = ratios or tuple(1.0 for _ in datasets)

    if tokenizer_name is None:
        tokenizer_name = DEFAULT_TOKENIZER
    tokenizer = None
    if tokenizer_name is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, token=token, trust_remote_code=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        def encode(text: str) -> list[int]:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if tokenizer.bos_token_id is not None:
                ids = [int(tokenizer.bos_token_id)] + ids
            return [int(i) % vocab_size for i in ids]

    else:

        def encode(text: str) -> list[int]:
            import zlib

            seed_hash = zlib.crc32(text.encode("utf-8"))
            return [(seed_hash * 31 + i) % vocab_size for i in range(len(text))]

    iterators = [
        _iter_dataset_text(name, split, token, seed + i)
        for i, name in enumerate(datasets)
    ]
    mixed = _mix_iterators(iterators, list(ratios), rng)

    # Keep one rolling token stream and cut it into seq_len windows.  This
    # preserves every token across document boundaries instead of dropping each
    # document's short tail.  A shuffle buffer then mixes windows from all
    # datasets before they are formed into batches.
    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    window_buffer: list[list[int]] = []
    shuffle_windows = max(1024, batch_size * 64)
    token_buffer: list[int] = []
    count = 0
    for text in mixed:
        if max_examples is not None and count >= max_examples:
            break
        ids = encode(text)
        if not ids:
            continue
        token_buffer.extend(ids)
        if eos_id is not None:
            token_buffer.append(int(eos_id) % vocab_size)
        while len(token_buffer) >= seq_len:
            window_buffer.append(token_buffer[:seq_len])
            del token_buffer[:seq_len]
            if len(window_buffer) >= shuffle_windows:
                rng.shuffle(window_buffer)
                while len(window_buffer) >= batch_size:
                    batch = [window_buffer.pop() for _ in range(batch_size)]
                    yield jnp.asarray(batch, dtype=jnp.int32)
        count += 1

    # Flush the last full windows, then pad only the final partial window so no
    # source tokens are discarded.
    if token_buffer:
        token_buffer.extend([pad_id] * (seq_len - len(token_buffer)))
        window_buffer.append(token_buffer)
    rng.shuffle(window_buffer)
    while len(window_buffer) >= batch_size:
        batch = [window_buffer.pop() for _ in range(batch_size)]
        yield jnp.asarray(batch, dtype=jnp.int32)
    if window_buffer:
        while len(window_buffer) < batch_size:
            window_buffer.append([pad_id] * seq_len)
        yield jnp.asarray(window_buffer, dtype=jnp.int32)


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
    last_step = start_step
    did_step = False
    schedule_steps = max(1, int(getattr(config, "learning_rate_schedule_steps", steps)))
    # Data-parallel input: global batch is sharded across the "data" mesh axis,
    # so each device sees per_device_batch_size examples.
    data_sharding = NamedSharding(mesh, P("data", None))

    for step in range(start_step, steps):
        try:
            tokens = next(batch_iter)
        except StopIteration:
            max_logging.log(f"dataset exhausted at step {step}")
            break

        tokens = jax.device_put(tokens, data_sharding)
        loss, metrics = train_step(model, optimizer, tokens)
        last_metrics = {k: float(v) for k, v in metrics.items()}
        last_step = step
        did_step = True

        if log_every > 0 and (step % log_every == 0 or step == steps - 1):
            lr = float(schedule(min(step, schedule_steps - 1)))
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
    if ckpt_dir and did_step:
        save_checkpoint(ckpt_dir, model, optimizer)
        max_logging.log(f"final checkpoint at step {last_step}")

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
    data_parallelism = max(1, int(getattr(config, "ici_data_parallelism", 1)))
    global_batch = int(config.per_device_batch_size) * data_parallelism

    batches = hf_streaming_batches(
        loced_cfg.vocab_size,
        int(config.max_target_length),
        global_batch,
        datasets=datasets,
        ratios=ratios,
        seed=seed,
        tokenizer_name=tokenizer_name,
    )

    _, _, metrics = train_timed(
        config,
        batches,
        log_every=int(os.environ.get("HF_LOG_EVERY", "50")),
        long_run_threshold_s=float(
            os.environ.get("LOCED_CKPT_AFTER_S", str(LONG_RUN_THRESHOLD_S))
        ),
    )
    max_logging.log(f"finished LoCED HF run: {metrics}")


if __name__ == "__main__":
    main()
