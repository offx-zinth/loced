"""Real-data streaming for the LoCED reference path (Hugging Face datasets).

The 12-layer LoCED model used to be trained on synthetic tokens only.  This
module ports the proven streaming design of ``loced.maxtext.hf_train`` to the
pure-JAX reference path so the same model can be trained on real text:

  * one text iterator per dataset;
  * a deterministic weighted round-robin mix, so every cycle of the stream
    contains every dataset (a per-example probability draw can produce long
    single-dataset runs);
  * a ROLLING token buffer: documents are concatenated with an EOS separator,
    so no token is dropped at a document boundary;
  * a shuffle buffer of fixed-length windows, then batching.

The eight default datasets do NOT share a schema, so every dataset has an
explicit text-field mapping (``DATASET_TEXT_FIELDS``).  A generic "first
string column" fallback exists for unknown datasets, but it is never used for
the default eight: a silent fallback would train on titles, UUIDs or
half-examples.

The Hugging Face token is read from the environment (``HF_TOKEN`` /
``HF_HUB_TOKEN``); it is never stored in this file.  Six of the eight default
datasets are gated, so they need a token whose owner has accepted the
dataset terms.  A dataset that cannot be loaded (missing token, terms not
accepted, network error) is reported and skipped -- it never aborts the run.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

import jax.numpy as jnp

LOGGER = logging.getLogger("loced.hf_data")

# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------

#: The eight real datasets used by the 12-layer reference path.
DEFAULT_DATASETS: tuple[str, ...] = (
    "nampdn-ai/tiny-textbooks",
    "nampdn-ai/tiny-strange-textbooks",
    "nampdn-ai/tiny-math-textbooks",
    "nampdn-ai/tiny-codes",
    "nampdn-ai/tiny-code-textbooks",
    "open-phi/programming_books_llama",
    "nampdn-ai/tiny-webtext",
    "SciPhi/textbooks-are-all-you-need-lite",
)

#: GPT-2's vocabulary is exactly 50,257 entries, so token ids need no modulo.
DEFAULT_TOKENIZER = "gpt2"
DEFAULT_VOCAB_SIZE = 50257

#: Dataset-specific text columns (verified against the HF datasets-server).
#:
#: The datasets do not share a schema, and several of them contain decoy
#: columns that would silently train the model on the wrong content:
#:
#:   tiny-textbooks               ``text`` is a ~500-char excerpt, ``textbook``
#:                                 is the full ~3,300-char lesson
#:   tiny-math-textbooks          ``response`` is the textbook body, ``lesson``
#:                                 is a one-word topic
#:   tiny-codes                   ``prompt`` is the task, ``response`` the code
#:   tiny-code-textbooks          ``content`` is the raw source code,
#:                                 ``response`` is the explanation textbook
#:   programming_books_llama      ``markdown`` is the chapter, ``context`` is
#:                                 usually an empty list
#:   tiny-webtext                 ``human`` is the web text, ``bot`` its
#:                                 continuation, ``part_of_speech`` a label
#:   textbooks-are-all-you-need   ``completion`` is the chapter body, ``notes``
#:                                 the generation notes
DATASET_TEXT_FIELDS: dict[str, tuple[str, ...]] = {
    "tiny-textbooks": ("textbook", "text"),
    "tiny-strange-textbooks": ("text",),
    "tiny-math-textbooks": ("response",),
    "tiny-codes": ("prompt", "response"),
    "tiny-code-textbooks": ("response", "content"),
    "programming_books_llama": ("markdown",),
    "tiny-webtext": ("human", "bot"),
    "textbooks-are-all-you-need-lite": ("completion", "notes"),
}

#: Column names commonly used for raw text (generic fallback only).
_TEXT_COLUMNS = ("text", "content", "body", "document", "prompt", "completion")

#: Columns that look like text but are identifiers.
_ID_LIKE = {"id", "idx", "uuid", "uuid4", "index"}


# ---------------------------------------------------------------------------
# helpers (pure, offline-safe)
# ---------------------------------------------------------------------------

def short_name(dataset_name: str) -> str:
    """``"nampdn-ai/tiny-codes"`` -> ``"tiny-codes"``."""
    return dataset_name.rsplit("/", 1)[-1]


def hf_token() -> str | None:
    """The Hugging Face token from the environment (never hard-coded)."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HF_HUB_TOKEN")


def clamp_token_ids(ids: Iterable[int], vocab_size: int) -> list[int]:
    """Clamp token ids into ``[0, vocab_size)``.

    GPT-2's vocabulary is exactly 50,257 entries, so its ids already fit; the
    clamp keeps a mismatched tokenizer from indexing the embedding out of
    range.
    """
    if vocab_size < 1:
        raise ValueError(f"vocab_size must be >= 1, got {vocab_size}")
    top = vocab_size - 1
    return [min(max(int(i), 0), top) for i in ids]


def text_column(features: Any) -> str | None:
    """Generic fallback: the first plain string column that is not id-like.

    Only used for datasets that are not in ``DATASET_TEXT_FIELDS``.  List
    columns are only used as a last resort (their values are not strings, so
    :func:`extract_text` skips them).
    """
    if not features:
        return None
    plain: list[str] = []
    sequences: list[str] = []
    for name, feat in features.items():
        if name.lower() in _ID_LIKE:
            continue
        if getattr(feat, "dtype", None) != "string":
            continue
        if type(feat).__name__ == "Value":
            plain.append(name)
        else:
            sequences.append(name)
    for name in _TEXT_COLUMNS:
        if name in plain:
            return name
    if plain:
        return plain[0]
    return sequences[0] if sequences else None


def resolve_text_fields(dataset_name: str, features: Any = None) -> tuple[str, ...]:
    """The text fields used for one dataset (explicit map, else fallback)."""
    fields = DATASET_TEXT_FIELDS.get(short_name(dataset_name))
    if fields:
        return tuple(fields)
    column = text_column(features)
    return (column,) if column else ()


def extract_text(dataset_name: str, example: dict,
                 features: Any = None) -> str | None:
    """Build the training text from the dataset-specific columns.

    Only string values are concatenated; missing fields, empty strings and
    list-valued columns (e.g. the ``context`` column of
    ``open-phi/programming_books_llama``) are skipped.
    """
    fields = resolve_text_fields(dataset_name, features)
    parts: list[str] = []
    for field in fields:
        value = example.get(field) if isinstance(example, dict) else None
        if not isinstance(value, str):
            continue
        text = value.strip()
        if not text:
            continue
        parts.append(text)
    return "\n\n".join(parts) if parts else None


def describe_dataset_fields(datasets: Sequence[str] = DEFAULT_DATASETS) -> list[str]:
    """One auditable line per dataset: the name and the fields it will use."""
    lines: list[str] = []
    for name in datasets:
        fields = DATASET_TEXT_FIELDS.get(short_name(name))
        if fields:
            lines.append(f"{name}: fields={','.join(fields)}")
        else:
            lines.append(
                f"{name}: fields=(generic fallback: first string column)")
    return lines


def log_dataset_fields(datasets: Sequence[str] = DEFAULT_DATASETS) -> None:
    """Print the dataset -> text-field map (one line per dataset)."""
    print("loced.hf_data: dataset text-field map", flush=True)
    for line in describe_dataset_fields(datasets):
        print(f"  {line}", flush=True)


# ---------------------------------------------------------------------------
# streaming
# ---------------------------------------------------------------------------

def local_dataset_files(dataset_name: str,
                        local_root: str | os.PathLike) -> list[str]:
    """The local train parquet files of one prefetched dataset.

    ``snapshot_download`` mirrors a dataset repository, so the files live under
    a subdirectory named after the dataset.  Test/validation shards must not
    leak into the training pass.
    """
    root = Path(local_root) / short_name(dataset_name)
    files = sorted(
        str(path)
        for path in root.rglob("*.parquet")
        if "test" not in path.name.lower()
        and "validation" not in path.name.lower()
    )
    if not files:
        raise FileNotFoundError(
            f"No local parquet files for {dataset_name} under {root}. "
            "Prefetch the datasets or unset --local-data."
        )
    return files


def iter_dataset_text(
    dataset_name: str,
    split: str = "train",
    token: str | None = None,
    seed: int = 0,
    shuffle_buffer: int = 10_000,
    local_root: str | os.PathLike | None = None,
) -> Iterator[str]:
    """Yield the text examples of one streaming HF dataset or local mirror."""
    from datasets import load_dataset  # imported lazily so CPU envs keep working

    if local_root is not None:
        files = local_dataset_files(dataset_name, local_root)
        ds = load_dataset(
            "parquet",
            data_files=files,
            split=split,
            streaming=True,
        )
    else:
        try:
            ds = load_dataset(
                dataset_name,
                split=split,
                streaming=True,
                token=token,
            )
        except Exception as exc:
            # `trust_remote_code` was removed in datasets 5.x, but older
            # releases still need it for datasets shipped as a loading script.
            message = str(exc).lower()
            if "trust_remote_code" not in message \
                    and "loading script" not in message:
                raise
            ds = load_dataset(
                dataset_name,
                split=split,
                streaming=True,
                token=token,
                trust_remote_code=True,
            )
    features = ds.features
    if not resolve_text_fields(dataset_name, features):
        raise ValueError(
            f"No text column found in {dataset_name}; "
            f"features={dict(features)}"
        )
    if shuffle_buffer > 0:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    for example in ds:
        text = extract_text(dataset_name, example, features)
        if text is not None:
            yield text


def _guarded(dataset_name: str, iterator: Iterator[str]) -> Iterator[str]:
    """Stop (instead of crashing) when one dataset fails mid-stream."""
    try:
        yield from iterator
    except Exception as exc:  # pragma: no cover - network/backend dependent
        LOGGER.warning(
            "dataset %s failed mid-stream and was skipped: %s: %s",
            dataset_name, type(exc).__name__, exc)


def open_dataset_streams(
    datasets: Sequence[str],
    split: str = "train",
    token: str | None = None,
    seed: int = 0,
    local_root: str | os.PathLike | None = None,
) -> tuple[list[tuple[int, str, Iterator[str]]], list[str]]:
    """Open one text stream per dataset; skip the ones that cannot be loaded.

    Returns ``(alive, failures)`` where ``alive`` holds ``(index, name,
    iterator)`` triples and ``failures`` holds one message per skipped
    dataset.  A dataset that fails (missing token, terms not accepted,
    network error) is reported and never aborts the run.
    """
    alive: list[tuple[int, str, Iterator[str]]] = []
    failures: list[str] = []
    for i, name in enumerate(datasets):
        iterator = iter_dataset_text(
            name, split=split, token=token, seed=seed + i,
            local_root=local_root)
        try:
            first = next(iterator)
        except StopIteration:
            message = f"dataset {name} is empty and was skipped"
            LOGGER.warning("%s", message)
            failures.append(message)
            continue
        except Exception as exc:
            message = (f"dataset {name} could not be loaded and was skipped: "
                       f"{type(exc).__name__}: {exc}")
            LOGGER.warning("%s", message)
            failures.append(message)
            print(f"[hf_data] WARNING: {message}", flush=True)
            continue
        alive.append((i, name, _chain(first, iterator)))
    return alive, failures


def mix_iterators(iterators: Sequence[Iterator[str]],
                  ratios: Sequence[float],
                  rng: np.random.Generator) -> Iterator[str]:
    """Weighted round-robin mix so every cycle contains all datasets.

    A probability-per-example draw can accidentally produce a long run from a
    single dataset.  This builds a deterministic ratio schedule (e.g. 21/21/12/
    5/8/34 slots per 100 examples), reshuffles each cycle, and skips datasets
    once they are exhausted.  That keeps the training stream mixed.
    """
    weights = [max(0.0, float(r)) for r in ratios]
    if len(weights) != len(iterators):
        raise ValueError(
            f"got {len(weights)} ratios for {len(iterators)} datasets")
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


def build_tokenizer(tokenizer_name: str, token: str | None = None):
    """Load the tokenizer (lazily, so the module imports without network)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _encode(tokenizer, text: str, vocab_size: int) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is not None:
        ids = [int(bos)] + list(ids)
    return clamp_token_ids(ids, vocab_size)


def hf_batches(
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    seq_len: int = 128,
    batch_size: int = 8,
    datasets: Sequence[str] = DEFAULT_DATASETS,
    ratios: Sequence[float] | None = None,
    seed: int = 0,
    split: str = "train",
    tokenizer_name: str = DEFAULT_TOKENIZER,
    max_examples: int | None = None,
    local_root: str | os.PathLike | None = None,
) -> Iterator[jnp.ndarray]:
    """Yield int32 ``[batch, seq]`` token batches from streaming HF datasets.

    Documents are concatenated through a rolling token buffer (with an EOS
    separator), so trailing tokens are not dropped; only the final short
    window is padded.  A shuffle buffer mixes windows from all datasets
    before batching.  When ``local_root`` is set, parquet files are read from
    a prefetched local mirror (one subdirectory per dataset name).

    A dataset that cannot be loaded (missing token, terms not accepted,
    network error) is reported and skipped; the run continues with the rest.
    """
    rng = np.random.default_rng(seed)
    token = hf_token()
    ratios = list(ratios) if ratios else [1.0] * len(datasets)
    if len(ratios) != len(datasets):
        raise ValueError(
            f"got {len(ratios)} ratios for {len(datasets)} datasets")

    tokenizer = build_tokenizer(tokenizer_name, token)
    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    eos_id = getattr(tokenizer, "eos_token_id", None)

    # Prime every dataset: a dataset that cannot be loaded is skipped here,
    # with the reason in the log, and the run continues.
    alive, failures = open_dataset_streams(
        datasets, split=split, token=token, seed=seed, local_root=local_root)
    if not alive:
        raise RuntimeError(
            "no Hugging Face dataset could be loaded: "
            + ("; ".join(failures) if failures else "all datasets were empty")
            + ". Check HF_TOKEN (and the gated-dataset terms) and retry.")
    alive_names = [name for _, name, _ in alive]
    alive_ratios = [float(ratios[index]) for index, _, _ in alive]
    log_dataset_fields(alive_names)

    mixed = mix_iterators(
        [_guarded(name, it) for _, name, it in alive],
        alive_ratios,
        rng,
    )

    # One rolling token stream, cut into seq_len windows: every token survives
    # a document boundary instead of dropping each document's short tail.
    shuffle_windows = max(1024, batch_size * 64)
    token_buffer: list[int] = []
    window_buffer: list[list[int]] = []
    count = 0
    for text in mixed:
        if max_examples is not None and count >= max_examples:
            break
        ids = _encode(tokenizer, text, vocab_size)
        if not ids:
            continue
        token_buffer.extend(ids)
        if eos_id is not None:
            token_buffer.extend(clamp_token_ids([eos_id], vocab_size))
        while len(token_buffer) >= seq_len:
            window_buffer.append(token_buffer[:seq_len])
            del token_buffer[:seq_len]
            if len(window_buffer) >= shuffle_windows:
                rng.shuffle(window_buffer)
                while len(window_buffer) >= batch_size:
                    batch = window_buffer[-batch_size:]
                    del window_buffer[-batch_size:]
                    yield jnp.asarray(batch, dtype=jnp.int32)
        count += 1

    # Flush the last full windows, then pad only the final partial window so
    # no source tokens are discarded.
    if token_buffer:
        token_buffer.extend([pad_id] * (seq_len - len(token_buffer)))
        window_buffer.append(token_buffer)
    rng.shuffle(window_buffer)
    while len(window_buffer) >= batch_size:
        batch = window_buffer[-batch_size:]
        del window_buffer[-batch_size:]
        yield jnp.asarray(batch, dtype=jnp.int32)
    if window_buffer:
        while len(window_buffer) < batch_size:
            window_buffer.append([pad_id] * seq_len)
        yield jnp.asarray(window_buffer, dtype=jnp.int32)


def _chain(first: str, iterator: Iterator[str]) -> Iterator[str]:
    """A generator that yields ``first`` and then the rest of ``iterator``."""
    yield first
    yield from iterator


# The schema choice is printed once at import time so a training log records
# exactly which columns fed the model.
log_dataset_fields(DEFAULT_DATASETS)
