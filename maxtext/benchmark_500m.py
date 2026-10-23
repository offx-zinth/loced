"""Single-device ~500M LoCED smoke benchmark.

This is deliberately separate from ``test_integration.py``: the integration
suite is the exact 0.55M parity suite.  Here we only answer the scale question:

  * does the ~500M model fit on one accelerator,
  * does MaxText Muon take a real update,
  * how long do the first (compile) and second (steady) steps take,
  * what is the peak device memory?

Usage::

    python -m loced.maxtext.benchmark_500m

The model/run config defaults to ``loced/maxtext/configs/loced_500m_tpu.yml``;
override it with ``LOCED_500M_CONFIG``.  Any extra CLI arguments are MaxText
overrides, e.g. ``per_device_batch_size=1 steps=2``.
"""

from __future__ import annotations

import json
import os
import sys
import time

import jax
import numpy as np
from maxtext.configs import pyconfig

try:  # package import
    from .config import LoCEDMaxTextConfig, loced_config_from_maxtext
    from .train import (
        create_model_and_mesh,
        create_optimizer,
        synthetic_batches,
        train_step,
    )
except ImportError:  # direct script execution
    from loced.maxtext.config import LoCEDMaxTextConfig, loced_config_from_maxtext
    from loced.maxtext.train import (
        create_model_and_mesh,
        create_optimizer,
        synthetic_batches,
        train_step,
    )


DEFAULT_CONFIG = "loced/maxtext/configs/loced_500m_tpu.yml"


def _metrics_to_float(metrics: dict) -> dict:
    return {k: float(v) for k, v in metrics.items()}


def main(argv: list[str] | None = None) -> dict:
    argv = list(sys.argv if argv is None else argv)
    config_path = os.environ.get("LOCED_500M_CONFIG", DEFAULT_CONFIG)
    config = pyconfig.initialize(
        [argv[0], config_path, "skip_jax_distributed_system=true", *argv[1:]],
        config_class=LoCEDMaxTextConfig,
    )
    cfg = loced_config_from_maxtext(config)

    expected_backend = os.environ.get("LOCED_EXPECT_BACKEND")
    backend = jax.default_backend()
    devices = jax.devices()
    if expected_backend:
        assert backend == expected_backend, (
            f"Expected {expected_backend}, got {backend}. "
            "Check the Colab/TPU runtime."
        )

    print("config  :", config_path)
    print("backend :", backend)
    print("devices :", devices)
    print("shape   :", f"d={cfg.hidden_size} layers={cfg.num_layers} "
          f"experts={cfg.num_experts} I={cfg.intermediate_size} "
          f"vocab={cfg.vocab_size} seq={cfg.max_seq_len}")

    t0 = time.perf_counter()
    model, mesh = create_model_and_mesh(config)
    build_s = time.perf_counter() - t0
    params = model.count_params()
    print("params  :", f"{params:,} ({params / 1e6:.2f}M)")
    print("mesh    :", dict(mesh.shape))
    print("build   :", f"{build_s:.1f}s")

    # ~500.2M by construction; keep a 1% band so a field typo cannot pass.
    assert 495_000_000 <= params <= 505_000_000, (
        f"Expected ~500M parameters, got {params:,}. "
        "Check configs/models/loced-500m.yml."
    )

    schedule, optimizer = create_optimizer(config, model, mesh)
    tokens = next(
        iter(
            synthetic_batches(
                cfg.vocab_size,
                cfg.max_seq_len,
                int(config.per_device_batch_size),
                1,
                seed=0,
            )
        )
    )

    # Small leaf so the change check does not copy a 500M array.
    before = np.asarray(model.W_ik.get_value()).copy()

    t0 = time.perf_counter()
    loss0, metrics0 = train_step(model, optimizer, tokens)
    jax.block_until_ready(loss0)
    first_step_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    loss1, metrics1 = train_step(model, optimizer, tokens)
    jax.block_until_ready(loss1)
    second_step_s = time.perf_counter() - t0

    after = np.asarray(model.W_ik.get_value())
    changed = not np.array_equal(before, after)
    step = int(optimizer.step.get_value())
    loss_first = float(loss0)
    loss_second = float(loss1)
    assert np.isfinite(loss_first) and np.isfinite(loss_second)
    assert step == 2, f"Expected 2 optimizer steps, got {step}"
    assert changed, "MaxText Muon did not change the model"

    memory = {}
    for device in devices:
        stats = device.memory_stats() or {}
        memory[str(device)] = {
            "peak_bytes_in_use": stats.get("peak_bytes_in_use"),
            "bytes_limit": stats.get("bytes_limit"),
        }

    report = {
        "config": config_path,
        "backend": backend,
        "devices": len(devices),
        "device_kind": str(devices[0].device_kind),
        "params": params,
        "architecture": {
            "hidden_size": cfg.hidden_size,
            "num_layers": cfg.num_layers,
            "num_visits": cfg.num_visits,
            "num_experts": cfg.num_experts,
            "intermediate_size": cfg.intermediate_size,
            "vocab_size": cfg.vocab_size,
            "seq_len": cfg.max_seq_len,
            "encode": list(cfg.encode_layers),
            "decode": list(cfg.decode_layers),
        },
        "mesh": {str(k): int(v) for k, v in mesh.shape.items()},
        "build_s": build_s,
        "first_step_s": first_step_s,
        "second_step_s": second_step_s,
        "loss_first": loss_first,
        "loss_second": loss_second,
        "metrics_first": _metrics_to_float(metrics0),
        "metrics_second": _metrics_to_float(metrics1),
        "optimizer_step": step,
        "param_changed": changed,
        "learning_rate": float(schedule(0)),
        "memory": memory,
    }

    print("step 0  :", f"{first_step_s:.2f}s (includes compile)")
    print("step 1  :", f"{second_step_s:.2f}s")
    print("loss    :", f"{loss_first:.6f} -> {loss_second:.6f}")
    print("muon step:", step, "changed:", changed)
    for name, stats in memory.items():
        print("memory  :", name, stats)

    report_path = os.environ.get("LOCED_500M_REPORT")
    if report_path:
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print("report  :", report_path)
    return report


if __name__ == "__main__":
    main()
