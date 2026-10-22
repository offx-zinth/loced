"""MaxText v0.2.5 integration for LoCED.

Layout
------
* ``modeling``  -- Flax NNX model graph (framework-facing, algorithm-free shell)
* ``config``    -- MaxText pydantic config class + mapping to ``LoCEDConfig``
* ``train``     -- MaxText-style trainer (mesh, optimizer, checkpoints)

The reference implementation lives in the parent package and remains the
single source of truth for the algorithm; this package only adds the
framework contracts (NNX parameters, sharding metadata, config, training
loop).  See ``MAXTEXT.md`` for the migration notes.
"""

from __future__ import annotations

__all__ = ["modeling", "config", "train"]

try:
    from . import modeling  # noqa: F401
except ImportError:  # flax not installed in the reference CPU environment
    modeling = None

try:
    from . import config  # noqa: F401
except ImportError:  # pragma: no cover
    config = None

# ``train`` is intentionally not imported eagerly: it requires the full
# MaxText stack (including its data/checkpoint utilities), while ``modeling``
# and ``config`` are usable from the CPU reference environment.  Import it
# explicitly with ``from loced.maxtext import train`` or
# ``import loced.maxtext.train`` in a MaxText container.
