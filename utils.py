"""Shared utilities: logging, reproducibility, VRAM measurement, timing.

torch is imported lazily (inside functions, wrapped in try/except) rather
than at module level: this module must import cleanly on machines without
a GPU or torch installed, including this local dev environment — see
technical_lessons_learned.md. Only `train.py`/`model.py`, which genuinely
require a GPU, are expected to run where torch is present.
"""

from __future__ import annotations

import logging
import random
import time
from types import TracebackType

import numpy as np

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging() -> None:
    """Configure a single, consistent log format for all scripts.

    `force=True` is required: `logging.basicConfig` is a silent no-op if
    the root logger already has handlers attached (e.g. pytest, or a
    prior call to this function elsewhere in the same process) - without
    it, this function would sometimes configure logging and sometimes
    do nothing, depending on what ran before it.
    """
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, force=True)


def set_seed(seed: int) -> None:
    """Fix random/numpy/torch seeds for reproducibility.

    Args:
        seed: The seed value to apply to every RNG this project uses.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def measure_vram() -> dict:
    """Return current GPU VRAM usage in bytes.

    Used for the 16-bit vs. 4-bit VRAM footprint comparison in
    technical_assignment.md. Returns zeros (not an exception) when no
    GPU/torch is available, so it is safe to call during local, CPU-only
    development without special-casing every call site.

    Returns:
        A dict with `allocated_bytes` and `reserved_bytes`.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return {"allocated_bytes": 0, "reserved_bytes": 0}
        return {
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
        }
    except ImportError:
        return {"allocated_bytes": 0, "reserved_bytes": 0}


class Timer:
    """Context manager measuring wall-clock elapsed time in seconds.

    Used to compute generation latency / tokens-per-second (see
    technical_assignment.md metrics). Usage:

        with Timer() as timer:
            ...
        print(timer.elapsed_seconds)
    """

    def __enter__(self) -> Timer:  # noqa: PYI034 - typing.Self needs py3.11+
        self._start = time.perf_counter()
        self.elapsed_seconds: float = 0.0
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.elapsed_seconds = time.perf_counter() - self._start
