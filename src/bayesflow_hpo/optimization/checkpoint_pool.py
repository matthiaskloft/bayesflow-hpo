"""Top-k checkpoint pool for persisting the best trial weights.

During HPO every trial trains to a temporary checkpoint.  This module
maintains a small pool of the best *k* trials' weights on disk so that
the winning configurations can be loaded immediately after the study
without retraining.

The pool directory layout::

    checkpoints/
        trial_042/
            ...keras or .pt files...
        trial_017/
            ...
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default pool size.
DEFAULT_POOL_SIZE = 5


class CheckpointPool:
    """Maintains the top-*k* trial checkpoints on disk.

    Parameters
    ----------
    pool_dir
        Root directory for the checkpoint pool (default ``"checkpoints"``).
    pool_size
        Maximum number of checkpoints to keep (default 5).
    """

    def __init__(
        self,
        pool_dir: str | Path = "checkpoints",
        pool_size: int = DEFAULT_POOL_SIZE,
    ):
        self.pool_dir = Path(pool_dir)
        self.pool_size = pool_size
        # (objective_value, trial_number) → checkpoint path
        self._entries: list[tuple[float, int, Path]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def maybe_save(
        self,
        trial_number: int,
        objective_value: float,
        workflow: Any,
    ) -> bool:
        """Save *workflow* weights if they're in the current top-k.

        Returns ``True`` when the checkpoint was saved (or the pool was
        updated), ``False`` otherwise.
        """
        if (
            len(self._entries) >= self.pool_size
            and objective_value >= self._entries[-1][0]
        ):
            return False

        dest = self.pool_dir / f"trial_{trial_number:04d}"
        try:
            dest.mkdir(parents=True, exist_ok=True)
            approximator = getattr(workflow, "approximator", workflow)
            approximator.save_weights(str(dest / "weights.weights.h5"))
        except Exception:
            logger.warning(
                "Failed to save checkpoint for trial %d", trial_number,
                exc_info=True,
            )
            return False

        self._entries.append((objective_value, trial_number, dest))
        self._entries.sort(key=lambda e: e[0])

        # Evict worst if pool is over capacity.
        while len(self._entries) > self.pool_size:
            _, evicted_num, evicted_path = self._entries.pop()
            _safe_rmtree(evicted_path)
            logger.debug("Evicted trial %d from checkpoint pool", evicted_num)

        return True

    @property
    def best_checkpoint_dir(self) -> Path | None:
        """Path to the best checkpoint, or ``None`` if the pool is empty."""
        if not self._entries:
            return None
        return self._entries[0][2]

    @property
    def trial_numbers(self) -> list[int]:
        """Trial numbers currently in the pool, sorted by objective."""
        return [num for _, num, _ in self._entries]

    def cleanup(self) -> None:
        """Remove the entire pool directory."""
        _safe_rmtree(self.pool_dir)
        self._entries.clear()


def _safe_rmtree(path: Path) -> None:
    try:
        if path.exists():
            shutil.rmtree(path)
    except OSError:
        logger.debug("Could not remove %s", path, exc_info=True)
