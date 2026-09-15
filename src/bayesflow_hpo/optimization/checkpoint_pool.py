"""Checkpoint pools for persisting trial weights.

During HPO every trial trains to a temporary checkpoint.  This module
maintains a small pool of the best *k* trials' weights on disk so that
the winning configurations can be loaded immediately after the study
without retraining.

A study's *pruned* trials are discarded by that pool, because a pruned
trial raises before it is ever scored.  That is the right default when
the question is "which model do I ship", but pruned trials are the only
under-trained models a study produces, and diagnostic work needs them
(bayesflow-hpo#106).  ``pruned_pool_size`` opts into retaining a bounded
sample of them in a *separate* pool, so pruned trials can never evict a
scored one.

The pool directory layout::

    checkpoints/
        trial_042/
            weights.weights.h5
            checkpoint.json
        trial_017/
            ...
        pruned/
            trial_031/
                weights.weights.h5
                checkpoint.json
"""

from __future__ import annotations

import json
import logging
import random
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default pool size.
DEFAULT_POOL_SIZE = 5

#: Default size of the pruned-trial pool.  Zero, i.e. off: retaining pruned
#: weights is opt-in.
DEFAULT_PRUNED_POOL_SIZE = 0

_WEIGHTS_FILE = "weights.weights.h5"
_METADATA_FILE = "checkpoint.json"


class CheckpointPool:
    """Maintains the top-*k* trial checkpoints on disk.

    Only *weights* are written (``weights.weights.h5``), never a full
    serialized model.  ``keras.saving.load_model`` on a checkpoint
    directory therefore fails: the architecture is not stored alongside
    the weights.  Rebuild the approximator from the trial's
    ``trial.params`` first, then call ``approximator.load_weights()`` on
    the file.  A mismatch between the rebuilt architecture and the saved
    weights fails loudly on shape, so this is recoverable rather than
    silent -- but the reason is not obvious from the error.

    Each checkpoint directory carries a ``checkpoint.json`` sidecar
    recording the trial number, the trial's state, its objective value
    and -- for pruned trials -- the validation step it stopped at.  A
    pruned trial's weights are only meaningful together with the rung
    they were taken at, and that cannot be inferred from the file.

    Parameters
    ----------
    pool_dir
        Root directory for the checkpoint pool (default ``"checkpoints"``).
    pool_size
        Maximum number of scored-trial checkpoints to keep (default 5).
    pruned_pool_size
        Maximum number of *pruned*-trial checkpoints to keep, in a separate
        pool under ``pool_dir / "pruned"``.  Default 0, i.e. pruned weights
        are discarded as before.  When the cap is reached, retention is a
        uniform random sample of every pruned trial the pool was offered:
        the *n*-th offer is kept with probability ``pruned_pool_size / n``
        and replaces a uniformly chosen incumbent.  Not top-*k*, because
        pruned trials stop at different rungs and their objective values
        are not comparable across rungs; a uniform sample instead gives
        coverage of the whole quality range, which is what the diagnostic
        use cases need.
    seed
        Seed for the retention sampler.  ``None`` (default) uses an
        unseeded :class:`random.Random`.
    """

    def __init__(
        self,
        pool_dir: str | Path = "checkpoints",
        pool_size: int = DEFAULT_POOL_SIZE,
        pruned_pool_size: int = DEFAULT_PRUNED_POOL_SIZE,
        seed: int | None = None,
    ):
        self.pool_dir = Path(pool_dir)
        self.pool_size = pool_size
        self.pruned_pool_size = pruned_pool_size
        # (objective_value, trial_number) -> checkpoint path
        self._entries: list[tuple[float, int, Path]] = []
        # (trial_number, checkpoint path), retention order, not ranked.
        self._pruned_entries: list[tuple[int, Path]] = []
        # Every pruned trial OFFERED, not every one kept: the uniform
        # retention probability is over the population, not the pool.
        self._pruned_seen = 0
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def pruned_pool_dir(self) -> Path:
        """Root directory of the pruned-trial pool."""
        return self.pool_dir / "pruned"

    def maybe_save(
        self,
        trial_number: int,
        objective_value: float,
        approximator: Any,
    ) -> bool:
        """Save *approximator* weights if they're in the current top-k.

        Returns ``True`` when the checkpoint was saved (or the pool was
        updated), ``False`` otherwise.
        """
        if (
            len(self._entries) >= self.pool_size
            and objective_value >= self._entries[-1][0]
        ):
            return False

        dest = self.pool_dir / f"trial_{trial_number:04d}"
        if not _write_checkpoint(
            dest,
            approximator,
            {
                "trial_number": trial_number,
                "state": "complete",
                "objective_value": float(objective_value),
            },
            trial_number,
        ):
            return False

        self._entries.append((objective_value, trial_number, dest))
        self._entries.sort(key=lambda e: e[0])

        # Evict worst if pool is over capacity.
        while len(self._entries) > self.pool_size:
            _, evicted_num, evicted_path = self._entries.pop()
            _safe_rmtree(evicted_path)
            logger.debug("Evicted trial %d from checkpoint pool", evicted_num)

        return True

    def save_pruned(
        self,
        trial_number: int,
        approximator: Any,
        step: int | None = None,
        objective_value: float | None = None,
    ) -> bool:
        """Retain a pruned trial's weights, if the pruned pool is enabled.

        Parameters
        ----------
        trial_number
            Optuna trial number.
        approximator
            The approximator as it stood when pruning was raised.
        step
            Intermediate-validation step the trial was pruned at, recorded
            in the sidecar.  ``None`` when the trial was pruned before any
            intermediate validation ran.
        objective_value
            Scalar summary of the trial's scores at that step, if known.
            Recorded but not used for ranking -- see the class docstring.

        Returns
        -------
        bool
            ``True`` when the checkpoint was written and retained.
        """
        if self.pruned_pool_size <= 0:
            return False

        n_offered = self._pruned_seen + 1

        evict_slot: int | None = None
        if len(self._pruned_entries) >= self.pruned_pool_size:
            # Keep the n-th offer with probability k/n, replacing a
            # uniformly chosen incumbent. This is what keeps every pruned
            # trial of the study equally likely to be in the final pool,
            # without knowing the population size up front. Verified
            # empirically by
            # ``test_retention_is_uniform_over_the_population``.
            if self._rng.random() >= self.pruned_pool_size / n_offered:
                self._pruned_seen = n_offered
                return False
            evict_slot = self._rng.randrange(self.pruned_pool_size)

        dest = self.pruned_pool_dir / f"trial_{trial_number:04d}"
        if not _write_checkpoint(
            dest,
            approximator,
            {
                "trial_number": trial_number,
                "state": "pruned",
                "pruned_at_step": step,
                "objective_value": (
                    None if objective_value is None else float(objective_value)
                ),
            },
            trial_number,
        ):
            # A write that never landed must not consume a draw: counting it
            # would make the retained set a non-uniform sample of the trials
            # that were actually retainable.
            return False

        self._pruned_seen = n_offered

        if evict_slot is None:
            self._pruned_entries.append((trial_number, dest))
        else:
            evicted_num, evicted_path = self._pruned_entries[evict_slot]
            # A re-offer of a trial already in the pool can draw its own
            # slot, and removing `evicted_path` would then delete the
            # checkpoint just written to `dest`.
            if evicted_path != dest:
                _safe_rmtree(evicted_path)
                logger.debug(
                    "Evicted pruned trial %d from pruned checkpoint pool",
                    evicted_num,
                )
            self._pruned_entries[evict_slot] = (trial_number, dest)

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

    @property
    def pruned_trial_numbers(self) -> list[int]:
        """Pruned trial numbers currently retained, ascending."""
        return sorted(num for num, _ in self._pruned_entries)

    def cleanup(self) -> None:
        """Remove the entire pool directory."""
        _safe_rmtree(self.pool_dir)
        self._entries.clear()
        self._pruned_entries.clear()
        self._pruned_seen = 0


def _write_checkpoint(
    dest: Path,
    approximator: Any,
    metadata: dict[str, Any],
    trial_number: int,
) -> bool:
    """Write weights plus the ``checkpoint.json`` sidecar to *dest*."""
    try:
        dest.mkdir(parents=True, exist_ok=True)
        approximator.save_weights(str(dest / _WEIGHTS_FILE))
    except Exception:
        logger.warning(
            "Failed to save checkpoint for trial %d", trial_number,
            exc_info=True,
        )
        return False

    try:
        (dest / _METADATA_FILE).write_text(
            json.dumps(
                {
                    **metadata,
                    "weights_file": _WEIGHTS_FILE,
                    "note": (
                        "Weights only. Rebuild the approximator from "
                        "trial.params, then load_weights(); "
                        "keras.saving.load_model() cannot read this."
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError:
        # The weights are the payload; a missing sidecar is a loss of
        # provenance, not of the checkpoint.
        logger.debug(
            "Could not write checkpoint metadata for trial %d", trial_number,
            exc_info=True,
        )
    return True


def _safe_rmtree(path: Path) -> None:
    """Remove a directory tree, suppressing OS errors (e.g. file locks on Windows)."""
    try:
        if path.exists():
            shutil.rmtree(path)
    except OSError:
        logger.debug("Could not remove %s", path, exc_info=True)
