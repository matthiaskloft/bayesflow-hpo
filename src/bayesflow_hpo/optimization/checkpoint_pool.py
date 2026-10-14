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
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeAlias

logger = logging.getLogger(__name__)

# Default pool size.
DEFAULT_POOL_SIZE = 5

#: Default size of the pruned-trial pool.  Zero, i.e. off: retaining pruned
#: weights is opt-in.
DEFAULT_PRUNED_POOL_SIZE = 0

_WEIGHTS_FILE = "weights.weights.h5"
_METADATA_FILE = "checkpoint.json"

#: Sidecar keys the pool writes itself.  Caller-supplied metadata may not
#: use them: a domain package must never be able to make a checkpoint lie
#: about which trial, state or rung it holds.
RESERVED_METADATA_KEYS: frozenset[str] = frozenset({
    "trial_number",
    "state",
    "objective_value",
    "pruned_at_step",
    "weights_file",
    "note",
})

#: Extra ``checkpoint.json`` content: a mapping, or a callable
#: ``(approximator, trial_number) -> mapping`` evaluated at save time.
ExtraMetadata: TypeAlias = (
    Mapping[str, Any] | Callable[[Any, int], Mapping[str, Any]]
)


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
    ``extra_metadata`` adds caller-supplied keys to that sidecar, for facts
    a domain package needs to rebuild the approximator correctly that
    ``trial.params`` does not carry (e.g. adapter moments computed at
    training time).

    Scored trials are ranked by the single ``objective_value`` the caller
    passes to :meth:`maybe_save`, lower is better.  Under
    :class:`~bayesflow_hpo.optimization.objective.GenericObjective` that
    value is :func:`~bayesflow_hpo.objectives.mean_objective_score`, the
    mean of the quality objectives with the cost score dropped.  For a
    multi-objective study this is *not* a Pareto ranking: the pool can
    keep a dominated trial and evict a Pareto-optimal one.  Select from
    the Pareto front with
    :func:`~bayesflow_hpo.results.extraction.get_pareto_trials` and treat
    the pool as a cache of good weights, not as the front.

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
        and replaces a uniformly chosen incumbent.  That is Algorithm R of
        Vitter (1985), Alg. R and Def. 1, Sec. 2, p. 39, which needs no
        advance knowledge of the population size -- a study's pruned count
        is unknown until it ends.  Not top-*k*, because
        pruned trials stop at different rungs and their objective values
        are not comparable across rungs; a uniform sample instead gives
        coverage of the whole quality range, which is what the diagnostic
        use cases need.
    seed
        Seed for the retention sampler.  ``None`` (default) uses an
        unseeded :class:`random.Random`.
    extra_metadata
        Pool-level extra sidecar content, used by :meth:`maybe_save` and
        :meth:`save_pruned` whenever a call passes none of its own.  A
        mapping, or a callable ``(approximator, trial_number) -> mapping``
        evaluated at save time.  Keys must be strings outside
        :data:`RESERVED_METADATA_KEYS` (nested mappings must have string
        keys too), values strict-JSON-serializable:
        ``NaN``/``inf`` are rejected, and so are NumPy scalars other than
        ``np.float64`` -- convert with ``float()`` / ``.tolist()``.  This is
        how a hooks package supplies metadata to checkpoints written by
        ``optimize()`` without subclassing the pool.  Default ``None``:
        the sidecar is unchanged.

    Raises
    ------
    ValueError
        If a mapping ``extra_metadata`` uses a reserved key.
    TypeError
        If a mapping ``extra_metadata`` has a non-string key or a value
        that is not strict-JSON-serializable.

    Notes
    -----
    Invalid metadata raises instead of being dropped, so no checkpoint is
    written without it.  Inside ``optimize()`` a callable that raises (or
    returns invalid metadata) for a *completed* trial therefore propagates
    out of ``study.optimize()``: that trial is recorded as FAIL and the
    run stops.  For a *pruned* trial the error is logged at WARNING and
    the trial is not retained; the study continues.
    """

    def __init__(
        self,
        pool_dir: str | Path = "checkpoints",
        pool_size: int = DEFAULT_POOL_SIZE,
        pruned_pool_size: int = DEFAULT_PRUNED_POOL_SIZE,
        seed: int | None = None,
        extra_metadata: ExtraMetadata | None = None,
    ):
        if extra_metadata is not None and not callable(extra_metadata):
            # Fail at construction, not at the first trial that finishes.
            _validate_extra_metadata(extra_metadata)
        self.pool_dir = Path(pool_dir)
        self.pool_size = pool_size
        self.pruned_pool_size = pruned_pool_size
        self.extra_metadata = extra_metadata
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
        extra_metadata: ExtraMetadata | None = None,
    ) -> bool:
        """Save *approximator* weights if they're in the current top-k.

        Parameters
        ----------
        trial_number
            Optuna trial number.
        objective_value
            Scalar ranking score, lower is better.  See the class docstring
            for what the objective passes here.
        approximator
            The trained approximator whose weights are saved.
        extra_metadata
            Extra sidecar content for this checkpoint; see the class
            parameter of the same name.  ``None`` (default) falls back to
            the pool-level :attr:`extra_metadata`.  A callable is evaluated
            only when the trial is actually saved.

        Returns
        -------
        bool
            ``True`` when the checkpoint was saved (or the pool was
            updated), ``False`` otherwise.

        Raises
        ------
        ValueError
            If the resolved extra metadata uses a reserved key.
        TypeError
            If it is not a mapping with string keys and JSON-serializable
            values.
        """
        if (
            len(self._entries) >= self.pool_size
            and objective_value >= self._entries[-1][0]
        ):
            return False

        extra = self._resolve_extra_metadata(
            extra_metadata, approximator, trial_number,
        )
        dest = self.pool_dir / f"trial_{trial_number:04d}"
        if not _write_checkpoint(
            dest,
            approximator,
            {
                **extra,
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
        extra_metadata: ExtraMetadata | None = None,
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
        extra_metadata
            Extra sidecar content for this checkpoint; see the class
            parameter of the same name.  ``None`` (default) falls back to
            the pool-level :attr:`extra_metadata`.  A callable is evaluated
            only when the trial is actually retained.

        Returns
        -------
        bool
            ``True`` when the checkpoint was written and retained.

        Raises
        ------
        ValueError
            If the resolved extra metadata uses a reserved key.
        TypeError
            If it is not a mapping with string keys and JSON-serializable
            values.
        """
        if self.pruned_pool_size <= 0:
            return False

        dest = self.pruned_pool_dir / f"trial_{trial_number:04d}"
        existing_slot = next(
            (
                i
                for i, (num, _) in enumerate(self._pruned_entries)
                if num == trial_number
            ),
            None,
        )

        n_offered = self._pruned_seen + 1

        evict_slot: int | None = None
        if existing_slot is not None:
            # A trial already in the pool is one member of the population,
            # not a second one: re-offering it must refresh its checkpoint
            # in place rather than appending a duplicate entry (which a
            # later eviction could then delete out from under) or drawing
            # again against `_pruned_seen`.
            n_offered = self._pruned_seen
            evict_slot = existing_slot
        elif len(self._pruned_entries) >= self.pruned_pool_size:
            # Keep the n-th offer with probability k/n, replacing a
            # uniformly chosen incumbent -- Vitter (1985), Alg. R, Sec. 2,
            # p. 39. This is what keeps every pruned trial of the study
            # equally likely to be in the final pool, without knowing the
            # population size up front. Verified empirically by
            # ``test_retention_is_uniform_over_the_population``.
            if self._rng.random() >= self.pruned_pool_size / n_offered:
                self._pruned_seen = n_offered
                return False
            evict_slot = self._rng.randrange(self.pruned_pool_size)

        # Resolved after the retention draw, so a rejected offer never runs
        # the callable; and before the write, so a bad value raises without
        # counting the offer (`_pruned_seen` is untouched until success).
        # The RNG has advanced by then, as on a failed write: uniformity is
        # unaffected, only the seeded sequence shifts.
        extra = self._resolve_extra_metadata(
            extra_metadata, approximator, trial_number,
        )
        if not _write_checkpoint(
            dest,
            approximator,
            {
                **extra,
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
            if evicted_path != dest:
                _safe_rmtree(evicted_path)
                logger.debug(
                    "Evicted pruned trial %d from pruned checkpoint pool",
                    evicted_num,
                )
            self._pruned_entries[evict_slot] = (trial_number, dest)

        return True

    def _resolve_extra_metadata(
        self,
        extra_metadata: ExtraMetadata | None,
        approximator: Any,
        trial_number: int,
    ) -> dict[str, Any]:
        """Return the validated extra sidecar content for one save."""
        source = (
            self.extra_metadata if extra_metadata is None else extra_metadata
        )
        if source is None:
            return {}
        if callable(source):
            source = source(approximator, trial_number)
        return _validate_extra_metadata(source)

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


def _validate_extra_metadata(extra: Any) -> dict[str, Any]:
    """Check caller-supplied sidecar content and return it as a dict.

    Raises rather than dropping keys: the metadata exists because the
    checkpoint cannot be rebuilt correctly without it, so a checkpoint
    silently missing it is worse than a loud failure.
    """
    if not isinstance(extra, Mapping):
        raise TypeError(
            "extra_metadata must be a mapping or a callable returning one, "
            f"got {type(extra).__name__}"
        )
    _check_string_keys(extra, "extra_metadata")
    reserved = sorted(RESERVED_METADATA_KEYS.intersection(extra))
    if reserved:
        raise ValueError(
            f"extra_metadata may not set reserved checkpoint keys {reserved}"
        )
    result = dict(extra)
    try:
        # allow_nan=False: NaN/inf would be written as bare `NaN`, which
        # strict JSON parsers reject.
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"extra_metadata must be JSON-serializable: {exc}"
        ) from exc
    return result


def _check_string_keys(value: Any, path: str) -> None:
    """Reject non-string keys at any nesting depth.

    ``json.dumps`` coerces ``int``/``float``/``bool``/``None`` keys to
    strings, so ``{1: "a", "1": "b"}`` would serialize to ``{"1": "b"}``
    and silently lose an entry.  Checked recursively because nested dicts
    are the natural shape for adapter metadata.
    """
    if isinstance(value, Mapping):
        bad_keys = [k for k in value if not isinstance(k, str)]
        if bad_keys:
            raise TypeError(
                f"{path} keys must be strings, got {bad_keys!r}"
            )
        for key, item in value.items():
            _check_string_keys(item, f"{path}[{key!r}]")
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _check_string_keys(item, f"{path}[{i}]")


def _write_checkpoint(
    dest: Path,
    approximator: Any,
    metadata: dict[str, Any],
    trial_number: int,
) -> bool:
    """Write weights plus the ``checkpoint.json`` sidecar to *dest*.

    Staged through a temporary directory and published by a rename, so
    the pool never exposes a half-written checkpoint.  Two failure modes
    make that worth the extra move: a saver that writes bytes and then
    raises would otherwise leave an orphan directory that no eviction
    can reach, because the caller never records it; and weights whose
    sidecar failed to write would be retained without the rung that
    makes a pruned checkpoint interpretable at all.
    """
    staging = dest.parent / f".{dest.name}.tmp"
    _safe_rmtree(staging)
    try:
        staging.mkdir(parents=True, exist_ok=True)
        approximator.save_weights(str(staging / _WEIGHTS_FILE))
        (staging / _METADATA_FILE).write_text(
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
    except Exception:
        logger.warning(
            "Failed to save checkpoint for trial %d", trial_number,
            exc_info=True,
        )
        _safe_rmtree(staging)
        return False

    try:
        # Only reached once the staged copy is complete, so the existing
        # checkpoint is never dropped for a write that then fails.
        _safe_rmtree(dest)
        staging.replace(dest)
    except OSError:
        logger.warning(
            "Failed to publish checkpoint for trial %d", trial_number,
            exc_info=True,
        )
        _safe_rmtree(staging)
        return False
    return True


def _safe_rmtree(path: Path) -> None:
    """Remove a directory tree, suppressing OS errors (e.g. file locks on Windows)."""
    try:
        if path.exists():
            shutil.rmtree(path)
    except OSError:
        logger.debug("Could not remove %s", path, exc_info=True)
