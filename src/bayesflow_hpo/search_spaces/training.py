"""Training hyperparameter search space."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from bayesflow_hpo.search_spaces.base import (
    _UNSET,
    BaseSearchSpace,
    DerivedDimension,
    Dimension,
    FloatDimension,
    IntDimension,
)

#: Name given to the sampled learning-rate coordinate when it is
#: reparametrized, so that it cannot be confused with the ``initial_lr`` the
#: optimizer receives.
REFERENCE_LR_NAME = "lr_ref"


def _bounds(dimension: IntDimension, role: str) -> tuple[int, int]:
    """Return the smallest and largest value an ``IntDimension`` can take."""
    if not isinstance(dimension, IntDimension):
        raise TypeError(
            f"{role} must be an IntDimension when simulation_budget is set, "
            f"got {type(dimension).__name__}."
        )
    if dimension.constant is not _UNSET:
        value = int(dimension.constant)
        return value, value
    if dimension.low is None or dimension.high is None:
        # pragma: no cover - IntDimension.__post_init__ rejects this
        raise ValueError(f"IntDimension({dimension.name!r}) has no bounds.")
    return int(dimension.low), int(dimension.high)


@dataclass
class TrainingSpace(BaseSearchSpace):
    """Search space for optimizer/training knobs.

    Batch size and peak learning rate are jointly tunable because their useful
    relationship depends on the workload and compute budget (Smith et al.,
    2018; Shallue et al., 2019).

    Two optional couplings turn that joint space into the coordinates the
    relationship is actually expressed in:

    ``lr_reference_batch_size``
        Reparametrizes the learning rate as ``initial_lr = lr_ref *
        batch_size / lr_reference_batch_size``, so ``lr_ref`` is the peak
        learning rate *at the reference batch size* and the sampled
        ``(lr_ref, batch_size)`` rectangle follows the linear scaling
        relationship of batch size to learning rate reported by Smith et al.
        (2018).  Searching the coupled coordinates directly matters for
        samplers that model parameters marginally, such as Optuna's TPE.

    ``simulation_budget``
        Derives ``num_batches = simulation_budget // (batch_size * epochs)``,
        which keeps every trial simulation-matched while ``batch_size`` and
        ``epochs`` vary.  In online SBI the batch size is simultaneously the
        data-volume knob (``simulations = batch_size * epochs *
        num_batches``), so at most two of the three can be fixed; fixing the
        simulation budget is the one proportional to compute spent.

    Warmup length is deliberately *not* a dimension here.  It is configured
    once on the objective (``lr_warmup_fraction`` / ``lr_warmup_epochs``),
    because adding a third correlated schedule axis to ``{batch_size,
    learning rate}`` is what Shallue et al. (2019, Sec. 5.1) report as having
    made their own tuning unreliable, at a far larger budget than a typical
    HPO run here.

    Parameters
    ----------
    initial_lr
        Peak learning rate.  When ``lr_reference_batch_size`` is set, this
        dimension is sampled under the name ``lr_ref`` and the effective
        ``initial_lr`` is derived from it.
    batch_size
        Online simulation batch size.
    epochs
        Optional epoch dimension.  Required when ``simulation_budget`` is set,
        since the derived ``num_batches`` depends on it.  When left ``None``,
        the objective's ``epochs`` setting is used.
    lr_reference_batch_size
        Batch size at which ``lr_ref`` is interpreted.  ``None`` (default)
        disables the reparametrization and samples ``initial_lr`` directly.
    simulation_budget
        Total online simulations per trial.  ``None`` (default) leaves
        ``num_batches`` to the objective's setting.

    Notes
    -----
    Derived values do not appear in ``trial.params``, because Optuna records
    only what it sampled.  The objective stores them under the
    ``derived_params`` trial user attribute, so ``best_config()`` and the
    results tables still report the configuration a trial trained with.

    References
    ----------
    https://doi.org/10.48550/arXiv.1711.00489
    https://www.jmlr.org/papers/v20/18-789.html
    """

    initial_lr: FloatDimension = field(
        default_factory=lambda: FloatDimension(
            "initial_lr", low=1e-4, high=1e-2, log=True
        )
    )
    batch_size: IntDimension = field(
        default_factory=lambda: IntDimension(
            "batch_size", low=32, high=256, step=32
        )
    )
    epochs: IntDimension | None = None
    lr_reference_batch_size: int | None = None
    simulation_budget: int | None = None

    def __post_init__(self) -> None:
        if self.lr_reference_batch_size is not None:
            if self.lr_reference_batch_size < 1:
                raise ValueError(
                    "lr_reference_batch_size must be >= 1, got "
                    f"{self.lr_reference_batch_size}."
                )
            if self.initial_lr.name == REFERENCE_LR_NAME:
                raise ValueError(
                    "The learning-rate dimension cannot be named "
                    f"{REFERENCE_LR_NAME!r}: that name is assigned to it "
                    "automatically when lr_reference_batch_size is set."
                )

        if self.simulation_budget is None:
            return
        if self.epochs is None:
            raise ValueError(
                "simulation_budget requires an 'epochs' dimension on "
                "TrainingSpace: num_batches is derived from both, and "
                "the objective's epochs setting is not visible here."
            )
        min_batch, max_batch = _bounds(self.batch_size, "batch_size")
        min_epochs, max_epochs = _bounds(self.epochs, "epochs")
        if min_batch < 1 or min_epochs < 1:
            # Otherwise the floor division divides by zero at sample time,
            # inside `search_space.sample()` -- which aborts the whole study
            # instead of rejecting one trial.
            raise ValueError(
                "simulation_budget requires batch_size and epochs to be "
                f">= 1, got batch_size >= {min_batch} and "
                f"epochs >= {min_epochs}."
            )
        if self.simulation_budget < max_batch * max_epochs:
            raise ValueError(
                f"simulation_budget={self.simulation_budget} is too small: "
                "the largest batch_size x epochs combination in this space "
                f"needs {max_batch * max_epochs} simulations for a single "
                "batch per epoch."
            )

    @property
    def dimensions(self) -> list[Dimension]:
        """Return the declared dimensions plus any derived couplings.

        The couplings are resolved here rather than stored on the instance so
        that the dimension objects a caller passed in are never mutated: one
        ``FloatDimension`` shared between two spaces, or a space rebuilt with
        ``dataclasses.replace``, would otherwise inherit a rename it never
        asked for and then sample the wrong coordinate.
        """
        declared = super().dimensions
        if self.lr_reference_batch_size is None and self.simulation_budget is None:
            return declared

        resolved: list[Dimension] = []
        for dimension in declared:
            if dimension is self.initial_lr and self.lr_reference_batch_size:
                resolved.append(replace(dimension, name=REFERENCE_LR_NAME))
            else:
                resolved.append(dimension)

        if self.lr_reference_batch_size is not None:
            reference_batch = float(self.lr_reference_batch_size)
            resolved.append(
                DerivedDimension(
                    "initial_lr",
                    lambda p: float(p[REFERENCE_LR_NAME])
                    * float(p["batch_size"])
                    / reference_batch,
                )
            )
        if self.simulation_budget is not None:
            budget = int(self.simulation_budget)
            resolved.append(
                DerivedDimension(
                    "num_batches",
                    lambda p: budget // (int(p["batch_size"]) * int(p["epochs"])),
                )
            )
        return resolved
