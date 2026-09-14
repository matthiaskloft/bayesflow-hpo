"""Training hyperparameter search space."""

from __future__ import annotations

from dataclasses import dataclass, field

from bayesflow_hpo.search_spaces.base import (
    _UNSET,
    BaseSearchSpace,
    DerivedDimension,
    FloatDimension,
    IntDimension,
)


def _upper_bound(dimension: IntDimension) -> int:
    """Largest value an ``IntDimension`` can take."""
    if dimension.constant is not _UNSET:
        return int(dimension.constant)
    if dimension.high is None:  # pragma: no cover - IntDimension validates this
        raise ValueError(f"IntDimension({dimension.name!r}) has no upper bound.")
    return int(dimension.high)


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
        relationship ``B`` proportional to ``epsilon`` reported by Smith et
        al. (2018).  Searching the coupled coordinates directly matters for
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
        dimension is sampled as ``lr_ref`` and the effective ``initial_lr``
        is derived from it.
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
    scaled_lr: DerivedDimension | None = field(default=None, init=False)
    num_batches: DerivedDimension | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.lr_reference_batch_size is not None:
            if self.lr_reference_batch_size < 1:
                raise ValueError(
                    "lr_reference_batch_size must be >= 1, got "
                    f"{self.lr_reference_batch_size}."
                )
            # The sampled coordinate is no longer the learning rate that the
            # optimizer receives, so it must not be recorded under that name:
            # `initial_lr` in `trial.params` would then disagree with the
            # `initial_lr` in `hparams`.
            if self.initial_lr.name == "initial_lr":
                self.initial_lr.name = "lr_ref"
            ref_name = self.initial_lr.name
            reference_batch = float(self.lr_reference_batch_size)
            self.scaled_lr = DerivedDimension(
                "initial_lr",
                lambda p: float(p[ref_name])
                * float(p["batch_size"])
                / reference_batch,
            )

        if self.simulation_budget is not None:
            if self.epochs is None:
                raise ValueError(
                    "simulation_budget requires an 'epochs' dimension on "
                    "TrainingSpace: num_batches is derived from both, and "
                    "the objective's epochs setting is not visible here."
                )
            max_simulations_per_batch = _upper_bound(self.batch_size) * _upper_bound(
                self.epochs
            )
            if self.simulation_budget < max_simulations_per_batch:
                raise ValueError(
                    f"simulation_budget={self.simulation_budget} is too small "
                    f"for this space: the largest batch_size x epochs "
                    f"combination needs {max_simulations_per_batch} "
                    "simulations for a single batch per epoch."
                )
            budget = int(self.simulation_budget)
            self.num_batches = DerivedDimension(
                "num_batches",
                lambda p: budget // (int(p["batch_size"]) * int(p["epochs"])),
            )
