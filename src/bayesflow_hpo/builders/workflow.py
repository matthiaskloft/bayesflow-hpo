"""Approximator construction helpers.

Builds a ``ContinuousApproximator`` for a single HPO trial from the
sampled hyperparameters.  Optimizer helpers provide a finite-horizon cosine
schedule and a horizon-free inverse-square-root schedule with warmup.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import bayesflow as bf
import keras
from keras import ops

from bayesflow_hpo.search_spaces.composite import CompositeSearchSpace

logger = logging.getLogger(__name__)


@keras.saving.register_keras_serializable(package="bayesflow_hpo")
class InverseSqrtDecay(keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by inverse-square-root decay.

    The peak learning rate is reached at ``warmup_steps``; subsequent values
    are proportional to the inverse square root of the optimizer step.  This
    is the horizon-free schedule described by Vaswani et al. (2017, Sec. 5.3).

    See Also
    --------
    https://doi.org/10.48550/arXiv.1706.03762
    https://keras.io/api/optimizers/learning_rate_schedules/learning_rate_schedule/
    """

    def __init__(self, peak_learning_rate: float, warmup_steps: int = 1):
        if warmup_steps < 1:
            raise ValueError(f"warmup_steps must be >= 1, got {warmup_steps}.")
        self.peak_learning_rate = float(peak_learning_rate)
        self.warmup_steps = int(warmup_steps)

    def __call__(self, step: Any) -> Any:
        dtype = keras.backend.floatx()
        step_number = ops.cast(step, dtype) + 1.0
        warmup_steps = ops.cast(self.warmup_steps, dtype)
        peak = ops.cast(self.peak_learning_rate, dtype)
        warmup_lr = peak * step_number / warmup_steps
        decay_lr = peak * ops.sqrt(warmup_steps / step_number)
        return ops.where(step_number <= warmup_steps, warmup_lr, decay_lr)

    def get_config(self) -> dict[str, Any]:
        """Return a serializable schedule configuration."""
        return {
            "peak_learning_rate": self.peak_learning_rate,
            "warmup_steps": self.warmup_steps,
        }


def _make_cosine_decay_optimizer(
    initial_lr: float,
    decay_steps: int,
    warmup_steps: int = 0,
) -> keras.optimizers.Optimizer:
    """Create an Adam optimizer with optional warmup and cosine decay.

    Parameters
    ----------
    initial_lr
        Peak learning rate.
    decay_steps
        Total optimizer-step budget, including warmup.
    warmup_steps
        Linear-warmup steps before cosine decay. Keras implements this with
        ``warmup_target`` and ``warmup_steps`` as documented by its
        :class:`~keras.optimizers.schedules.CosineDecay` API.

    See Also
    --------
    https://keras.io/api/optimizers/learning_rate_schedules/cosine_decay/
    https://doi.org/10.48550/arXiv.1706.02677
    """
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}.")
    if warmup_steps >= decay_steps:
        raise ValueError(
            "warmup_steps must be smaller than the total training steps, "
            f"got {warmup_steps} >= {decay_steps}."
        )
    schedule_kwargs: dict[str, Any] = {}
    schedule_initial_lr = initial_lr
    cosine_steps = decay_steps
    if warmup_steps:
        schedule_initial_lr = 0.0
        cosine_steps -= warmup_steps
        schedule_kwargs = {
            "warmup_target": initial_lr,
            "warmup_steps": warmup_steps,
        }
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=schedule_initial_lr,
        decay_steps=cosine_steps,
        **schedule_kwargs,
    )
    return keras.optimizers.Adam(learning_rate=lr_schedule)


def _make_inverse_sqrt_optimizer(
    initial_lr: float,
    warmup_steps: int,
) -> keras.optimizers.Optimizer:
    """Create an Adam optimizer with horizon-free inverse-sqrt decay."""
    return keras.optimizers.Adam(
        learning_rate=InverseSqrtDecay(initial_lr, warmup_steps),
    )


def _compile_for_compat(candidate: Any, optimizer: Any) -> None:
    """Try common compile signatures without raising on incompatible variants.

    BayesFlow's ``compile()`` signature varies across versions and model
    types.  This helper tries the optimizer-accepting variants first to
    ensure the optimizer is actually applied, then falls back to no-arg
    compile only when the model doesn't accept an optimizer at all.
    """
    compile_fn = getattr(candidate, "compile", None)
    if compile_fn is None:
        return

    try:
        compile_fn(optimizer=optimizer)
        return
    except TypeError:
        pass

    try:
        compile_fn(optimizer)
        return
    except TypeError:
        pass

    try:
        compile_fn()
    except TypeError:
        # All three signatures failed — the approximator is left uncompiled.
        # This is not necessarily an error (some custom approximators compile
        # themselves), but log a warning so the caller can investigate if
        # training fails later.
        import logging

        logging.getLogger(__name__).warning(
            "No compile() signature succeeded for %s — model may be uncompiled.",
            type(candidate).__name__,
        )
        return


def build_continuous_approximator(
    hparams: dict[str, Any],
    adapter: bf.adapters.Adapter,
    search_space: CompositeSearchSpace,
    checkpoint_dir: str | Path | None = None,
) -> Any:
    """Build an uncompiled ``ContinuousApproximator`` from search-space hparams.

    This is the default used by ``optimize()`` when ``build_approximator_fn``
    is ``None``.  It:

    1. Constructs inference and summary networks from the search space.
    2. Wraps them in a ``ContinuousApproximator``.
    3. Optionally loads pre-trained weights from *checkpoint_dir*.

    The returned approximator is **uncompiled** — the objective handles
    compilation separately.

    Note: this function has a broader signature than ``BuildApproximatorFn``
    because it needs the adapter and search space.  Inside ``optimize()``
    these are captured internally via a closure.

    Parameters
    ----------
    hparams
        Sampled hyperparameters from the search space.
    adapter
        BayesFlow adapter for data preprocessing.
    search_space
        Composite search space defining the tunable dimensions.
    checkpoint_dir
        Optional directory containing ``weights.weights.h5``.  When
        provided, the approximator is warm-started from these weights.
        Use ``CheckpointPool.best_checkpoint_dir`` to load the best
        trial's weights.  If loading fails (file missing, incompatible
        shapes), a warning is logged and the model continues with
        fresh weights.

    Returns
    -------
    ContinuousApproximator
        Uncompiled approximator ready for ``compile()`` + ``fit()``.
    """
    inference_net = search_space.inference_space.build(hparams)

    summary_net = None
    if search_space.summary_space is not None:
        summary_net = search_space.summary_space.build(hparams)

    approximator = bf.ContinuousApproximator(
        inference_network=inference_net,
        summary_network=summary_net,
        adapter=adapter,
    )

    if checkpoint_dir is not None:
        checkpoint_path = Path(checkpoint_dir) / "weights.weights.h5"
        try:
            approximator.load_weights(str(checkpoint_path))
            logger.info("Loaded checkpoint weights from %s", checkpoint_path)
        except Exception as exc:
            logger.warning(
                "Failed to load checkpoint from %s: %s. "
                "Continuing with fresh weights.",
                checkpoint_path, exc,
            )

    return approximator
