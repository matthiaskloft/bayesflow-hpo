"""Approximator construction helpers.

Builds a ``ContinuousApproximator`` for a single HPO trial from the
sampled hyperparameters.  The optimizer helper uses **Adam with
CosineDecay**, which decays the learning rate from ``initial_lr`` to
near-zero over the full training budget (``epochs * batches_per_epoch``
steps).
"""

from __future__ import annotations

from typing import Any

import bayesflow as bf
import keras

from bayesflow_hpo.search_spaces.composite import CompositeSearchSpace


def _make_cosine_decay_optimizer(
    initial_lr: float,
    decay_steps: int,
) -> keras.optimizers.Optimizer:
    """Create an Adam optimizer with CosineDecay schedule.

    Parameters
    ----------
    initial_lr
        Peak learning rate.
    decay_steps
        Total steps over which to decay (``epochs * batches_per_epoch``).
    """
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_lr,
        decay_steps=max(1, decay_steps),
    )
    return keras.optimizers.Adam(learning_rate=lr_schedule)


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
) -> Any:
    """Build an uncompiled ``ContinuousApproximator`` from search-space hparams.

    This is the default used by ``optimize()`` when ``build_approximator_fn``
    is ``None``.  It:

    1. Constructs inference and summary networks from the search space.
    2. Wraps them in a ``ContinuousApproximator``.

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
    return approximator
