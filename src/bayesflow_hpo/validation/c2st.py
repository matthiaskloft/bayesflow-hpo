"""Classifier two-sample tests for multivariate posterior validation.

Provides two standalone diagnostic functions and a ValidateFn factory:

- **L-C2ST** (Linhart et al., 2023): reference-free local posterior
  diagnostic using joint samples from the simulator. No true posterior
  samples required.
- **Global C2ST** (López-Paz & Oquab, 2017): standard classifier
  two-sample test requiring samples from both approximate and reference
  posteriors.
- **make_lc2st_validate_fn()**: factory returning a ``ValidateFn``
  compatible with ``optimize(validate_fn=...)``. Runs standard
  per-parameter metrics and L-C2ST from a single inference pass.

All functions require scikit-learn as an optional dependency. Install
via ``pip install bayesflow-hpo[sklearn]``.

References
----------
Linhart, J., Gramfort, A., & Rodrigues, P. L. C. (2023). L-C2ST: Local
    diagnostics for posterior approximations in simulation-based inference.
    In *Advances in Neural Information Processing Systems 36*.
    https://doi.org/10.48550/arXiv.2306.03580

López-Paz, D., & Oquab, M. (2017). Revisiting classifier two-sample
    tests. In *Proceedings of the 5th International Conference on Learning
    Representations (ICLR 2017)*. https://arxiv.org/abs/1610.06545
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import norm

from bayesflow_hpo.validation.data import ValidationDataset
from bayesflow_hpo.validation.inference import DEFAULT_MAX_SAMPLES_PER_CALL
from bayesflow_hpo.validation.pipeline import run_validation_pipeline
from bayesflow_hpo.validation.registry import (
    REQUIRES_SCALAR_PARAMETERS,
    JointMetricFn,
    JointMetricInputs,
    register_joint_metric,
)

# ---------------------------------------------------------------------------
# Lazy sklearn import
# ---------------------------------------------------------------------------


def _require_sklearn() -> tuple[type, type]:
    """Lazily import MLPClassifier and KFold from scikit-learn.

    Returns
    -------
    tuple
        ``(MLPClassifier, KFold)`` classes.

    Raises
    ------
    ImportError
        If scikit-learn is not installed, with install instructions.
    """
    try:
        from sklearn.model_selection import KFold
        from sklearn.neural_network import MLPClassifier
    except ImportError:
        raise ImportError(
            "C2ST metrics require scikit-learn. "
            "Install it with: pip install bayesflow-hpo[sklearn]"
        ) from None
    return MLPClassifier, KFold


# ---------------------------------------------------------------------------
# Default classifier config (SBIBM reference implementation)
# ---------------------------------------------------------------------------


def _default_clf_kwargs(ndim: int) -> dict[str, Any]:
    """Return SBIBM-style MLP config for L-C2ST.

    Matches the ``sbibm_clf_kwargs`` from the reference implementation
    (JuliaLinhart/lc2st).

    Parameters
    ----------
    ndim
        Dimensionality of the input features (determines hidden layer
        width as ``10 * ndim``).

    Returns
    -------
    dict[str, Any]
        Keyword arguments for ``sklearn.neural_network.MLPClassifier``.
    """
    return {
        "hidden_layer_sizes": (10 * ndim, 10 * ndim),
        "activation": "relu",
        "solver": "adam",
        "max_iter": 25000,
        "alpha": 0,
        "early_stopping": True,
        "n_iter_no_change": 50,
    }


# ---------------------------------------------------------------------------
# L-C2ST (Linhart et al., 2023)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LC2STResult:
    """Result container for :func:`lc2st`.

    Attributes
    ----------
    statistic
        Mean single-class MSE_0 across observations (Theorem 3.1).
        Values near 0 indicate a well-calibrated posterior; larger values
        indicate misspecification.
    p_value
        Permutation p-value (fraction of null statistics >= observed).
        ``None`` when ``n_null_trials=0``.
    null_statistics
        Array of shape ``(n_null_trials,)`` with null distribution
        statistics, or empty array if skipped.
    per_observation_stats
        Array of shape ``(n_eval,)`` with per-observation MSE_0 values
        from cross-validation.
    """

    statistic: float
    p_value: float | None
    null_statistics: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    per_observation_stats: np.ndarray = field(
        default_factory=lambda: np.array([])
    )


def _run_lc2st_cv(
    feats_joint: np.ndarray,
    feats_approx: np.ndarray,
    labels: np.ndarray,
    n_folds: int,
    clf_kwargs: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    """Run K-fold CV and return class-1 probabilities for class-0 samples.

    For each fold, trains a binary classifier on joint (class-1) and
    approximate (class-0) features, then predicts class-1 probability
    on the held-out class-0 validation samples. Under the null (well-
    calibrated posterior), these probabilities should be near 0.5.

    Parameters
    ----------
    feats_joint
        Class-1 features, shape ``(n_sims, n_features)``.
    feats_approx
        Class-0 features, shape ``(n_sims, n_features)``.
    labels
        Binary labels, shape ``(2 * n_sims,)``.
    n_folds
        Number of cross-validation folds.
    clf_kwargs
        Keyword arguments for ``MLPClassifier``.
    rng
        Random number generator for fold splitting.

    Returns
    -------
    np.ndarray
        Class-1 predicted probabilities for class-0 (approximate
        posterior) samples on their held-out validation folds,
        shape ``(n_sims,)``.
    """
    mlp_cls, kfold_cls = _require_sklearn()

    n_sims = len(feats_joint)
    feats_all = np.concatenate([feats_joint, feats_approx], axis=0)

    # Pre-allocate predicted probs for class-0 observations
    probs_class0 = np.zeros(n_sims)

    kf = kfold_cls(
        n_splits=n_folds,
        shuffle=True,
        random_state=int(rng.integers(2**31)),
    )

    # Fold indices operate on [0, n_sims) — same fold assignment for
    # both classes to keep training balanced
    for fold_idx, (train_idx, val_idx) in enumerate(
        kf.split(np.arange(n_sims))
    ):
        # Training data: fold-train from both classes
        train_rows = np.concatenate(
            [train_idx, train_idx + n_sims]
        )
        x_train = feats_all[train_rows]
        y_train = labels[train_rows]

        # Train classifier
        clf = mlp_cls(**clf_kwargs, random_state=fold_idx)
        clf.fit(x_train, y_train)

        # Predict on fold-val class-0 samples only
        x_val = feats_approx[val_idx]
        probs_class0[val_idx] = clf.predict_proba(x_val)[:, 1]

    return probs_class0


def lc2st(
    posterior_samples: np.ndarray,
    true_params: np.ndarray,
    observations: np.ndarray,
    *,
    n_folds: int = 5,
    n_null_trials: int = 0,
    clf_kwargs: dict[str, Any] | None = None,
    seed: int = 42,
) -> LC2STResult:
    """Local Classifier Two-Sample Test (Linhart et al., 2023).

    # Linhart et al. (2023), Algorithms 1-2

    Implements Algorithms 1-2 from the paper. Tests whether the
    approximate posterior ``q(theta|x)`` matches the true posterior
    ``p(theta|x)`` using joint samples, without requiring true
    posterior samples.

    Parameters
    ----------
    posterior_samples
        Posterior draws, shape ``(n_sims, n_samples, n_params)``.
        One sample per simulation is used for the classifier
        (index 0).
    true_params
        Ground-truth parameters, shape ``(n_sims, n_params)``.
    observations
        Observed data, shape ``(n_sims, n_obs)`` or
        ``(n_sims, ...)``. Flattened to 2D if needed.
    n_folds
        Number of cross-validation folds (default 5).
    n_null_trials
        Number of label-permutation trials for the null distribution.
        Set to 0 (default) to skip the permutation test.
    clf_kwargs
        Override keyword arguments for ``MLPClassifier``. If ``None``,
        uses SBIBM defaults via :func:`_default_clf_kwargs`.
    seed
        Random seed for reproducibility.

    Returns
    -------
    LC2STResult
        Result with statistic, optional p-value, null distribution,
        and per-observation statistics.

    Raises
    ------
    ValueError
        If input shapes are inconsistent or ``n_sims < n_folds``.
    ImportError
        If scikit-learn is not installed.
    """
    _require_sklearn()
    rng = np.random.default_rng(seed)

    # --- Input validation ---
    posterior_samples = np.asarray(posterior_samples)
    true_params = np.asarray(true_params)
    observations = np.asarray(observations)

    if posterior_samples.ndim != 3:
        raise ValueError(
            "posterior_samples must be 3D "
            f"(n_sims, n_samples, n_params), "
            f"got shape {posterior_samples.shape}"
        )
    n_sims = posterior_samples.shape[0]

    if true_params.ndim == 1:
        true_params = true_params[:, None]
    if true_params.shape[0] != n_sims:
        raise ValueError(
            f"true_params has {true_params.shape[0]} sims, "
            f"expected {n_sims}"
        )
    if posterior_samples.shape[2] != true_params.shape[1]:
        raise ValueError(
            f"Parameter dimension mismatch: posterior_samples has "
            f"{posterior_samples.shape[2]}, true_params has "
            f"{true_params.shape[1]}"
        )

    # Flatten observations to 2D
    if observations.ndim == 1:
        observations = observations[:, None]
    elif observations.ndim > 2:
        observations = observations.reshape(n_sims, -1)
    if observations.shape[0] != n_sims:
        raise ValueError(
            f"observations has {observations.shape[0]} sims, "
            f"expected {n_sims}"
        )

    if n_folds < 2:
        raise ValueError(
            f"n_folds must be >= 2 for cross-validation, got {n_folds}"
        )
    if n_sims < n_folds:
        raise ValueError(
            f"n_sims ({n_sims}) must be >= n_folds ({n_folds})"
        )

    # --- Build training data (Algorithm 1, Step 3) ---
    # Class 1 (joint): concat(true_params, observations)
    feats_joint = np.concatenate(
        [true_params, observations], axis=1
    )
    # Class 0 (approximate): concat(posterior[:, 0, :], observations)
    feats_approx = np.concatenate(
        [posterior_samples[:, 0, :], observations], axis=1
    )

    ndim = feats_joint.shape[1]
    if clf_kwargs is None:
        clf_kwargs = _default_clf_kwargs(ndim)

    labels = np.concatenate([np.ones(n_sims), np.zeros(n_sims)])

    # --- Cross-validated predictions ---
    probs_class0 = _run_lc2st_cv(
        feats_joint, feats_approx, labels,
        n_folds, clf_kwargs, rng,
    )

    # --- Single-class MSE_0 (Theorem 3.1) ---
    per_obs = (probs_class0 - 0.5) ** 2
    statistic = float(np.mean(per_obs))

    # --- Null distribution (Algorithm 1, lines 10-14) ---
    null_statistics = np.array([])
    p_value: float | None = None

    if n_null_trials > 0:
        null_stats = np.zeros(n_null_trials)
        for trial_i in range(n_null_trials):
            # Permute labels
            perm_labels = labels.copy()
            rng.shuffle(perm_labels)
            perm_probs = _run_lc2st_cv(
                feats_joint, feats_approx, perm_labels,
                n_folds, clf_kwargs, rng,
            )
            null_stats[trial_i] = float(
                np.mean((perm_probs - 0.5) ** 2)
            )
        null_statistics = null_stats
        p_value = float(np.mean(null_stats >= statistic))

    return LC2STResult(
        statistic=statistic,
        p_value=p_value,
        null_statistics=null_statistics,
        per_observation_stats=per_obs,
    )


# ---------------------------------------------------------------------------
# Global C2ST (López-Paz & Oquab, 2017)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GlobalC2STResult:
    """Result container for :func:`global_c2st`.

    Attributes
    ----------
    accuracy
        Classification accuracy on the held-out test set.
        Values near 0.5 indicate indistinguishable distributions.
    p_value
        One-sided p-value from the normal approximation to the
        null distribution (Theorem 1).
    n_test
        Number of test samples used.
    """

    accuracy: float
    p_value: float
    n_test: int


def global_c2st(
    samples_p: np.ndarray,
    samples_q: np.ndarray,
    *,
    clf_kwargs: dict[str, Any] | None = None,
    seed: int = 42,
) -> GlobalC2STResult:
    """Standard Classifier Two-Sample Test (López-Paz & Oquab, 2017).

    Trains a binary classifier to discriminate samples from two
    distributions. If classification accuracy significantly exceeds
    chance (0.5), the distributions differ.

    Parameters
    ----------
    samples_p
        Samples from distribution P, shape ``(n, d)``. Must have the
        same number of samples as ``samples_q``.
    samples_q
        Samples from distribution Q, shape ``(n, d)``. Must have the
        same number of samples as ``samples_p``.
    clf_kwargs
        Override keyword arguments for ``MLPClassifier``. If ``None``,
        uses a simple MLP with ``hidden_layer_sizes=(20,)``.
    seed
        Random seed for reproducibility.

    Returns
    -------
    GlobalC2STResult
        Result with accuracy, p-value, and test set size.

    Raises
    ------
    ValueError
        If inputs have different numbers of features.
    ImportError
        If scikit-learn is not installed.
    """
    mlp_cls, _ = _require_sklearn()
    rng = np.random.default_rng(seed)

    samples_p = np.asarray(samples_p)
    samples_q = np.asarray(samples_q)

    if samples_p.ndim == 1:
        samples_p = samples_p[:, None]
    if samples_q.ndim == 1:
        samples_q = samples_q[:, None]
    if samples_p.shape[1] != samples_q.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: samples_p has "
            f"{samples_p.shape[1]}, samples_q has "
            f"{samples_q.shape[1]}"
        )
    if len(samples_p) != len(samples_q):
        raise ValueError(
            f"Samples must have equal size for the normal-"
            f"approximation p-value (Theorem 1). Got "
            f"len(samples_p)={len(samples_p)}, "
            f"len(samples_q)={len(samples_q)}."
        )

    # Pool and shuffle
    features = np.concatenate([samples_p, samples_q], axis=0)
    targets = np.concatenate([
        np.ones(len(samples_p)),
        np.zeros(len(samples_q)),
    ])
    perm = rng.permutation(len(features))
    features = features[perm]
    targets = targets[perm]

    # 50-50 train/test split
    n_total = len(features)
    n_train = n_total // 2
    n_test = n_total - n_train
    x_train, x_test = features[:n_train], features[n_train:]
    y_train, y_test = targets[:n_train], targets[n_train:]

    # Train classifier
    if clf_kwargs is None:
        clf_kwargs = {
            "hidden_layer_sizes": (20,),
            "max_iter": 100,
            "solver": "adam",
        }
    clf = mlp_cls(**clf_kwargs, random_state=seed)
    clf.fit(x_train, y_train)

    accuracy = float(clf.score(x_test, y_test))

    # López-Paz & Oquab (2017), Theorem 1: p-value from normal approximation
    # Null: accuracy ~ N(0.5, 1/(4*n_test))
    se = np.sqrt(1.0 / (4.0 * n_test))
    z = (accuracy - 0.5) / se
    p_value = float(1.0 - norm.cdf(z))

    return GlobalC2STResult(
        accuracy=accuracy, p_value=p_value, n_test=n_test
    )


# ---------------------------------------------------------------------------
# L-C2ST as a joint metric
# ---------------------------------------------------------------------------


def _joint_observations(inputs: JointMetricInputs) -> np.ndarray:
    """Flatten and concatenate the data keys of one condition batch.

    L-C2ST classifies on ``concat(params, observations)``, so every
    observable has to arrive as a flat ``(n_sims, d)`` block regardless of
    the shape the simulator produced.
    """
    parts = [np.asarray(inputs.sim_batch[k]) for k in inputs.data_keys]
    flat = [
        p.reshape(p.shape[0], -1) if p.ndim > 1 else p[:, None]
        for p in parts
    ]
    return np.concatenate(flat, axis=1)


_LC2ST_SCALAR_ONLY = (
    "L-C2ST classifies on concat(params, observations), one row per "
    "simulation. With vector-valued parameters the pipeline pools one row per "
    "(simulation, element), and those rows do not line up with the "
    "observations' rows. Use a TARP metric, or validate L-C2ST with a custom "
    "`validate_fn`."
)


def _subsampled_conditions(n_conditions: int, n_keep: int) -> set[int]:
    """Pick *n_keep* condition indices spread evenly over the grid.

    Evenly spaced rather than the first *n_keep*: a validation grid is
    ordered, so a prefix samples one corner of it. `np.linspace` with
    rounding gives the same set on every trial of a study, which matters
    because a metric evaluated on different conditions between trials is
    not a comparable objective.
    """
    if n_keep >= n_conditions:
        return set(range(n_conditions))
    if n_keep < 1:
        # Silently keeping one condition would break the documented "at
        # most this many" contract in the direction that costs money, and
        # zero conditions would leave the metric absent from every summary
        # -- which the objective reads as a failure and penalizes.
        raise ValueError(f"max_conditions must be at least 1, got {n_keep}.")
    if n_keep == 1:
        # `np.linspace(0, n-1, 1)` is [0], the grid's first corner -- the
        # exact prefix this function exists to avoid. The middle is the
        # least unrepresentative single condition available.
        return {n_conditions // 2}
    idx = np.linspace(0, n_conditions - 1, n_keep)
    return {int(round(i)) for i in idx}


def _lc2st_settings(
    *,
    n_folds: int,
    n_null_trials: int,
    clf_kwargs: dict[str, Any] | None,
    seed: int,
    max_conditions: int | None,
) -> dict[str, Any]:
    """Everything the L-C2ST statistic moves with, as a JSON-safe dict.

    A free function rather than a line inside the factory, because the
    REGISTERED ``"lc2st"`` name has to declare the same settings without
    calling the factory: the factory guards on scikit-learn, and running it
    at module scope would make an optional dependency mandatory. That is the
    same trap `_default_lc2st_metric` exists to avoid, and reaching for
    ``make_lc2st_joint_metric().joint_metric_settings`` to fill it walks
    straight back into it.
    """
    return {
        "n_folds": int(n_folds),
        "n_null_trials": int(n_null_trials),
        "seed": int(seed),
        "max_conditions": (
            None if max_conditions is None else int(max_conditions)
        ),
        # The classifier changes the statistic, but its kwargs are an
        # arbitrary nested dict; a repr is comparable and JSON-safe, which
        # is all the pin needs.
        "clf_kwargs": (
            None if clf_kwargs is None else repr(sorted(clf_kwargs.items()))
        ),
    }


def make_lc2st_joint_metric(
    n_folds: int = 5,
    n_null_trials: int = 0,
    clf_kwargs: dict[str, Any] | None = None,
    seed: int = 42,
    max_conditions: int | None = None,
) -> JointMetricFn:
    """Create an L-C2ST metric for the pipeline's joint dispatch.

    L-C2ST is joint by construction: it asks whether a classifier can tell
    ``(theta, x)`` drawn from the joint apart from ``(theta_hat, x)`` with
    ``theta_hat`` from the approximate posterior, so it needs all parameters
    at once AND the data -- which is precisely what
    :class:`~bayesflow_hpo.validation.registry.JointMetricInputs` carries.

    Parameters
    ----------
    n_folds
        Number of cross-validation folds.
    n_null_trials
        Permutation trials for the null distribution. Default 0: during HPO
        the statistic is compared across trials rather than against a null,
        and each permutation costs another full cross-validation.
    clf_kwargs
        Classifier keyword arguments. ``None`` uses the SBIBM defaults.
    seed
        Base seed. Each condition uses ``seed + cond_id`` so that conditions
        do not share a draw, which would correlate their noise.
    max_conditions
        Evaluate on at most this many conditions, spread evenly over the
        grid, instead of all of them. ``None`` (default) uses every
        condition. The cost is linear in the condition count and the
        constant is large -- see the note below -- so this is the lever that
        makes L-C2ST affordable as an objective without changing what it
        measures per condition. The subset is deterministic, so every trial
        in a study is scored on the same conditions; a subset that varied
        between trials would not be a comparable objective.

    Returns
    -------
    JointMetricFn
        Callable emitting ``{"lc2st": statistic}`` for one condition.

    Raises
    ------
    ImportError
        If scikit-learn is not installed, raised at factory call time
        rather than per condition.

    Notes
    -----
    **L-C2ST is expensive**, and measurably so: at 500 simulations with 15
    parameters one condition took ~54 s, against ~79 ms for a TARP
    evaluation on the same draws -- a factor of roughly 700. The cost is in
    fitting a classifier per fold, so it scales with `n_folds` and with
    ``n_null_trials + 1``. Budget for it before making it an objective, and
    see ``docs/plans/plan-joint-metric-path.md`` D9 for the measurements.

    References
    ----------
    Linhart, J., Gramfort, A., & Rodrigues, P. L. C. (2023). L-C2ST: Local
        diagnostics for posterior approximations in simulation-based
        inference. In *Advances in Neural Information Processing Systems
        36*. https://doi.org/10.48550/arXiv.2306.03580
        Algorithm 1 and Theorem 3.1: the single-class MSE_0 statistic.
    """
    _require_sklearn()
    # Validated HERE, not per condition. These options do not depend on the
    # data, so leaving them to the numerical guard turns a typo into a
    # per-condition exception, which the guard converts into the metric's
    # registered worst case -- so every trial trains to completion and
    # scores an identical 0.25, and the study optimizes a constant behind a
    # warning log. A configuration error the caller can fix in a line must
    # not cost a training run, let alone a whole study's worth.
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}.")
    if n_null_trials < 0:
        raise ValueError(
            f"n_null_trials must be non-negative, got {n_null_trials}."
        )
    if max_conditions is not None and max_conditions < 1:
        raise ValueError(
            f"max_conditions must be at least 1, got {max_conditions}."
        )

    def _lc2st_metric(inputs: JointMetricInputs) -> dict[str, float]:
        if max_conditions is not None:
            keep = _subsampled_conditions(inputs.n_conditions, max_conditions)
            if inputs.cond_id not in keep:
                # An empty dict contributes no row, and the pipeline means
                # each joint key over the conditions that reported it. So a
                # skipped condition costs nothing and biases nothing --
                # unlike a sentinel value, which would be averaged in.
                return {}
        true_params = np.column_stack([
            np.asarray(inputs.sim_batch[k]).ravel()
            for k in inputs.param_keys
        ])
        result = lc2st(
            posterior_samples=inputs.draws,
            true_params=true_params,
            observations=_joint_observations(inputs),
            n_folds=n_folds,
            n_null_trials=n_null_trials,
            clf_kwargs=clf_kwargs,
            seed=seed + inputs.cond_id,
        )
        return {"lc2st": float(result.statistic)}

    setattr(_lc2st_metric, REQUIRES_SCALAR_PARAMETERS, _LC2ST_SCALAR_ONLY)
    _lc2st_metric.joint_metric_settings = _lc2st_settings(  # type: ignore[attr-defined]
        n_folds=n_folds,
        n_null_trials=n_null_trials,
        clf_kwargs=clf_kwargs,
        seed=seed,
        max_conditions=max_conditions,
    )
    return _lc2st_metric


def _default_lc2st_metric(inputs: JointMetricInputs) -> dict[str, float]:
    """The registered ``"lc2st"`` name, at its default configuration.

    Builds the metric per call rather than once at import, so that the
    sklearn guard fires when the metric RUNS. Calling the factory at module
    scope would raise `ImportError` during ``import bayesflow_hpo`` on any
    installation without scikit-learn -- making an optional dependency
    mandatory, which is the opposite of what `requires="sklearn"` is meant
    to express. (`requires=` itself gates nothing today: it is read only by
    `describe_metrics` for display. The explicit guard is the real
    mechanism.)
    """
    return make_lc2st_joint_metric()(inputs)


# The registered name must declare the SAME settings the default factory
# produces, or the pin would cover a configured L-C2ST and not the
# registry's own -- so a study resumed with plain
# `objective_metrics=["lc2st"]` would compare against settings that were
# never this metric's.
#
# Note what this does NOT buy: a study driven by `make_lc2st_validate_fn`
# is on the `validate_fn` branch, which never reaches the pin at all
# (`optimization/objective.py` calls
# `check_or_stamp_joint_metric_settings` only in the `else` of
# `if config.validate_fn is not None`). A hook returns a flat dict, not a
# `ValidationResult`, so there is nothing to read the declaration off.
# `objectives.check_or_stamp_joint_metric_settings` records that as the
# fourth thing the pin cannot do.
def _check_lc2st_dependency() -> None:
    """Resolve-time guard for the registered ``lc2st``.

    Checked when the metric is RESOLVED, not per condition. Left to run
    time, the ImportError is caught by the pipeline's joint guard, the
    metric is invalidated, and the objective substitutes its registered
    worst case of 0.25 -- identically on every trial. A study with a
    missing optional dependency then trains to completion over and over
    while optimizing a constant, with only a warning log to say so. That is
    the same "fixable in one line, do not pay for a training run first"
    case `_bf_hpo_requires_configuration` exists for.

    This is what `requires="sklearn"` would do if it gated anything. It does
    not -- `_REQUIRES` is read only by `describe_metrics` (see the plan's
    section 3) -- so the guard is wired explicitly.

    Indirects through the module attribute rather than binding
    `_require_sklearn` directly, so the guard a caller or a test replaces is
    the one that runs; binding the function object at import time makes the
    check unpatchable, which is how a guard quietly stops being observable.
    """
    _require_sklearn()


setattr(_default_lc2st_metric, REQUIRES_SCALAR_PARAMETERS, _LC2ST_SCALAR_ONLY)

_default_lc2st_metric._bf_hpo_resolve_check = (  # type: ignore[attr-defined]
    _check_lc2st_dependency
)

_default_lc2st_metric.joint_metric_settings = _lc2st_settings(  # type: ignore[attr-defined]
    n_folds=5, n_null_trials=0, clf_kwargs=None, seed=42, max_conditions=None
)


# Registered here rather than in `registry.py` because `c2st` imports the
# registry and the reverse would be a cycle. `validation/__init__` imports
# this module, so the name is present for anyone who can reach the registry
# at all.
register_joint_metric(
    "lc2st",
    _default_lc2st_metric,
    description=(
        "L-C2ST local posterior diagnostic on the full joint posterior "
        "(expensive: ~700x a TARP evaluation)"
    ),
    requires="sklearn",
    overwrite=True,
)


# ---------------------------------------------------------------------------
# ValidateFn factory
# ---------------------------------------------------------------------------


def make_lc2st_validate_fn(
    base_metrics: list[str] | None = None,
    n_folds: int = 5,
    n_null_trials: int = 0,
    clf_kwargs: dict[str, Any] | None = None,
    seed: int = 42,
    max_conditions: int | None = None,
    max_samples_per_call: int | None = DEFAULT_MAX_SAMPLES_PER_CALL,
) -> Callable[[Any, ValidationDataset, int], dict[str, float]]:
    """Create a ``ValidateFn`` that computes standard metrics + L-C2ST.

    The returned function is compatible with
    ``optimize(validate_fn=...)``. It runs inference once per condition,
    computes per-parameter standard metrics, and additionally runs
    L-C2ST on the full multivariate posterior.

    Parameters
    ----------
    base_metrics
        List of standard metric names to compute alongside L-C2ST.
        If ``None``, uses ``["calibration_error", "nrmse"]``.
    n_folds
        Number of CV folds for L-C2ST.
    n_null_trials
        Number of permutation trials for L-C2ST null distribution.
        Default 0 (skip permutation test during HPO for speed).
    clf_kwargs
        Override classifier kwargs for L-C2ST. If ``None``, uses
        SBIBM defaults.
    seed
        Random seed for L-C2ST reproducibility.
    max_conditions
        Evaluate L-C2ST on at most this many conditions, spread evenly over
        the grid. ``None`` (default) uses all of them. See
        :func:`make_lc2st_joint_metric`.
    max_samples_per_call
        Cap on posterior draws per ``approximator.sample()`` call, forwarded
        to :func:`~bayesflow_hpo.validation.pipeline.run_validation_pipeline`.
        Set here rather than read from ``optimize()``: the ``ValidateFn``
        contract is ``(approximator, validation_data, n_posterior_samples)``,
        so a hook cannot receive ``optimize(max_samples_per_call=...)`` and
        this factory call is the only place to change it.

    Returns
    -------
    ValidateFn
        ``(approximator, validation_data, n_posterior_samples) -> dict``

    Raises
    ------
    ImportError
        If scikit-learn is not installed (at factory call time).

    Notes
    -----
    This is now a thin wrapper over
    :func:`~bayesflow_hpo.validation.pipeline.run_validation_pipeline` with
    one joint metric passed in. It previously reimplemented that pipeline's
    condition loop -- inference, the per-parameter branch, the
    single-parameter squeeze, cross-condition aggregation -- in order to
    reach state the loop had and discarded. Everything the duplicate
    open-coded is now shared, including two things it never had:
    ``cleanup_trial()`` between conditions, and a guard that keeps an
    L-C2ST failure from costing the trial its standard metrics as well.

    Because L-C2ST's configuration belongs to this factory call rather than
    to the process, the metric is passed through ``joint_metrics=`` instead
    of being registered globally. The registry's built-in ``"lc2st"`` name
    carries the defaults and is what ``objective_metrics=["lc2st"]``
    resolves to.
    """
    _require_sklearn()

    if base_metrics is None:
        base_metrics = ["calibration_error", "nrmse"]

    joint = make_lc2st_joint_metric(
        n_folds=n_folds,
        n_null_trials=n_null_trials,
        clf_kwargs=clf_kwargs,
        seed=seed,
        max_conditions=max_conditions,
    )

    def _validate_fn(
        approximator: Any,
        validation_data: ValidationDataset,
        n_posterior_samples: int,
    ) -> dict[str, float]:
        result = run_validation_pipeline(
            approximator=approximator,
            validation_data=validation_data,
            n_posterior_samples=n_posterior_samples,
            metrics=list(base_metrics),
            joint_metrics={"lc2st": joint},
            max_samples_per_call=max_samples_per_call,
        )
        return dict(result.summary)

    return _validate_fn
