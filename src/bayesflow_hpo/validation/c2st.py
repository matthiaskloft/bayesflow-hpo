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
from bayesflow_hpo.validation.inference import make_bayesflow_infer_fn
from bayesflow_hpo.validation.metrics import (
    aggregate_condition_rows,
    compute_condition_metrics,
)
from bayesflow_hpo.validation.registry import resolve_metrics

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
    """Run K-fold CV and return per-observation predicted probs.

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
        Predicted probabilities for class-0 samples on their
        respective validation folds, shape ``(n_sims,)``.
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

    if n_sims < n_folds:
        raise ValueError(
            f"n_sims ({n_sims}) must be >= n_folds ({n_folds})"
        )

    # --- Build training data (paper eq. 299) ---
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

    # --- Single-class MSE_0 (Theorem 3.1, eq. 310) ---
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
        Samples from distribution P, shape ``(n, d)``.
    samples_q
        Samples from distribution Q, shape ``(n, d)``.
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

    # p-value from normal approximation (Theorem 1)
    # Null: accuracy ~ N(0.5, 1/(4*n_test))
    se = np.sqrt(1.0 / (4.0 * n_test))
    z = (accuracy - 0.5) / se
    p_value = float(1.0 - norm.cdf(z))

    return GlobalC2STResult(
        accuracy=accuracy, p_value=p_value, n_test=n_test
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

    Returns
    -------
    ValidateFn
        ``(approximator, validation_data, n_posterior_samples) -> dict``

    Raises
    ------
    ImportError
        If scikit-learn is not installed (at factory call time).
    """
    _require_sklearn()

    if base_metrics is None:
        base_metrics = ["calibration_error", "nrmse"]

    def _validate_fn(
        approximator: Any,
        validation_data: ValidationDataset,
        n_posterior_samples: int,
    ) -> dict[str, float]:
        metric_fns = resolve_metrics(base_metrics)

        # Create inference closure (reuses validation/inference.py)
        available_keys = set(validation_data.simulations[0].keys())
        infer_fn = make_bayesflow_infer_fn(
            approximator=approximator,
            param_keys=validation_data.param_keys,
            data_keys=validation_data.data_keys,
            available_keys=available_keys,
        )

        condition_rows: list[dict[str, Any]] = []
        lc2st_stats: list[float] = []
        n_params = len(validation_data.param_keys)

        for cond_id, sim_batch in enumerate(
            validation_data.simulations
        ):
            # Single inference pass
            draws = infer_fn(sim_batch, n_posterior_samples)

            # --- Standard per-parameter metrics ---
            if n_params == 1:
                true_values = np.asarray(
                    sim_batch[validation_data.param_keys[0]]
                ).ravel()
                draws_2d = (
                    draws if draws.ndim == 2 else draws[..., 0]
                )
                row = compute_condition_metrics(
                    draws_2d, true_values, cond_id, metric_fns,
                )
                condition_rows.append(row)
            else:
                for p_idx, p_key in enumerate(
                    validation_data.param_keys
                ):
                    true_values = np.asarray(
                        sim_batch[p_key]
                    ).ravel()
                    draws_2d = draws[:, :, p_idx]
                    row = compute_condition_metrics(
                        draws_2d, true_values, cond_id, metric_fns,
                    )
                    row["param_key"] = p_key
                    condition_rows.append(row)

            # --- L-C2ST on full multivariate posterior ---
            tp = np.column_stack([
                np.asarray(sim_batch[k]).ravel()
                for k in validation_data.param_keys
            ])

            obs_parts = [
                np.asarray(sim_batch[k])
                for k in validation_data.data_keys
            ]
            obs_flat = [
                p.reshape(p.shape[0], -1)
                if p.ndim > 1
                else p[:, None]
                for p in obs_parts
            ]
            obs = np.concatenate(obs_flat, axis=1)

            # Ensure draws are 3D for lc2st
            draws_3d = (
                draws[:, :, None] if draws.ndim == 2 else draws
            )

            result = lc2st(
                posterior_samples=draws_3d,
                true_params=tp,
                observations=obs,
                n_folds=n_folds,
                n_null_trials=n_null_trials,
                clf_kwargs=clf_kwargs,
                seed=seed + cond_id,
            )
            lc2st_stats.append(result.statistic)

        # Aggregate standard metrics across conditions
        summary = aggregate_condition_rows(condition_rows)

        # Average L-C2ST statistic across conditions
        summary["lc2st"] = float(np.mean(lc2st_stats))

        return summary

    return _validate_fn
