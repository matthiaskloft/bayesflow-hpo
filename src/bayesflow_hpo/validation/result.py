"""Structured validation result with table display methods.

``ValidationResult`` is the return type of :func:`run_validation_pipeline`.
It stores per-condition metrics, overall summary statistics, and optional
per-parameter breakdowns for multi-parameter models.  The ``summary``
dict is the primary interface consumed by the objective function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from bayesflow_hpo._display import DisplayDataFrame


@dataclass(frozen=True)
class ValidationResult:
    """Immutable container for grid-based validation results.

    Parameters
    ----------
    condition_metrics
        DataFrame with one row per condition, columns are metric values.
        Marginal metrics only; joint ones have no parameter axis and live in
        *joint_condition_metrics*.
    summary
        Overall configured reduction for each metric key (mean by default).
    per_parameter
        Optional mapping from parameter name to per-parameter
        ``ValidationResult`` (for multi-parameter models).
    timing
        Wall-clock seconds for ``"inference"`` and ``"metrics"`` phases.
    n_conditions
        Number of conditions in the validation grid.
    n_posterior_samples
        Number of posterior samples drawn per simulation.
    metric_names
        Ordered list of metric names that were computed.
    joint_metric_settings
        Mapping from joint metric name to the configuration it declared,
        for metrics that declare one. A joint metric's score moves with its
        settings, so this is what lets a resumed study detect that its
        objective changed scale under it.
    failed_joint_metrics
        Mapping from joint metric name to the exception that invalidated it
        for this trial, empty when none failed. A joint metric that raises
        on any condition is dropped from *summary* entirely so the objective
        substitutes its registered worst case; without this field the only
        evidence would be a penalty value, which is indistinguishable from a
        genuinely bad model.
    joint_condition_metrics
        DataFrame with one row per condition and one column per surviving
        joint metric output, plus ``id_cond``; empty when no joint metric
        ran. A joint metric can be valid on one condition and blind on
        another, and the reduction in *summary* cannot show which, so the
        values behind it are kept. Columns of a metric listed in
        *failed_joint_metrics* are absent here as they are from *summary*.
    """

    condition_metrics: pd.DataFrame
    summary: dict[str, float]
    per_parameter: dict[str, ValidationResult] | None = None
    timing: dict[str, float] = field(default_factory=dict)
    n_conditions: int = 0
    n_posterior_samples: int = 0
    metric_names: list[str] = field(default_factory=list)
    failed_joint_metrics: dict[str, str] = field(default_factory=dict)
    joint_metric_settings: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )
    joint_condition_metrics: pd.DataFrame = field(
        default_factory=pd.DataFrame
    )

    # ------------------------------------------------------------------
    # Table methods
    # ------------------------------------------------------------------

    def summary_table(self) -> DisplayDataFrame:
        """Single-row DataFrame with overall summary metrics."""
        return DisplayDataFrame([self.summary])

    def condition_table(self, metric: str | None = None) -> DisplayDataFrame:
        """Per-condition DataFrame, optionally filtered to columns matching *metric*."""
        if metric is None:
            return DisplayDataFrame(self.condition_metrics)
        cols = [
            c for c in self.condition_metrics.columns
            if metric in c or c == "id_cond"
        ]
        return DisplayDataFrame(self.condition_metrics[cols])

    def joint_condition_table(
        self, metric: str | None = None
    ) -> DisplayDataFrame:
        """Per-condition joint DataFrame, optionally filtered to *metric*.

        The joint counterpart of :meth:`condition_table`; empty when the run
        computed no joint metrics.
        """
        if metric is None:
            return DisplayDataFrame(self.joint_condition_metrics)
        cols = [
            c for c in self.joint_condition_metrics.columns
            if metric in c or c == "id_cond"
        ]
        return DisplayDataFrame(self.joint_condition_metrics[cols])

    def parameter_table(self) -> DisplayDataFrame | None:
        """Per-parameter summary (multi-parameter models only)."""
        if self.per_parameter is None:
            return None
        rows: list[dict[str, Any]] = []
        for param_name, param_result in self.per_parameter.items():
            row: dict[str, Any] = {"parameter": param_name}
            row.update(param_result.summary)
            rows.append(row)
        return DisplayDataFrame(rows)

    # ------------------------------------------------------------------
    # Objective extraction
    # ------------------------------------------------------------------

    def objective_scalar(self, key: str = "calibration_error") -> float:
        """Extract a single scalar for HPO objective from summary dict.

        Falls back to ``1.0`` if *key* is missing.
        """
        return float(self.summary.get(key, 1.0))

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        lines = [
            f"ValidationResult(n_conditions={self.n_conditions}, "
            f"n_posterior_samples={self.n_posterior_samples})",
        ]
        if self.summary:
            lines.append("  Summary:")
            for k, v in self.summary.items():
                if isinstance(v, float):
                    lines.append(f"    {k}: {v:.4f}")
                else:
                    lines.append(f"    {k}: {v}")
        if self.per_parameter:
            lines.append(f"  Parameters: {list(self.per_parameter.keys())}")
        if self.failed_joint_metrics:
            lines.append("  Failed joint metrics:")
            # Distinct names from the summary loop above, whose `v` mypy
            # infers as float from `dict[str, float]`. Reusing them assigns
            # a str to a float-typed variable.
            for name, reason in self.failed_joint_metrics.items():
                lines.append(f"    {name}: {reason}")
        if self.timing:
            total = sum(self.timing.values())
            lines.append(f"  Timing: {total:.1f}s total")
        return "\n".join(lines)
