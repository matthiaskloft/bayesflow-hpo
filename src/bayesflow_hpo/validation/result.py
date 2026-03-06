"""Structured validation result with table display methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ValidationResult:
    """Immutable container for grid-based validation results.

    Parameters
    ----------
    condition_metrics
        DataFrame with one row per condition, columns are metric values.
    summary
        Overall mean across conditions for each metric key.
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
    """

    condition_metrics: pd.DataFrame
    summary: dict[str, float]
    per_parameter: dict[str, ValidationResult] | None = None
    timing: dict[str, float] = field(default_factory=dict)
    n_conditions: int = 0
    n_posterior_samples: int = 0
    metric_names: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Table methods
    # ------------------------------------------------------------------

    def summary_table(self) -> pd.DataFrame:
        """Single-row DataFrame with overall summary metrics."""
        return pd.DataFrame([self.summary])

    def condition_table(self, metric: str | None = None) -> pd.DataFrame:
        """Per-condition DataFrame, optionally filtered to columns matching *metric*."""
        if metric is None:
            return self.condition_metrics.copy()
        cols = [
            c for c in self.condition_metrics.columns
            if metric in c or c == "id_cond"
        ]
        return self.condition_metrics[cols].copy()

    def parameter_table(self) -> pd.DataFrame | None:
        """Per-parameter summary (multi-parameter models only)."""
        if self.per_parameter is None:
            return None
        rows: list[dict[str, Any]] = []
        for param_name, param_result in self.per_parameter.items():
            row: dict[str, Any] = {"parameter": param_name}
            row.update(param_result.summary)
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Objective extraction
    # ------------------------------------------------------------------

    def objective_scalar(self, key: str = "mean_cal_error") -> float:
        """Extract a single scalar for HPO objective from summary dict.

        Falls back to ``calibration_error`` → ``1.0`` if *key* is missing.
        """
        return float(self.summary.get(key, self.summary.get("mean_cal_error", 1.0)))

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
        if self.timing:
            total = sum(self.timing.values())
            lines.append(f"  Timing: {total:.1f}s total")
        return "\n".join(lines)
