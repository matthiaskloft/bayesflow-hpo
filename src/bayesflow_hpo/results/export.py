"""Export and load workflows with metadata sidecars."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import bayesflow as bf
import numpy as np


def get_workflow_metadata(
    config: dict[str, Any],
    model_type: str,
    validation_results: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect generic reproducibility metadata for a trained workflow."""
    metadata: dict[str, Any] = {
        "config": config,
        "versions": {
            "bayesflow": bf.__version__,
            "numpy": np.__version__,
        },
        "created_at": datetime.now().isoformat(),
        "model_type": model_type,
    }

    if validation_results is not None:
        metadata["validation"] = validation_results

    if extra is not None:
        metadata.update(extra)

    return metadata


def save_workflow_with_metadata(approximator: Any, path: str | Path, metadata: dict[str, Any]) -> Path:
    """Save workflow model (`.keras`) and metadata (`.json`)."""
    base_path = Path(path)
    model_path = base_path.with_suffix(".keras")
    meta_path = base_path.with_suffix(".json")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    approximator.save(model_path)
    meta_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return model_path


def load_workflow_with_metadata(path: str | Path) -> tuple[Any, dict[str, Any]]:
    """Load approximator model and metadata sidecar if present.

    Returns the *approximator* (a Keras model), not a full ``BasicWorkflow``,
    because BayesFlow 2.x persists the approximator via ``keras.saving`` and
    reconstructing a workflow requires the original simulator and adapter.
    """
    import keras

    base_path = Path(path)
    model_path = base_path.with_suffix(".keras")
    meta_path = base_path.with_suffix(".json")

    approximator = keras.saving.load_model(model_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return approximator, metadata
