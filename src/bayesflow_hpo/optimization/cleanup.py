"""Cleanup helpers between optimization trials."""

from __future__ import annotations

import gc


def cleanup_trial() -> None:
    """Clear CPU/GPU state between trials."""
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass

    try:
        import tensorflow as tf

        tf.keras.backend.clear_session()
    except (ImportError, AttributeError):
        pass
