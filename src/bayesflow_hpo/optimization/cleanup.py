"""Cleanup helpers between optimization trials.

Keras/PyTorch can accumulate GPU memory across trials due to lazy
deallocation and graph caching.  Explicit cleanup between trials
prevents OOM errors that would otherwise appear after several
consecutive large models.
"""

from __future__ import annotations

import gc


def cleanup_trial() -> None:
    """Clear CPU/GPU state between trials.

    Runs Python garbage collection, then backend-specific cleanup:

    - **PyTorch**: empties the CUDA cache and synchronizes.
    - **TensorFlow**: clears the Keras session graph.

    Called after every trial (successful, failed, or pruned) and after
    each validation condition batch.
    """
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
