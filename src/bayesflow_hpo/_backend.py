"""Fail early, with an actionable message, when no Keras backend is installed.

``bayesflow-hpo`` installs Keras but no backend (see the README). BayesFlow
2.x auto-selects an installed backend only when ``KERAS_BACKEND`` is unset
(``bayesflow/__init__.py``, ``setup()``, bayesflow 2.0.12). When the variable
names a backend that is not installed, the failure surfaces deep inside Keras
instead. This check covers both cases before anything imports Keras.
"""

from __future__ import annotations

import os
from importlib.util import find_spec

# KERAS_BACKEND value -> importable package. The backends BayesFlow 2.0.12
# auto-detects (``setup()`` in ``bayesflow/__init__.py``).
_BACKEND_PACKAGES = {"jax": "jax", "torch": "torch", "tensorflow": "tensorflow"}

_INSTALL_HINT = (
    "bayesflow-hpo does not install a Keras backend. Install one, e.g.\n"
    "    pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
    "and set KERAS_BACKEND=torch before importing bayesflow_hpo. JAX and "
    "TensorFlow also work. See "
    "https://keras.io/getting_started/#configuring-your-backend"
)


def check_backend() -> None:
    """Raise ``ImportError`` if the selected or any Keras backend is missing.

    Raises
    ------
    ImportError
        If ``KERAS_BACKEND`` names jax, torch or tensorflow and that package
        is not installed, or if ``KERAS_BACKEND`` is unset and none of the
        three is installed. Other ``KERAS_BACKEND`` values are left to Keras.
    """
    selected = os.environ.get("KERAS_BACKEND")
    if selected is not None:
        package = _BACKEND_PACKAGES.get(selected.lower())
        if package is not None and find_spec(package) is None:
            raise ImportError(
                f"KERAS_BACKEND={selected!r}, but {package!r} is not "
                f"installed.\n{_INSTALL_HINT}"
            )
        return
    if not any(find_spec(p) is not None for p in _BACKEND_PACKAGES.values()):
        raise ImportError(f"No Keras backend is installed.\n{_INSTALL_HINT}")
