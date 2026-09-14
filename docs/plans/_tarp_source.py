"""Locate and load ``compute_tarp_coverage`` for the #82 cost benchmarks.

TARP lives in ``bayesflow-irt``, which is not a dependency of this package and
is not vendored here. Both benchmark scripts need exactly one function from it,
so this module resolves that dependency explicitly instead of each script
reaching for a scratch file that happens to be next to it.

Resolution order
----------------
1. ``--tarp-source PATH`` on the command line, or ``$BF_HPO_TARP_SOURCE``.
2. An installed ``bayesflow_irt`` package, if one is importable.
3. Failure, with instructions.

Obtaining the file without installing ``bayesflow-irt``::

    git -C /path/to/bayesflow-irt show \
        ffc68d59ae311d1aeaa5e066f39f7e0badc6c853:src/bayesflow_irt/sbc.py \
        > /tmp/irt_sbc.py
    python docs/plans/bench_joint_metric_cost.py --tarp-source /tmp/irt_sbc.py

``sbc.py`` at that commit imports only ``math``/``numpy`` and the stdlib, so it
loads standalone with no ``bayesflow_irt`` package context. The commit is
pinned because the measurements in ``plan-joint-metric-path.md`` were taken
against it: a later revision may change ``tarp_error``'s definition, and the
docstring this plan quotes for the floor behaviour is specific to it.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from collections.abc import Callable
from pathlib import Path

TARP_COMMIT = "ffc68d59ae311d1aeaa5e066f39f7e0badc6c853"

_INSTRUCTIONS = f"""\
Could not locate bayesflow-irt's `compute_tarp_coverage`.

Supply it one of these ways:

  1. Extract the pinned revision from a bayesflow-irt checkout:

       git -C <bayesflow-irt> show \
           {TARP_COMMIT}:src/bayesflow_irt/sbc.py > /tmp/irt_sbc.py

     then re-run with:  --tarp-source /tmp/irt_sbc.py
     (or set $BF_HPO_TARP_SOURCE to that path)

  2. Install bayesflow-irt into this environment so `bayesflow_irt.sbc`
     imports. Note the numbers in plan-joint-metric-path.md were measured
     against commit {TARP_COMMIT[:7]}; a different revision may not be
     comparable.
"""


def add_tarp_argument(parser: argparse.ArgumentParser) -> None:
    """Register the shared ``--tarp-source`` option on *parser*."""
    parser.add_argument(
        "--tarp-source",
        type=Path,
        default=os.environ.get("BF_HPO_TARP_SOURCE"),
        help=(
            "Path to bayesflow-irt's sbc.py at commit "
            f"{TARP_COMMIT[:7]}. Defaults to $BF_HPO_TARP_SOURCE, then to an "
            "installed bayesflow_irt package."
        ),
    )


def load_compute_tarp_coverage(source: Path | str | None = None) -> Callable:
    """Return ``compute_tarp_coverage``, or raise with instructions."""
    if source is not None:
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"--tarp-source {path} does not exist.\n\n{_INSTRUCTIONS}")
        spec = importlib.util.spec_from_file_location("_irt_sbc", path)
        if spec is None or spec.loader is None:
            raise SystemExit(f"Could not load a module from {path}.")
        module = importlib.util.module_from_spec(spec)
        sys.modules["_irt_sbc"] = module
        spec.loader.exec_module(module)
    else:
        try:
            from bayesflow_irt import sbc as module  # type: ignore[no-redef]
        except ImportError:
            raise SystemExit(_INSTRUCTIONS) from None

    try:
        return module.compute_tarp_coverage
    except AttributeError:
        raise SystemExit(
            f"{source or 'bayesflow_irt.sbc'} has no `compute_tarp_coverage`. "
            f"Expected the module at commit {TARP_COMMIT[:7]}.\n\n{_INSTRUCTIONS}"
        ) from None
