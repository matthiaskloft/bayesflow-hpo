"""Does the refactored make_lc2st_validate_fn return what the duplicate did?

Step 3 of ``plan-joint-metric-path.md`` refactors L-C2ST off its own copy of
the validation loop and onto the shared joint-metric path. The claim that
this is a refactor rather than a rewrite is checkable, so it is checked
here: the pre-refactor implementation is reconstructed verbatim from git and
run against the same approximator, dataset and seed as the current one.

Pass the revision holding the PRE-refactor implementation:

    KERAS_BACKEND=torch python \
        docs/plans/check_lc2st_refactor_equivalence.py \
        --old-rev 6340216~1

The default is ``HEAD~1``, not ``HEAD``: reading ``HEAD`` compares the
working tree against itself on a clean checkout and prints ``EQUIVALENT``
without having compared anything. A check that cannot fail is worse than no
check, because it is quoted as evidence.

Result at the time of the refactor: bit-identical for a single parameter,
max abs diff 6.9e-18 for three. That residual is expected and is not noise
in the metric -- the duplicate pooled every (condition, parameter) row into
one mean, while the pipeline means over conditions per parameter and then
across parameters. With a balanced grid both equal the grand mean; only the
floating-point association order differs.

One case this does NOT cover: `aggregate_condition_rows` skips NaN per key,
so if a marginal metric returns NaN for some (condition, parameter) pairs,
pooling and two-stage averaging weight the survivors differently and the two
implementations genuinely disagree. The pipeline's per-parameter mean is the
better-defined of the two -- a NaN in one parameter no longer reweights the
others -- but it is a behaviour change, not an identity, and the number
above does not speak to it.

Requires scikit-learn.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np  # noqa: E402

from bayesflow_hpo.validation.data import ValidationDataset  # noqa: E402


class FakeApproximator:
    """Deterministic per-condition draws, so both implementations agree.

    The draws are a function of the condition's data, not of call order, so
    the old and new factories see identical input for the same condition
    however many times each is invoked.
    """

    def __init__(
        self, param_keys: list[str], n_sims: int, seed: int = 0
    ) -> None:
        self.param_keys = param_keys
        self.n_sims = n_sims
        self.seed = seed

    def sample(
        self, *, conditions: dict[str, Any], num_samples: int
    ) -> dict[str, np.ndarray]:
        # Deterministic in the condition, so both implementations see
        # identical draws for the same condition.
        key = float(np.asarray(conditions["x"]).sum())
        rng = np.random.default_rng(abs(hash((self.seed, round(key, 6)))) % 2**32)
        return {
            k: rng.normal(size=(self.n_sims, num_samples, 1))
            for k in self.param_keys
        }


def dataset(
    param_keys: list[str], n_conditions: int, n_sims: int
) -> ValidationDataset:
    """Build a validation dataset with one entry per condition."""
    rng = np.random.default_rng(1)
    sims = [
        {
            **{pk: rng.normal(size=(n_sims,)) for pk in param_keys},
            "x": rng.normal(size=(n_sims, 3)),
        }
        for _ in range(n_conditions)
    ]
    return ValidationDataset(
        simulations=sims,
        condition_labels=[{"c": i} for i in range(n_conditions)],
        param_keys=list(param_keys),
        data_keys=["x"],
        seed=0,
    )


def load_old_factory(old_rev: str) -> Callable[..., Any]:
    """Import the pre-refactor ``c2st.py`` as a standalone module.

    Parameters
    ----------
    old_rev
        Git revision holding the implementation to compare against. Must
        NOT resolve to the same content as the working tree, or the
        comparison is vacuous.

    Returns
    -------
    Callable
        That revision's ``make_lc2st_validate_fn``.

    Raises
    ------
    SystemExit
        If *old_rev* cannot be read, or if its ``c2st.py`` is byte-identical
        to the working tree's -- which would make every comparison below
        trivially pass.
    """
    spec = f"{old_rev}:src/bayesflow_hpo/validation/c2st.py"
    proc = subprocess.run(
        ["git", "show", spec], capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"Could not read {spec!r}: {proc.stderr.strip()}\n"
            "Pass --old-rev pointing at the commit BEFORE the refactor."
        )
    src = proc.stdout
    current = Path("src/bayesflow_hpo/validation/c2st.py").read_text(
        encoding="utf-8"
    )
    if src == current:
        raise SystemExit(
            f"{spec!r} is byte-identical to the working tree, so every "
            "comparison below would pass without comparing anything. Pass "
            "--old-rev pointing at the commit BEFORE the refactor."
        )
    mod = types.ModuleType("old_c2st")
    mod.__dict__["__file__"] = "old_c2st.py"
    # Registered BEFORE exec: @dataclass resolves its own module by name.
    sys.modules["old_c2st"] = mod
    exec(compile(src, "old_c2st.py", "exec"), mod.__dict__)
    return mod.make_lc2st_validate_fn


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old-rev",
        default="HEAD~1",
        help=(
            "Git revision holding the pre-refactor c2st.py. Default HEAD~1; "
            "HEAD would compare the working tree against itself."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    old_factory = load_old_factory(args.old_rev)
    from bayesflow_hpo.validation.c2st import make_lc2st_validate_fn as new_factory

    for param_keys in (["theta"], ["a", "b", "c"]):
        data = dataset(param_keys, n_conditions=3, n_sims=40)
        approximator = FakeApproximator(param_keys, 40)

        old = old_factory(base_metrics=["calibration_error", "nrmse"], seed=7)(
            approximator, data, 25
        )
        new = new_factory(base_metrics=["calibration_error", "nrmse"], seed=7)(
            approximator, data, 25
        )

        print(f"\n--- param_keys={param_keys} ---")
        keys = sorted(set(old) | set(new))
        worst = 0.0
        for k in keys:
            o, n = old.get(k), new.get(k)
            if o is None or n is None:
                print(f"  {k:24s} old={o!r:>12} new={n!r:>12}   KEY MISMATCH")
                continue
            d = abs(float(o) - float(n))
            worst = max(worst, d)
            flag = "" if d < 1e-9 else "   <-- DIFFERS"
            print(f"  {k:24s} old={float(o):12.8f} new={float(n):12.8f}{flag}")
        print(f"  max abs diff: {worst:.3e}")
        assert set(old) == set(new), "summary key sets differ"
        assert worst < 1e-9, "values differ"

    print("\nEQUIVALENT")


if __name__ == "__main__":
    main()
