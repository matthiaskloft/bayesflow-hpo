"""Does the refactored make_lc2st_validate_fn return what the duplicate did?

Step 3 of ``plan-joint-metric-path.md`` refactors L-C2ST off its own copy of
the validation loop and onto the shared joint-metric path. The claim that
this is a refactor rather than a rewrite is checkable, so it is checked
here: the pre-refactor implementation is reconstructed verbatim from git and
run against the same approximator, dataset and seed as the current one.

Run it against the commit that introduced the refactor:

    git show <refactor-commit>~1:src/bayesflow_hpo/validation/c2st.py

is what ``load_old_factory`` reads, via ``HEAD`` -- so check the refactor out
and run this with ``HEAD`` at its parent, or edit the ref below.

    KERAS_BACKEND=torch python docs/plans/check_lc2st_refactor_equivalence.py

Result at the time of the refactor: bit-identical for a single parameter,
max abs diff 6.9e-18 for three. That residual is expected and is not noise
in the metric -- the duplicate pooled every (condition, parameter) row into
one mean, while the pipeline means over conditions per parameter and then
across parameters. With a balanced grid both equal the grand mean; only the
floating-point association order differs.

Requires scikit-learn.
"""

from __future__ import annotations

import os
import subprocess
import sys
import types

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np  # noqa: E402

from bayesflow_hpo.validation.data import ValidationDataset  # noqa: E402


class FakeApproximator:
    def __init__(self, param_keys, n_sims, seed=0):
        self.param_keys = param_keys
        self.n_sims = n_sims
        self.seed = seed

    def sample(self, *, conditions, num_samples):
        # Deterministic in the condition, so both implementations see
        # identical draws for the same condition.
        key = float(np.asarray(conditions["x"]).sum())
        rng = np.random.default_rng(abs(hash((self.seed, round(key, 6)))) % 2**32)
        return {
            k: rng.normal(size=(self.n_sims, num_samples, 1))
            for k in self.param_keys
        }


def dataset(param_keys, n_conditions, n_sims):
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


def load_old_factory():
    """Import the pre-refactor c2st.py as a standalone module."""
    src = subprocess.run(
        ["git", "show", "HEAD:src/bayesflow_hpo/validation/c2st.py"],
        capture_output=True, text=True, check=True,
    ).stdout
    mod = types.ModuleType("old_c2st")
    mod.__dict__["__file__"] = "old_c2st.py"
    # Registered BEFORE exec: @dataclass resolves its own module by name.
    sys.modules["old_c2st"] = mod
    exec(compile(src, "old_c2st.py", "exec"), mod.__dict__)
    return mod.make_lc2st_validate_fn


def main() -> None:
    old_factory = load_old_factory()
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
