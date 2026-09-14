"""Per-condition cost of TARP, for bayesflow-hpo issue #82 open question 4.

Times `compute_tarp_coverage` (bayesflow-irt, commit ffc68d5) on synthetic
arrays of the shapes the hpo validation pipeline would hand it. Pure numpy;
no GPU, no approximator. Reports milliseconds per condition, which multiplies
by the study's condition count to give the per-trial cost.
"""

from __future__ import annotations

import importlib.util
import platform
import sys
import time

import numpy as np

HERE = __file__.rsplit("\\", 1)[0]
spec = importlib.util.spec_from_file_location("irt_sbc", HERE + "\\irt_sbc.py")
assert spec is not None and spec.loader is not None
irt_sbc = importlib.util.module_from_spec(spec)
sys.modules["irt_sbc"] = irt_sbc
spec.loader.exec_module(irt_sbc)
compute_tarp_coverage = irt_sbc.compute_tarp_coverage


def timeit(fn, repeats: int = 3) -> float:
    """Best-of-`repeats` wall time in milliseconds."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def main() -> None:
    rng = np.random.default_rng(0)
    print(f"python {platform.python_version()}  numpy {np.__version__}")
    print(f"{platform.processor() or platform.machine()}\n")

    print(f"{'n_sims':>7} {'n_draws':>8} {'n_params':>9} {'res':>4} "
          f"{'ms/cond':>9} {'tarp_error':>11}")
    print("-" * 54)

    grid = [
        # (n_sims, n_draws, n_params, resolution)
        (100, 100, 2, 20),
        (100, 1000, 2, 20),
        (500, 1000, 2, 20),
        (100, 1000, 15, 20),
        (500, 1000, 15, 20),
        (500, 1000, 60, 20),
        (1000, 1000, 60, 20),
        (500, 1000, 15, 100),
    ]

    for n_sims, n_draws, n_params, resolution in grid:
        true_values = rng.normal(size=(n_sims, n_params))
        draws = true_values[:, None, :] + rng.normal(
            size=(n_sims, n_draws, n_params)
        )
        ref = rng.normal(size=(n_sims, n_params))

        out: dict = {}

        def call(d=draws, t=true_values, r=ref, res=resolution, o=out):
            o.update(
                compute_tarp_coverage(
                    d, t, reference_points=r, resolution=res, seed=0
                )
            )

        ms = timeit(call)
        print(f"{n_sims:>7} {n_draws:>8} {n_params:>9} {resolution:>4} "
              f"{ms:>9.1f} {out['tarp_error']:>11.4f}")

    # --- Scaling check: is it linear in n_draws, as the O(n_sims * n_draws *
    # n_params) distance count predicts? ---
    print("\nscaling in n_draws (n_sims=500, n_params=15, res=20):")
    base = None
    for n_draws in (125, 250, 500, 1000, 2000):
        true_values = rng.normal(size=(500, 15))
        draws = true_values[:, None, :] + rng.normal(size=(500, n_draws, 15))
        ref = rng.normal(size=(500, 15))
        ms = timeit(
            lambda d=draws, t=true_values, r=ref: compute_tarp_coverage(
                d, t, reference_points=r, resolution=20, seed=0
            )
        )
        base = base or ms
        print(f"  n_draws={n_draws:>5}  {ms:>8.1f} ms   x{ms / base:>5.2f}")


if __name__ == "__main__":
    main()
