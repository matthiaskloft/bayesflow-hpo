"""Matched-shape ratio: TARP cost vs real inference cost, same machine.

Answers the form #82 asked the cost question in. Both numerator and
denominator are measured here at identical (n_sims, n_draws, n_params), so no
cross-machine or cross-shape extrapolation enters the ratio.

Needs a CUDA device and bayesflow_hpo importable. TARP is not a dependency of
this package; see `_tarp_source.py`. From a clean checkout::

    git -C <bayesflow-irt> show \
        ffc68d5...:src/bayesflow_irt/sbc.py > /tmp/irt_sbc.py
    KERAS_BACKEND=torch python docs/plans/bench_inference_ratio.py \
        --tarp-source /tmp/irt_sbc.py

`--mem-fraction` caps this process's share of the device so that an overrun
fails the benchmark rather than an unrelated job sharing the card.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import bayesflow as bf
import keras
import numpy as np
import torch

from bayesflow_hpo.validation.inference import make_bayesflow_infer_fn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _tarp_source import (  # noqa: E402
    add_tarp_argument,
    load_compute_tarp_coverage,
)

parser = argparse.ArgumentParser(description=__doc__)
add_tarp_argument(parser)
parser.add_argument("--mem-fraction", type=float, default=0.20)
args = parser.parse_args()
compute_tarp_coverage = load_compute_tarp_coverage(args.tarp_source)

P, NOBS = 15, 50
SHAPES = [(100, 200), (200, 200), (100, 400), (150, 400)]
if not torch.cuda.is_available():
    raise SystemExit("No CUDA device: this benchmark measures GPU inference.")
torch.cuda.set_per_process_memory_fraction(args.mem_fraction, 0)

def prior_fn():
    return {"theta": np.random.normal(0, 1, size=(P,)).astype("float32")}
def like_fn(theta):
    x = np.random.normal(theta[None, :], 1.0, size=(NOBS, P))
    return {"x": x.astype("float32")}

sim = bf.simulators.make_simulator([prior_fn, like_fn])
adapter = (
    bf.Adapter()
    .as_set(["x"])
    .rename("theta", "inference_variables")
    .concatenate(["x"], into="summary_variables", axis=-1)
)
approx = bf.ContinuousApproximator(
    adapter=adapter,
    inference_network=bf.networks.FlowMatching(subnet_kwargs={"widths": (128, 128)}),
    summary_network=bf.networks.DeepSet(summary_dim=32, depth=2))
approx.compile(optimizer=keras.optimizers.Adam(1e-3))
approx.fit(simulator=sim, epochs=1, num_batches=8, batch_size=32, verbose=0)
torch.cuda.synchronize()

print(f"\ndevice {torch.cuda.get_device_name(0)}  n_params={P} n_obs={NOBS}")
print(
    f"{'n_sims':>7}{'n_draws':>9}{'infer ms':>11}"
    f"{'TARP ms':>10}{'TARP % of infer':>17}"
)
print("-" * 54)
for n_sims, n_draws in SHAPES:
    torch.cuda.empty_cache()
    batch = {k: np.asarray(v) for k, v in sim.sample((n_sims,)).items()}
    fn = make_bayesflow_infer_fn(approx, ["theta"], ["x"], set(batch))
    fn(batch, n_draws)                                   # warm-up
    torch.cuda.synchronize()
    ts = []
    for _ in range(3):
        t = time.perf_counter()
        draws = fn(batch, n_draws)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    infer_ms = min(ts)

    draws = np.asarray(draws)
    truth = np.asarray(batch["theta"])
    ref = np.random.default_rng(0).normal(size=truth.shape)
    tt = []
    for _ in range(3):
        t = time.perf_counter()
        compute_tarp_coverage(
            draws, truth, reference_points=ref, resolution=20, seed=0
        )
        tt.append((time.perf_counter() - t) * 1e3)
    tarp_ms = min(tt)
    pct = tarp_ms / infer_ms * 100
    print(f"{n_sims:>7}{n_draws:>9}{infer_ms:>11.1f}{tarp_ms:>10.1f}{pct:>16.2f}%")
print(f"\npeak GPU alloc {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
