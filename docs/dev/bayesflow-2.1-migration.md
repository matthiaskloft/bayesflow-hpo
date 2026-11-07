# BayesFlow 2.1 migration (bayesflow-hpo 0.8.0, issue #152)

Read from the installed BayesFlow 2.0.8 and 2.1.0 wheels (2026-11-07). Paths
are relative to `site-packages/bayesflow/`. Companion audit for the IRT
package: `bayesflow-irt/docs/architecture/bayesflow-2.1-audit.md`.

## Baseline

The unmodified 0.7.0 suite on 2.1.0 (Python 3.12, keras 3.15.1, torch 2.15
CPU): 2 failed, 1580 passed, 9 skipped
(`test_vector_params.py::test_scalar_params_unaffected[shape1]`,
`::test_width_may_vary_across_conditions`). The suite does not catch the
silent kwarg drops below, because it mostly passes arguments to real
constructors that accept `**kwargs`. `tests/test_bayesflow_21_compat.py` now
does.

## What BayesFlow 2.1 does that matters here

- Unknown network kwargs are discarded, not rejected:
  `utils/dict_utils.py:83-86` (`layer_kwargs` keeps `dtype`, `name`,
  `trainable`). `SummaryNetwork.__init__` (`networks/summary/summary_network.py:51`)
  and `TimeMLP` (`networks/subnets/mlp/time_mlp.py:74`) route through it.
- `SetTransformer`, `TimeSeriesTransformer`, `FusionTransformer`
  (`networks/summary/transformers/*.py`) take `expansion_factor`,
  `glu_variant`, `kernel_initializer="orthogonal"`, `use_bias=False`,
  `gate_attention`, `gate_ffn`; no `mlp_widths`/`mlp_depths`.
- `DeepSet` (`networks/summary/deep_set/deep_set.py:70-88`) takes `embed_dim`,
  `depth`, `mlp_widths`, `inner_pooling`, `num_heads`, `num_seeds`; no
  `mlp_widths_*`, `output_pooling`, `spectral_normalization`.
- `TimeMLP` (`subnets/mlp/time_mlp.py:57-72`) has no `spectral_normalization`.
  `FlowMatching` merges `TIME_MLP_DEFAULTS` (`networks/defaults.py`:
  `fourier_scale=16`, `film_use_gamma=True`, `persistent_conditioning=True`,
  `dropout=0`) under `subnet_kwargs` (`inference/flow_matching/flow_matching.py:129-130`).
- `CouplingFlow` lives in `networks/inference/coupling/`; the subnet default
  `COUPLING_MLP_DEFAULTS` uses `kernel_initializer="orthogonal"`
  (`networks/defaults.py`), was `glorot_uniform` in 2.0.8.
- `diagnostics.metrics.root_mean_squared_error`: default `normalize` is
  `"range"` in 2.0.8 and `"prior"` in 2.1 (random bootstrap), and the
  reduction changed (see CHANGELOG).
- `FlowMatching.INTEGRATE_DEFAULT_CONFIG` is gone; the dict is
  `defaults.FLOW_MATCHING_INTEGRATE_DEFAULTS`.

## What changed in this package

| File | Change |
|---|---|
| `pyproject.toml` | floors (bayesflow, keras, numpy, scipy, pandas, matplotlib), `requires-python`, classifiers, ruff `py312`, mypy `3.12`; ruff ignores UP040/UP047 (style-only PEP 695 rules that appear with py312) |
| `.github/workflows/*.yml` | lint/types/docs jobs and release on 3.12; test matrix 3.12, 3.13 |
| `search_spaces/summary/set_transformer.py:62`, `time_series_transformer.py:52`, `fusion_transformer.py:56` | `*_mlp_width`/`*_mlp_depth` -> constant `*_expansion_factor` (4.0) |
| `search_spaces/summary/deep_set.py:105-106` | `embed_dim` (new `ds_embed_dim`), `mlp_widths=(width, width)`; dropped `ds_spectral_normalization`, `ds_output_pooling` |
| `search_spaces/inference/flow_matching.py:20,41` | defaults from `TIME_MLP_DEFAULTS` / `FLOW_MATCHING_INTEGRATE_DEFAULTS`; dropped `fm_spectral_normalization` |
| `optimization/constraints.py:63-82` | transformer parameter estimate follows the 2.1 block |
| `validation/registry.py:1001` | `rmse` passes `normalize=None` |
| `tests/test_bayesflow_21_compat.py` | signature guard + build smoke test for all spaces |

Not changed because nothing needed it: `Adapter.add_transform` and the moved
`networks/coupling_flow`, `networks/mlp` module paths are not used anywhere in
`src/` (the package goes through `bf.networks.*`). `CouplingFlowSpace` sets no
`kernel_initializer`, so it inherits `orthogonal`.

## What the package now builds (defaults)

- Transformers: pre-LN RMSNorm blocks, GLU FFN of width
  `ceil64(int(embed_dim * expansion_factor * 2/3))`, bias-free, zero-initialized
  residual gates. `expansion_factor` is constant 4.0 (not tuned).
- DeepSet: `embed_dim=64`, equivariant MLP `(width, width)`, PMA + linear
  projection to `summary_dim`.

## Checkpoint compatibility (nothing was deleted or retrained)

- Any `weights.weights.h5` written by `CheckpointPool`
  (`optimization/checkpoint_pool.py:496`) or `.keras` written by
  `results/export.py:47` under 2.0.x fails to load on 2.1 (different variable
  sets). The pool and the exporter store no BayesFlow version
  (`get_workflow_metadata` records `bayesflow.__version__` only in the export
  sidecar), so a stale pool fails inside `load_weights`, not with a clean
  message. Related: issue #129.
- Optuna studies from 0.7.0 contain `st_mlp_width` etc.; resuming them against
  0.8.0 mixes incompatible parameter sets. Start new studies.

## Open points for review

1. `rmse` semantics (above). Decide whether a separate `rmse` over posterior
   means is wanted; nothing in this change adds one.
2. `nrmse` still uses BayesFlow's `normalize="range"`, whose reduction differs
   across 2.0.8/2.1.
3. `expansion_factor` is constant; making it a tuned dimension is a one-line
   change per space but widens the search.
4. No clean refusal for loading a 2.0.x checkpoint; candidate for a follow-up.
