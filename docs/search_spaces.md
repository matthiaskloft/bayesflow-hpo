# Search Spaces

Search spaces define which hyperparameters Optuna explores and how sampled values map to BayesFlow network instances.

## Dimension Types

All dimensions are defined in `search_spaces/base.py`:

| Type | Fields | Example |
|------|--------|---------|
| `IntDimension` | name, low, high, step, log, constant | `IntDimension("depth", 2, 12)` |
| `FloatDimension` | name, low, high, log, constant | `FloatDimension("dropout", 0.0, 0.3)` |
| `CategoricalDimension` | name, choices, constant | `CategoricalDimension("activation", ["relu", "silu"])` |
| `DerivedDimension` | name, derive | `DerivedDimension("num_batches", lambda p: ...)` |

The `constant` field controls whether a dimension is tuned:
- `constant` not set (default `_UNSET`) — dimension is tunable, Optuna samples from range/choices
- `constant=<value>` — dimension is fixed at the given value, not tuned by Optuna

This replaces the previous `enabled`/`include_optional` pattern. Dimensions that default to a BayesFlow default value use `constant=<bf_default>` so they are not tuned unless the user overrides them.

`DerivedDimension` is evaluated after all sampled and constant dimensions in
the same search space. It supports exact resource couplings without presenting
redundant coordinates to the sampler. For example, a training-space subclass
can fix the number of online simulations while tuning batch size:

This supports the workload-aware joint-tuning rationale of Shallue et al.
(2019); see [References](references.md).

```python
num_batches = hpo.DerivedDimension(
    "num_batches",
    lambda p: p["simulation_budget"] // (p["batch_size"] * p["epochs"]),
)
```

Derived values are included in the `hparams` passed to the builder and
`train_fn`, but are not recorded as independently sampled Optuna parameters.

## Inference Network Spaces

### CouplingFlowSpace

Coupling-based normalizing flow (BayesFlow `CouplingFlow`).

| Dimension | Type | Range | Tuned | Constant |
|-----------|------|-------|-------|----------|
| `cf_depth` | int | [2, 8] | yes | — |
| `cf_subnet_width` | int | [32, 256], log | yes | — |
| `cf_subnet_depth` | int | [1, 3] | yes | — |
| `cf_dropout` | float | [0.0, 0.3] | yes | — |
| `cf_activation` | cat | silu, relu, mish | no | `"silu"` |
| `cf_transform` | cat | affine, spline | no | `"affine"` |
| `cf_permutation` | cat | random, orthogonal | no | `"random"` |
| `cf_use_actnorm` | cat | True, False | no | `True` |

### FlowMatchingSpace

Continuous normalizing flow via flow matching (`FlowMatching`).

| Dimension | Type | Range | Tuned | Constant |
|-----------|------|-------|-------|----------|
| `fm_subnet_width` | int | [32, 256], step 32 | yes | — |
| `fm_subnet_depth` | int | [1, 6] | yes | — |
| `fm_dropout` | float | [0.0, 0.2] | yes | — |
| `fm_activation` | cat | — | no | `"mish"` |
| `fm_use_optimal_transport` | cat | — | no | `False` |
| `fm_time_power_law_alpha` | float | — | no | `0.0` |
| `fm_time_embedding_dim` | int | — | no | `32` |
| `fm_integrate_method` | cat | — | no | `"tsit5"` |
| `fm_integrate_steps` | cat | — | no | `"adaptive"` |
| `fm_merge` | cat | — | no | `"concat"` |
| `fm_norm` | cat | — | no | `"layer"` |
| `fm_residual` | cat | — | no | `True` |
| `fm_spectral_normalization` | cat | — | no | `False` |
| `fm_kernel_initializer` | cat | — | no | `"he_normal"` |

Untuned constants are synchronized to BayesFlow defaults at runtime
(`bf.networks.TimeMLP` defaults plus
`bf.networks.FlowMatching.INTEGRATE_DEFAULT_CONFIG`). The values in this
table reflect the current BayesFlow release used by this project.

`build()` maps flat `fm_*` params to:
- `subnet_kwargs={widths, dropout, activation, time_embedding_dim, merge, norm, residual, spectral_normalization, kernel_initializer}`
- `integrate_kwargs={method, steps}`

`fm_integrate_steps` usually has the strongest inference-time effect because it multiplies velocity-network evaluations during ODE sampling.

Profile constructors:
- `FlowMatchingSpace.fast()` for lower-latency solver/subnet ranges
- `FlowMatchingSpace.balanced()` for mixed speed/quality exploration
- `FlowMatchingSpace.quality()` for larger networks and finer solver settings
- `FlowMatchingSpace.preset("<name>")` with `default|fast|balanced|quality`

Speed-sensitive override example:

```python
import bayesflow_hpo as hpo

inference_space = hpo.FlowMatchingSpace(
    subnet_width=hpo.IntDimension("fm_subnet_width", 32, 128, step=32),
    subnet_depth=hpo.IntDimension("fm_subnet_depth", 1, 3),
    integrate_method=hpo.CategoricalDimension("fm_integrate_method", ["euler", "tsit5"]),
    integrate_steps=hpo.CategoricalDimension("fm_integrate_steps", [16, 24, 32]),
    merge=hpo.CategoricalDimension("fm_merge", ["add", "concat"]),
    norm=hpo.CategoricalDimension("fm_norm", [None, "layer"]),
)
```

### DiffusionModelSpace

Score-based diffusion model (`DiffusionModel`).

| Dimension | Type | Range | Tuned | Constant |
|-----------|------|-------|-------|----------|
| `dm_subnet_width` | int | [32, 256], log | yes | — |
| `dm_subnet_depth` | int | [1, 6] | yes | — |
| `dm_dropout` | float | [0.0, 0.2] | yes | — |
| `dm_activation` | cat | mish, silu | yes | — |
| `dm_noise_schedule` | cat | edm, cosine | no | `"edm"` |
| `dm_prediction_type` | cat | F, velocity, noise, x | no | `"F"` |

### ConsistencyModelSpace

Consistency model (`ConsistencyModel`).

| Dimension | Type | Range | Tuned | Constant |
|-----------|------|-------|-------|----------|
| `cm_subnet_width` | int | [32, 256], log | yes | — |
| `cm_subnet_depth` | int | [1, 6] | yes | — |
| `cm_dropout` | float | [0.0, 0.2] | yes | — |
| `cm_max_time` | int | [50, 500] | no | `200` |
| `cm_sigma2` | float | [0.1, 2.0] | no | `0.5` |
| `cm_s0` | int | [2, 30] | no | `2` |
| `cm_s1` | int | [20, 100] | no | `50` |

**Note:** `ConsistencyModelSpace` accepts `epochs` and `num_batches` in its constructor to compute `total_steps` for the consistency model schedule.

### StableConsistencyModelSpace

Stable variant of the consistency model (`StableConsistencyModel`).

| Dimension | Type | Range | Tuned | Constant |
|-----------|------|-------|-------|----------|
| `scm_subnet_width` | int | [32, 256], log | yes | — |
| `scm_subnet_depth` | int | [1, 6] | yes | — |
| `scm_dropout` | float | [0.0, 0.2] | yes | — |
| `scm_sigma` | float | [0.1, 2.0] | no | `0.5` |

## Summary Network Spaces

### DeepSetSpace

Permutation-invariant summary via DeepSets (`DeepSet`).

| Dimension | Type | Range | Tuned | Constant |
|-----------|------|-------|-------|----------|
| `ds_summary_dim` | int | [4, 64], step 4 | yes | — |
| `ds_depth` | int | [1, 4] | yes | — |
| `ds_width` | int | [32, 256], log | yes | — |
| `ds_dropout` | float | [0.0, 0.3] | yes | — |
| `ds_activation` | cat | silu, mish | no | `"silu"` |
| `ds_spectral_norm` | cat | True, False | no | `False` |

**Note:** `inner_pooling="mean"` and `output_pooling="mean"` are hardcoded in `build()`.

### SetTransformerSpace

Attention-based set summary (`SetTransformer`).

| Dimension | Type | Range | Tuned | Constant |
|-----------|------|-------|-------|----------|
| `st_summary_dim` | int | [8, 64], log | yes | — |
| `st_embed_dim` | int | [32, 256], log | yes | — |
| `st_num_heads` | cat | 1, 2, 4, 8 | yes | — |
| `st_num_layers` | int | [1, 4] | yes | — |
| `st_dropout` | float | [0.0, 0.3] | yes | — |
| `st_mlp_width` | int | [64, 512], log | no | `128` |
| `st_mlp_depth` | int | [1, 4] | no | `2` |
| `st_num_inducing` | int | [8, 64], step 8 | no | `None` |

### TimeSeriesNetworkSpace

CNN + RNN temporal summary (`TimeSeriesNetwork`).

| Dimension | Type | Range | Tuned | Constant |
|-----------|------|-------|-------|----------|
| `tsn_summary_dim` | int | [8, 64], log | yes | — |
| `tsn_recurrent_dim` | int | [32, 256], log | yes | — |
| `tsn_filters` | int | [16, 128], log | yes | — |
| `tsn_dropout` | float | [0.0, 0.3] | yes | — |
| `tsn_recurrent_type` | cat | gru, lstm | no | `"gru"` |
| `tsn_bidirectional` | cat | True, False | no | `True` |
| `tsn_skip_steps` | int | [1, 8] | no | `1` |

### TimeSeriesTransformerSpace

Transformer-based temporal summary (`TimeSeriesTransformer`).

| Dimension | Type | Range | Tuned | Constant |
|-----------|------|-------|-------|----------|
| `tst_summary_dim` | int | [8, 64], log | yes | — |
| `tst_embed_dim` | int | [32, 256], log | yes | — |
| `tst_num_heads` | cat | 1, 2, 4, 8 | yes | — |
| `tst_num_layers` | int | [1, 4] | yes | — |
| `tst_dropout` | float | [0.0, 0.3] | yes | — |
| `tst_mlp_width` | int | [64, 512], log | no | `128` |
| `tst_time_embed` | cat | time2vec, lstm, gru | no | `"time2vec"` |

### FusionTransformerSpace

Cross-attention fusion summary (`FusionTransformer`).

| Dimension | Type | Range | Tuned | Constant |
|-----------|------|-------|-------|----------|
| `ft_summary_dim` | int | [8, 64], log | yes | — |
| `ft_embed_dim` | int | [32, 256], log | yes | — |
| `ft_num_heads` | cat | 1, 2, 4, 8 | yes | — |
| `ft_num_layers` | int | [1, 4] | yes | — |
| `ft_template_dim` | int | [32, 256], log | yes | — |
| `ft_dropout` | float | [0.0, 0.3] | yes | — |
| `ft_mlp_width` | int | [64, 512] | no | `128` |
| `ft_mlp_depth` | int | [1, 4] | no | `2` |
| `ft_bidirectional` | cat | True, False | no | `True` |
| `ft_template_type` | cat | lstm, gru | no | `"lstm"` |

## Training Space

`TrainingSpace` controls optimizer hyperparameters:

| Dimension | Type | Range | Tuned | Constant |
|-----------|------|-------|-------|----------|
| `initial_lr` | float | [1e-4, 1e-2], log | yes | — |
| `batch_size` | int | [32, 256], step=32 | yes | — |
| `decay_rate` | float | [0.8, 0.99] | no | `0.95` |

When a dimension has `constant` set, the constant value is used directly. To make a constant dimension tunable, set `constant=_UNSET` or create a new dimension without a constant.

## Composite Spaces

### CompositeSearchSpace

Combines inference, summary (optional), and training spaces into a single searchable unit:

```python
space = CompositeSearchSpace(
    inference_space=CouplingFlowSpace(),
    summary_space=DeepSetSpace(),
    training_space=TrainingSpace(),
)
```

`sample(trial)` merges parameters from all sub-spaces into a flat dict.

### NetworkSelectionSpace

Lets Optuna choose among multiple inference network types:

```python
space = NetworkSelectionSpace(candidates={
    "coupling_flow": CouplingFlowSpace(),
    "flow_matching": FlowMatchingSpace(),
})
```

Adds a `inference_network_type` categorical to the trial and delegates to the chosen space.

### SummarySelectionSpace

Same pattern for summary networks — adds `summary_network_type` categorical.

## Custom Network Registration

Register a custom network type so it can be used by name in selection spaces:

```python
from bayesflow_hpo import register_custom_inference_network

register_custom_inference_network(
    name="my_custom_flow",
    space_factory=lambda: MyCustomFlowSpace(),
    builder=my_custom_builder_fn,   # optional
    aliases=["mcf"],
)
```

## Registry Aliases

Built-in short aliases for convenience:

| Alias | Full Name |
|-------|-----------|
| `cf` | `coupling_flow` |
| `fm` | `flow_matching` |
| `dm` | `diffusion_model` |
| `cm` | `consistency_model` |
| `scm` | `stable_consistency_model` |
| `ds` | `deep_set` |
| `st` | `set_transformer` |
| `tsn` | `time_series_network` |
| `tst` | `time_series_transformer` |
| `ft` | `fusion_transformer` |
