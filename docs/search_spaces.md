# Search Spaces

Search spaces define which hyperparameters Optuna explores and how sampled values map to BayesFlow network instances.

## Dimension Types

All dimensions are defined in `search_spaces/base.py`:

| Type | Fields | Example |
|------|--------|---------|
| `IntDimension` | name, low, high, step, log, constant | `IntDimension("depth", 2, 12)` |
| `FloatDimension` | name, low, high, log, constant | `FloatDimension("dropout", 0.0, 0.3)` |
| `CategoricalDimension` | name, choices, constant | `CategoricalDimension("activation", ["relu", "silu"])` |

The `constant` field controls whether a dimension is tuned:
- `constant` not set (default `_UNSET`) — dimension is tunable, Optuna samples from range/choices
- `constant=<value>` — dimension is fixed at the given value, not tuned by Optuna

This replaces the previous `enabled`/`include_optional` pattern. Dimensions that default to a BayesFlow default value use `constant=<bf_default>` so they are not tuned unless the user overrides them.

## Inference Network Spaces

### CouplingFlowSpace

Coupling-based normalizing flow (BayesFlow `CouplingFlow`).

| Dimension | Type | Range | Tuned | Constant |
|-----------|------|-------|-------|----------|
| `cf_depth` | int | [2, 12] | yes | — |
| `cf_subnet_width` | int | [32, 256], log | yes | — |
| `cf_subnet_depth` | int | [1, 3] | yes | — |
| `cf_dropout` | float | [0.0, 0.3] | yes | — |
| `cf_activation` | cat | silu, relu, mish | yes | — |
| `cf_transform` | cat | affine, spline | no | `"affine"` |
| `cf_permutation` | cat | random, orthogonal | no | `"random"` |
| `cf_use_actnorm` | cat | True, False | no | `True` |

### FlowMatchingSpace

Continuous normalizing flow via flow matching (`FlowMatching`).

| Dimension | Type | Range | Tuned | Constant |
|-----------|------|-------|-------|----------|
| `fm_subnet_width` | int | [32, 256], log | yes | — |
| `fm_subnet_depth` | int | [1, 6] | yes | — |
| `fm_dropout` | float | [0.0, 0.2] | yes | — |
| `fm_activation` | cat | mish, silu | yes | — |
| `fm_use_ot` | cat | True, False | no | `False` |
| `fm_time_alpha` | float | [0.0, 2.0] | no | `0.0` |

**Note:** `loss_fn="mse"` is hardcoded in `build()`.

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
| `st_mlp_width` | int | [64, 512], log | no | `2 * embed_dim` |
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
| `tst_mlp_width` | int | [64, 512], log | no | `2 * embed_dim` |
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
| `initial_lr` | float | [1e-4, 5e-3], log | yes | — |
| `batch_size` | int | [32, 1024], step=32 | no | `256` |
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
