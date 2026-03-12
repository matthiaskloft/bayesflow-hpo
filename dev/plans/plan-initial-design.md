# bayesflow-hpo: Generic Hyperparameter Optimization for BayesFlow

**Date**: 2026-03-03
**Status**: Planning
**Supersedes**: FUTURE_IMPROVEMENTS.md §3 ("Decouple Bayesian Optimization Infrastructure")

---

## 1. Motivation

`bayesflow_rct/core/` already contains a battle-tested Optuna multi-objective HPO pipeline,
but its search space is hardcoded to **DeepSet + CouplingFlow** for the ANCOVA use case.
BayesFlow 2.x ships **6 inference networks** and **6 summary networks**, each with 10-20
tunable hyperparameters. Researchers adopting BayesFlow need an easy way to find the optimal
architecture for *their* specific simulator — without reimplementing the optimization loop,
validation pipeline, and results analysis from scratch.

**bayesflow-hpo** extracts the generic infrastructure into a reusable package and adds
complete search space definitions for every BayesFlow network type.

---

## 2. Scope

### In scope

- Generic search space definitions for all BayesFlow 2.x network types
- Network builders that construct BayesFlow networks from Optuna trial suggestions
- Multi-objective Optuna study management (calibration error vs. model size)
- Training loop with early stopping, pruning, and GPU memory management
- SBC validation pipeline (generic, model-agnostic)
- Results analysis: Pareto front, parameter importance, trial summaries
- Clean user-facing API: researchers define simulator + adapter, HPO does the rest
- Adapting `bayesflow_rct` to depend on `bayesflow-hpo`

### Out of scope

- CalibrationMixin composability (FUTURE_IMPROVEMENTS.md §2 — separate effort)
- Custom domain-specific summary networks (e.g., EquivariantIRTSummary from bayesflow-IRT).
  The package covers standard BayesFlow networks; users can register custom networks.
- Web dashboard (optional, stays in bayesflow_rct if needed)

---

## 3. Package Structure

```
bayesflow-hpo/
├── pyproject.toml
├── CLAUDE.md
├── src/
│   └── bayesflow_hpo/
│       ├── __init__.py              # Public API re-exports
│       │
│       ├── search_spaces/           # Per-network-type search space definitions
│       │   ├── __init__.py
│       │   ├── base.py              # SearchSpace protocol, SearchDimension dataclass
│       │   ├── inference/
│       │   │   ├── __init__.py
│       │   │   ├── coupling_flow.py
│       │   │   ├── flow_matching.py
│       │   │   ├── diffusion.py
│       │   │   ├── consistency.py
│       │   │   └── stable_consistency.py
│       │   ├── summary/
│       │   │   ├── __init__.py
│       │   │   ├── deep_set.py
│       │   │   ├── set_transformer.py
│       │   │   ├── time_series_network.py
│       │   │   ├── time_series_transformer.py
│       │   │   └── fusion_transformer.py
│       │   ├── training.py          # LR, batch size, optimizer search dims
│       │   ├── composite.py         # Combines inference + summary + training spaces
│       │   └── registry.py          # Discover/register search spaces by network name
│       │
│       ├── builders/                # Construct BayesFlow networks from param dicts
│       │   ├── __init__.py
│       │   ├── inference.py         # build_inference_network(params) → InferenceNetwork
│       │   ├── summary.py           # build_summary_network(params) → SummaryNetwork
│       │   ├── workflow.py          # build_workflow(params, simulator, adapter) → BasicWorkflow
│       │   └── registry.py          # Network type → builder function mapping
│       │
│       ├── optimization/            # Optuna integration
│       │   ├── __init__.py
│       │   ├── study.py             # create_study(), resume_study()
│       │   ├── objective.py         # GenericObjective: trial → (cal_error, param_score)
│       │   ├── callbacks.py         # OptunaReportCallback, MovingAverageEarlyStopping
│       │   ├── sampling.py          # sample_hyperparameters(trial, space) → dict
│       │   ├── constraints.py       # Parameter budget, memory estimation
│       │   └── cleanup.py           # GPU cache cleanup between trials
│       │
│       ├── validation/              # SBC validation pipeline
│       │   ├── __init__.py
│       │   ├── data.py              # generate_validation_dataset(), ValidationDataset
│       │   ├── pipeline.py          # run_validation_pipeline()
│       │   ├── metrics.py           # compute_batch_metrics(), aggregate_metrics()
│       │   ├── sbc_tests.py         # KS, chi-squared, C2ST uniformity tests
│       │   └── inference.py         # make_bayesflow_infer_fn()
│       │
│       ├── results/                 # Post-optimization analysis
│       │   ├── __init__.py
│       │   ├── extraction.py        # get_pareto_trials(), trials_to_dataframe()
│       │   ├── visualization.py     # plot_pareto_front(), plot_param_importance()
│       │   └── export.py            # save/load workflow with metadata
│       │
│       ├── objectives.py            # get_param_count(), normalize_param_count()
│       └── utils.py                 # loguniform sampling, type helpers
│
├── tests/
│   ├── test_search_spaces/
│   │   ├── test_coupling_flow_space.py
│   │   ├── test_flow_matching_space.py
│   │   ├── test_deep_set_space.py
│   │   ├── test_set_transformer_space.py
│   │   └── ...
│   ├── test_builders/
│   │   ├── test_inference_builders.py
│   │   └── test_summary_builders.py
│   ├── test_optimization/
│   │   ├── test_objective.py
│   │   ├── test_study.py
│   │   └── test_sampling.py
│   ├── test_validation/
│   │   ├── test_pipeline.py
│   │   ├── test_metrics.py
│   │   └── test_sbc_tests.py
│   └── test_integration.py          # End-to-end: simple Gaussian simulator → HPO
│
└── examples/
    ├── quickstart.ipynb             # Minimal example with Gaussian model
    ├── custom_summary_network.ipynb # Registering a custom network
    └── multi_objective.ipynb        # Pareto front analysis
```

---

## 4. Core Design

### 4.1 Search Space Protocol

Every BayesFlow network type gets a `SearchSpace` that declaratively describes its tunable
dimensions. The user never writes Optuna `trial.suggest_*` calls manually.

**Key design principle**: Each search space distinguishes between **default** and
**optional** dimensions. Default dimensions cover the parameters that have the highest
impact on model quality across typical use cases (capacity, regularization). Optional
dimensions are for specialized tuning that most users don't need (algorithm-specific
knobs, normalization strategies, etc.). Optional dimensions use the BayesFlow library
default when not tuned.

```python
# src/bayesflow_hpo/search_spaces/base.py

from dataclasses import dataclass, field
from typing import Protocol, Sequence

@dataclass
class IntDimension:
    """Integer hyperparameter dimension."""
    name: str
    low: int
    high: int
    step: int | None = None
    log: bool = False
    default: bool = True   # True → tuned by default; False → opt-in

@dataclass
class FloatDimension:
    """Float hyperparameter dimension."""
    name: str
    low: float
    high: float
    log: bool = False
    default: bool = True

@dataclass
class CategoricalDimension:
    """Categorical hyperparameter dimension."""
    name: str
    choices: Sequence[str | int | float | bool]
    default: bool = True

Dimension = IntDimension | FloatDimension | CategoricalDimension


class SearchSpace(Protocol):
    """Protocol for network-specific search spaces."""

    @property
    def dimensions(self) -> list[Dimension]:
        """Return all tunable dimensions for this network type."""
        ...

    def sample(self, trial: optuna.Trial) -> dict:
        """Sample hyperparameters from an Optuna trial.

        Default implementation iterates over self.dimensions and calls
        the appropriate trial.suggest_* method. Override for
        conditional/dependent parameters.
        """
        ...

    def build(self, params: dict) -> keras.Layer:
        """Construct the network from sampled parameters."""
        ...
```

The default `sample()` implementation in a base class auto-maps dimensions to Optuna.
It only samples dimensions where `default=True` unless the user explicitly opts in
via `include_optional=True`:

```python
class BaseSearchSpace:
    include_optional: bool = False  # Set True to tune ALL dimensions

    def sample(self, trial: optuna.Trial) -> dict:
        params = {}
        for dim in self.dimensions:
            # Skip optional dimensions unless explicitly enabled
            if not dim.default and not self.include_optional:
                continue
            match dim:
                case IntDimension():
                    params[dim.name] = trial.suggest_int(
                        dim.name, dim.low, dim.high,
                        step=dim.step, log=dim.log,
                    )
                case FloatDimension():
                    params[dim.name] = trial.suggest_float(
                        dim.name, dim.low, dim.high, log=dim.log,
                    )
                case CategoricalDimension():
                    params[dim.name] = trial.suggest_categorical(
                        dim.name, dim.choices,
                    )
        return params
```

Users can opt-in to optional dimensions in two ways:

```python
# Option 1: Enable ALL optional dimensions for a space
space = hpo.CouplingFlowSpace(include_optional=True)

# Option 2: Promote a single optional dimension to default
space = hpo.CouplingFlowSpace(
    transform=hpo.CategoricalDimension(
        "cf_transform", choices=["affine", "spline"], default=True
    ),
)
```

### 4.2 Network-Specific Search Spaces

Each file defines a dataclass with sensible default ranges that users can override.
Dimensions marked `default=True` are tuned out of the box; dimensions marked
`default=False` use the BayesFlow library default and are only tuned when explicitly
opted in.

**Design principle**: Default dimensions are the ones with the highest impact-to-cost
ratio across typical use cases — network capacity (width, depth) and regularization
(dropout). Algorithm-specific knobs (transform type, permutation strategy, ODE
solver method) have good library defaults and only need tuning in specialized
scenarios.

```python
# src/bayesflow_hpo/search_spaces/inference/coupling_flow.py

@dataclass
class CouplingFlowSpace(BaseSearchSpace):
    """Search space for bf.networks.CouplingFlow.

    Default dimensions (5): depth, subnet_width, subnet_depth, dropout, activation.
    These cover model capacity and regularization — the highest-impact HPs.

    Optional dimensions (3): transform, permutation, actnorm.
    These have good library defaults (affine, random, True) and are only
    worth tuning when exploring advanced architectures.
    """

    # --- DEFAULT: tuned out of the box ---
    depth: IntDimension = field(
        default_factory=lambda: IntDimension("cf_depth", low=2, high=12)
    )
    subnet_width: IntDimension = field(
        default_factory=lambda: IntDimension("cf_subnet_width", low=32, high=256, step=32)
    )
    subnet_depth: IntDimension = field(
        default_factory=lambda: IntDimension("cf_subnet_depth", low=1, high=3)
    )
    dropout: FloatDimension = field(
        default_factory=lambda: FloatDimension("cf_dropout", low=0.0, high=0.3)
    )
    activation: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "cf_activation", choices=["silu", "relu", "mish"]
        )
    )

    # --- OPTIONAL: only tuned when include_optional=True or individually promoted ---
    transform: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "cf_transform", choices=["affine", "spline"], default=False
        )
    )
    permutation: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "cf_permutation", choices=["random", "orthogonal"], default=False
        )
    )
    use_actnorm: CategoricalDimension = field(
        default_factory=lambda: CategoricalDimension(
            "cf_actnorm", choices=[True, False], default=False
        )
    )

    @property
    def dimensions(self) -> list[Dimension]:
        return [
            self.depth, self.subnet_width, self.subnet_depth,
            self.dropout, self.activation,
            self.transform, self.permutation, self.use_actnorm,
        ]

    def build(self, params: dict) -> bf.networks.CouplingFlow:
        width = params["cf_subnet_width"]
        n_layers = params["cf_subnet_depth"]
        return bf.networks.CouplingFlow(
            depth=params["cf_depth"],
            # Optional params: use sampled value if present, else library default
            transform=params.get("cf_transform", "affine"),
            permutation=params.get("cf_permutation", "random"),
            use_actnorm=params.get("cf_actnorm", True),
            subnet_kwargs=dict(
                widths=tuple([width] * n_layers),
                activation=params["cf_activation"],
                dropout=params["cf_dropout"],
            ),
        )
```

Similarly for **FlowMatching**, **DiffusionModel**, **ConsistencyModel**, **DeepSet**,
**SetTransformer**, etc. — each with network-appropriate default and optional dimensions.

**Full search space coverage** (✓ = default, ○ = optional):

| Network | Default Dimensions | Optional Dimensions | # Default | # Total |
|---------|-------------------|---------------------|-----------|---------|
| **CouplingFlow** | depth, subnet_width, subnet_depth, dropout, activation | transform, permutation, actnorm | 5 | 8 |
| **FlowMatching** | subnet_width, subnet_depth, dropout, activation | use_ot, time_alpha, loss_fn | 4 | 7 |
| **DiffusionModel** | subnet_width, subnet_depth, dropout, activation | noise_schedule, prediction_type | 4 | 6 |
| **ConsistencyModel** | subnet_width, subnet_depth, dropout | max_time, sigma2, s0, s1 | 3 | 7 |
| **StableConsistencyModel** | subnet_width, subnet_depth, dropout | sigma | 3 | 4 |
| **DeepSet** | summary_dim, depth, width, dropout | activation, spectral_norm, pooling | 4 | 7 |
| **SetTransformer** | summary_dim, embed_dim, num_heads, num_layers, dropout | mlp_width, mlp_depth, num_inducing | 5 | 8 |
| **TimeSeriesNetwork** | summary_dim, recurrent_dim, filters, dropout | recurrent_type, bidirectional, skip_steps | 4 | 7 |
| **TimeSeriesTransformer** | summary_dim, embed_dim, num_heads, num_layers, dropout | mlp_width, time_embed | 5 | 7 |
| **FusionTransformer** | summary_dim, embed_dim, num_heads, num_layers, template_dim, dropout | template_type | 6 | 7 |
| **Training** | initial_lr | batch_size, decay_rate | 1 | 3 |

**Rationale for default selections**: The default set always includes **capacity** params
(width, depth, number of layers/heads) and **regularization** (dropout). These have the
highest impact on posterior quality across all use cases. Algorithm-specific knobs
(transform types, ODE integrators, noise schedules) have well-tuned library defaults
and only need optimization when researchers actively explore architectural variants.

### 4.3 Composite Search Space

Users pick which networks to optimize together:

```python
# src/bayesflow_hpo/search_spaces/composite.py

@dataclass
class CompositeSearchSpace:
    """Combines inference + summary + training search spaces."""
    inference_space: SearchSpace
    summary_space: SearchSpace | None = None
    training_space: TrainingSpace = field(default_factory=TrainingSpace)

    def sample(self, trial: optuna.Trial) -> dict:
        params = self.inference_space.sample(trial)
        if self.summary_space is not None:
            params.update(self.summary_space.sample(trial))
        params.update(self.training_space.sample(trial))
        return params
```

### 4.4 Network Type Selection as Hyperparameter (Optional)

For exploratory studies, users can let Optuna pick the network type itself:

```python
# src/bayesflow_hpo/search_spaces/composite.py

@dataclass
class NetworkSelectionSpace(BaseSearchSpace):
    """Lets Optuna choose between multiple inference network types."""
    candidates: dict[str, SearchSpace]  # e.g. {"coupling_flow": CouplingFlowSpace(), ...}

    def sample(self, trial: optuna.Trial) -> dict:
        network_type = trial.suggest_categorical(
            "inference_network_type",
            list(self.candidates.keys()),
        )
        params = self.candidates[network_type].sample(trial)
        params["_inference_network_type"] = network_type
        return params

    def build(self, params: dict) -> keras.Layer:
        network_type = params["_inference_network_type"]
        return self.candidates[network_type].build(params)
```

### 4.5 GenericObjective

The core objective function is a configurable factory. Users supply their
domain-specific pieces; HPO handles everything else.

The **validation dataset is created once** before the study starts and injected
into the objective. Every trial evaluates against the same fixed data (see §6).

```python
# src/bayesflow_hpo/optimization/objective.py

@dataclass
class ObjectiveConfig:
    """Everything needed to run a single HPO trial."""
    # User-provided (domain-specific)
    simulator: bf.simulators.Simulator
    adapter: bf.adapters.Adapter
    inference_conditions: list[str] | None   # Context keys for inference network

    # Validation data: fixed dataset, created once before the study starts.
    # Users provide this via one of:
    #   (a) generate_validation_dataset(simulator, condition_grid, ...)
    #   (b) a pre-built ValidationDataset (e.g. from disk or real data)
    #   (c) None → fall back to training loss only (not recommended)
    validation_data: ValidationDataset | None = None

    # Search space (user-selectable, sensible defaults)
    search_space: CompositeSearchSpace = None  # Auto-detected from network types if None

    # Training config
    epochs: int = 200
    batches_per_epoch: int = 50
    early_stopping_patience: int = 15
    early_stopping_window: int = 15

    # Constraints
    max_param_count: int = 2_000_000
    param_budget_penalty: tuple[float, float] = (1.0, 1.5)

    # Performance
    use_mixed_precision: bool = False
    use_torch_compile: bool = False


class GenericObjective:
    """Callable objective for Optuna: trial → (cal_error, param_score).

    The validation_data is set once at construction and reused for every
    trial. This guarantees fair comparison across architectures.
    """

    def __init__(self, config: ObjectiveConfig):
        self.config = config

    def __call__(self, trial: optuna.Trial) -> tuple[float, float]:
        params = self.config.search_space.sample(trial)

        # 1. Budget check
        estimated = estimate_param_count(params, self.config.search_space)
        if estimated > self.config.max_param_count:
            return self.config.param_budget_penalty

        # 2. Build networks
        inference_net = self.config.search_space.inference_space.build(params)
        summary_net = (
            self.config.search_space.summary_space.build(params)
            if self.config.search_space.summary_space else None
        )

        # 3. Create workflow
        workflow = build_workflow(
            inference_net=inference_net,
            summary_net=summary_net,
            simulator=self.config.simulator,
            adapter=self.config.adapter,
            params=params,
            config=self.config,
        )

        # 4. Train with callbacks
        try:
            workflow.fit(...)
        except optuna.TrialPruned:
            raise
        except Exception:
            return self.config.param_budget_penalty

        # 5. Validate against the FIXED dataset
        if self.config.validation_data is not None:
            results = run_validation_pipeline(
                approximator=workflow.approximator,
                validation_data=self.config.validation_data,
            )
            cal_error = results["calibration_error"]
        else:
            # Fallback: use final training loss as proxy
            cal_error = workflow.approximator.history["loss"][-1]

        # 6. Extract objectives
        param_score = normalize_param_count(
            get_param_count(workflow.approximator)
        )

        # 7. Cleanup
        cleanup_trial()
        return cal_error, param_score
```

### 4.6 User-Facing API (High Level)

The typical researcher workflow:

```python
import bayesflow as bf
import bayesflow_hpo as hpo

# ---- User defines their model (domain-specific) ----
simulator = bf.simulators.make_simulator(prior=my_prior, likelihood=my_likelihood)
adapter = bf.adapters.Adapter(...)

# ---- Option A: Quick start (condition grid → HPO generates validation data) ----
study = hpo.optimize(
    simulator=simulator,
    adapter=adapter,
    param_keys=["theta"],
    data_keys=["x"],
    # HPO generates a fixed validation dataset from this grid internally:
    validation_conditions={"N": [50, 100, 200]},
    sims_per_condition=200,
    n_trials=100,
    storage="sqlite:///hpo.db",
    study_name="my_model_hpo",
)

# ---- Option B: Pre-generate validation data (expensive simulator) ----
val_data = hpo.generate_validation_dataset(
    simulator=simulator,
    param_keys=["theta"],
    data_keys=["x"],
    condition_grid={"N": [50, 100, 200], "sigma": [0.1, 0.5, 1.0]},
    sims_per_condition=500,
    seed=42,
)
hpo.save_validation_dataset(val_data, "validation_data/")  # Persist to disk

study = hpo.optimize(
    simulator=simulator,
    adapter=adapter,
    validation_data=val_data,      # Pass pre-built dataset
    search_space=hpo.CompositeSearchSpace(
        inference_space=hpo.NetworkSelectionSpace({
            "coupling_flow": hpo.CouplingFlowSpace(
                depth=hpo.IntDimension("cf_depth", 4, 10),
            ),
            "flow_matching": hpo.FlowMatchingSpace(),
        }),
        summary_space=hpo.DeepSetSpace(),
        training_space=hpo.TrainingSpace(),
    ),
    n_trials=200,
    max_param_count=500_000,
    epochs=150,
    storage="sqlite:///hpo.db",
)

# ---- Option C: No conditions (unconditional model) ----
val_data = hpo.generate_validation_dataset(
    simulator=simulator,
    param_keys=["theta"],
    data_keys=["x"],
    # No condition_grid → single unconditional batch
    sims_per_condition=1000,
)
study = hpo.optimize(
    simulator=simulator,
    adapter=adapter,
    validation_data=val_data,
    n_trials=100,
)

# ---- Analyze results ----
hpo.plot_pareto_front(study)
best_params = hpo.get_best_trial(study)
best_workflow = hpo.build_best_workflow(study, simulator, adapter)
```

---

## 5. Search Space Defaults Per Network

### 5.1 Inference Networks

Legend: **✓** = default (always tuned), **○** = optional (opt-in)

#### CouplingFlow
| | Parameter | Range | Scale | Rationale |
|-|-----------|-------|-------|-----------|
| ✓ | `cf_depth` | 2–12 | linear | # coupling layers; primary capacity knob |
| ✓ | `cf_subnet_width` | 32–256, step 32 | linear | MLP width in coupling subnets |
| ✓ | `cf_subnet_depth` | 1–3 | linear | # hidden layers per subnet MLP |
| ✓ | `cf_dropout` | 0.0–0.3 | linear | Regularization |
| ✓ | `cf_activation` | silu, relu, mish | categorical | Nonlinearity choice |
| ○ | `cf_transform` | affine, spline | categorical | Affine works well; spline is ~3x slower |
| ○ | `cf_permutation` | random, orthogonal | categorical | Random is fine; orthogonal marginally better |
| ○ | `cf_actnorm` | True, False | categorical | Library default (True) is usually right |

#### FlowMatching
| | Parameter | Range | Scale | Rationale |
|-|-----------|-------|-------|-----------|
| ✓ | `fm_subnet_width` | 32–256, step 32 | linear | TimeMLP hidden width |
| ✓ | `fm_subnet_depth` | 1–4 | linear | TimeMLP layers |
| ✓ | `fm_dropout` | 0.0–0.2 | linear | Regularization |
| ✓ | `fm_activation` | mish, silu | categorical | Nonlinearity choice |
| ○ | `fm_use_ot` | True, False | categorical | Optimal transport: better quality but 2.5x slower |
| ○ | `fm_time_alpha` | 0.0–2.0 | linear | Power-law time sampling; 0=uniform is fine |
| ○ | `fm_loss` | mse | fixed | MSE is standard; no reason to change |

#### DiffusionModel
| | Parameter | Range | Scale | Rationale |
|-|-----------|-------|-------|-----------|
| ✓ | `dm_subnet_width` | 32–256, step 32 | linear | Score network capacity |
| ✓ | `dm_subnet_depth` | 1–4 | linear | Score network depth |
| ✓ | `dm_dropout` | 0.0–0.2 | linear | Regularization |
| ✓ | `dm_activation` | mish, silu | categorical | Nonlinearity choice |
| ○ | `dm_noise_schedule` | edm, cosine | categorical | EDM is modern default, rarely needs change |
| ○ | `dm_prediction_type` | F, velocity, noise, x | categorical | Includes BayesFlow-supported x-prediction |

#### ConsistencyModel
| | Parameter | Range | Scale | Rationale |
|-|-----------|-------|-------|-----------|
| ✓ | `cm_subnet_width` | 32–256, step 32 | linear | Network capacity |
| ✓ | `cm_subnet_depth` | 1–4 | linear | Network depth |
| ✓ | `cm_dropout` | 0.0–0.2 | linear | Regularization |
| ○ | `cm_max_time` | 50–500 | linear | Noise schedule max; paper defaults work |
| ○ | `cm_sigma2` | 0.1–2.0 | linear | Skip function shape; paper default works |
| ○ | `cm_s0` | 2–30 | linear | Initial discretization steps |
| ○ | `cm_s1` | 20–100 | linear | Final discretization steps |

#### StableConsistencyModel
| | Parameter | Range | Scale | Rationale |
|-|-----------|-------|-------|-----------|
| ✓ | `scm_subnet_width` | 64–256, step 32 | linear | Network capacity |
| ✓ | `scm_subnet_depth` | 1–3 | linear | Network depth |
| ✓ | `scm_dropout` | 0.0–0.2 | linear | Regularization |
| ○ | `scm_sigma` | 0.1–2.0 | linear | Noise level; default (1.0) is usually fine |

### 5.2 Summary Networks

#### DeepSet
| | Parameter | Range | Scale | Rationale |
|-|-----------|-------|-------|-----------|
| ✓ | `ds_summary_dim` | 4–32 | linear | Output summary dimension; critical for info retention |
| ✓ | `ds_depth` | 1–4 | linear | Aggregation stages |
| ✓ | `ds_width` | 32–256, step 32 | linear | Shared across all 4 equivariant/invariant MLPs |
| ✓ | `ds_dropout` | 0.0–0.3 | linear | Regularization |
| ○ | `ds_activation` | silu, mish | categorical | silu is BayesFlow default; rarely changes results |
| ○ | `ds_spectral_norm` | True, False | categorical | Lipschitz regularization; specialized |
| ○ | `ds_pooling` | mean | fixed | Mean is standard; PMA is rarely better |

#### SetTransformer
| | Parameter | Range | Scale | Rationale |
|-|-----------|-------|-------|-----------|
| ✓ | `st_summary_dim` | 8–64, step 8 | linear | Output summary dimension |
| ✓ | `st_embed_dim` | 32–256, step 32 | linear | Attention embedding |
| ✓ | `st_num_heads` | 1, 2, 4, 8 | categorical | Must divide embed_dim |
| ✓ | `st_num_layers` | 1–4 | linear | SAB/ISAB stacks |
| ✓ | `st_dropout` | 0.0–0.3 | linear | Regularization |
| ○ | `st_mlp_width` | 64–512, step 64 | linear | FFN width; usually scales with embed_dim |
| ○ | `st_mlp_depth` | 1–4 | linear | FFN depth; usually fine at 2 |
| ○ | `st_num_inducing` | 8–64, step 8 | linear | ISAB inducing points for large sets |

#### TimeSeriesNetwork
| | Parameter | Range | Scale | Rationale |
|-|-----------|-------|-------|-----------|
| ✓ | `tsn_summary_dim` | 8–64, step 8 | linear | Output summary dimension |
| ✓ | `tsn_recurrent_dim` | 32–256, step 32 | linear | GRU/LSTM hidden units |
| ✓ | `tsn_filters` | 16–128, step 16 | linear | Conv1D filters |
| ✓ | `tsn_dropout` | 0.0–0.3 | linear | Regularization |
| ○ | `tsn_recurrent_type` | gru, lstm | categorical | GRU is default; LSTM for long sequences |
| ○ | `tsn_bidirectional` | True, False | categorical | True is default; usually best |
| ○ | `tsn_skip_steps` | 1–8 | linear | Skip connections; depends on seq length |

#### TimeSeriesTransformer
| | Parameter | Range | Scale | Rationale |
|-|-----------|-------|-------|-----------|
| ✓ | `tst_summary_dim` | 8–64, step 8 | linear | Output summary dimension |
| ✓ | `tst_embed_dim` | 32–256, step 32 | linear | Attention embedding |
| ✓ | `tst_num_heads` | 1, 2, 4, 8 | categorical | Must divide embed_dim |
| ✓ | `tst_num_layers` | 1–4 | linear | Transformer depth |
| ✓ | `tst_dropout` | 0.0–0.3 | linear | Regularization |
| ○ | `tst_mlp_width` | 64–512, step 64 | linear | FFN width; usually scales with embed_dim |
| ○ | `tst_time_embed` | time2vec, sinusoidal | categorical | time2vec is default; sinusoidal is standard alternative |

#### FusionTransformer
| | Parameter | Range | Scale | Rationale |
|-|-----------|-------|-------|-----------|
| ✓ | `ft_summary_dim` | 8–64, step 8 | linear | Output summary dimension |
| ✓ | `ft_embed_dim` | 32–256, step 32 | linear | Attention embedding |
| ✓ | `ft_num_heads` | 1, 2, 4, 8 | categorical | Must divide embed_dim |
| ✓ | `ft_num_layers` | 1–4 | linear | Transformer depth |
| ✓ | `ft_template_dim` | 32–256, step 32 | linear | Recurrent template hidden units |
| ✓ | `ft_dropout` | 0.0–0.3 | linear | Regularization |
| ○ | `ft_template_type` | lstm, gru | categorical | LSTM is default; GRU slightly faster |

### 5.3 Training
| | Parameter | Range | Scale | Rationale |
|-|-----------|-------|-------|-----------|
| ✓ | `initial_lr` | 1e-4–5e-3 | log | Most impactful training HP |
| ○ | `batch_size` | 32–1024, step 32 | linear | Memory-dependent; 128–256 usually fine |
| ○ | `decay_rate` | 0.8–0.99 | linear | Exponential LR decay; default 0.95 works |

---

## 6. Validation Pipeline Design

### 6.1 Fixed Validation Dataset (Critical Design Choice)

**Problem**: If validation data is regenerated per trial, stochastic variation in the
simulated datasets introduces noise into the objective. Two architectures that perform
identically in expectation can receive different scores simply because they were validated
on different random draws. This makes trial comparisons unreliable and wastes HPO budget.

**Solution**: The validation dataset is generated **once** before the study starts and
reused across all trials. This ensures fair, deterministic comparison.

#### Three ways to provide validation data

1. **Condition grid + simulator (recommended)**: User provides a condition grid and
   `sims_per_condition`. HPO generates the dataset internally using the simulator,
   caches it, and reuses it for every trial.

2. **Pre-generated dataset**: User passes a `ValidationDataset` object directly.
   Useful when simulation is expensive and the user wants to generate data offline,
   or when the user has real observed data for validation.

3. **No validation data (fast prototyping)**: If neither is provided, the objective
   falls back to optimizing training/validation loss only (no SBC). A warning is
   emitted recommending proper validation for final architecture selection.

```python
# src/bayesflow_hpo/validation/data.py

from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ValidationDataset:
    """Immutable container for pre-generated validation data.

    Generated once, reused across all HPO trials to ensure fair comparison.

    Parameters
    ----------
    simulations : list[dict[str, np.ndarray]]
        One dict per condition-point. Each dict maps variable names to
        arrays of shape (n_sims, ...).
    condition_labels : list[dict[str, float | int]]
        The condition values that produced each simulation batch.
        Length must match ``simulations``.
    param_keys : list[str]
        Which keys in each simulation dict are inference targets
        (ground truth parameters for SBC).
    data_keys : list[str]
        Which keys are observed data fed to the summary/inference network.
    seed : int
        The random seed used to generate this dataset (for reproducibility).
    """
    simulations: list[dict[str, np.ndarray]]
    condition_labels: list[dict[str, float | int]]
    param_keys: list[str]
    data_keys: list[str]
    seed: int


def generate_validation_dataset(
    simulator: "bf.simulators.Simulator",
    param_keys: list[str],
    data_keys: list[str],
    condition_grid: dict[str, list] | None = None,
    sims_per_condition: int = 200,
    seed: int = 42,
) -> ValidationDataset:
    """Generate a fixed validation dataset from a simulator and condition grid.

    Parameters
    ----------
    simulator : bf.simulators.Simulator
        The BayesFlow simulator to draw from.
    param_keys : list[str]
        Parameter names that are inference targets.
    data_keys : list[str]
        Data variable names (observations).
    condition_grid : dict[str, list] | None
        Dict mapping condition names to lists of values.
        The Cartesian product of all values forms the full grid.
        If None, a single "default" condition is used (one batch
        of ``sims_per_condition`` draws from the simulator with
        no fixed conditions).
    sims_per_condition : int
        Number of simulations per condition point.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    ValidationDataset
        Frozen dataset for reuse across all HPO trials.
    """
    rng = np.random.default_rng(seed)

    if condition_grid is None:
        # No conditions — single batch from the prior
        batch_seed = int(rng.integers(0, 2**31))
        sims = simulator.sample(sims_per_condition, seed=batch_seed)
        return ValidationDataset(
            simulations=[sims],
            condition_labels=[{}],
            param_keys=param_keys,
            data_keys=data_keys,
            seed=seed,
        )

    # Build Cartesian product of condition grid
    import itertools
    keys = list(condition_grid.keys())
    values = [condition_grid[k] for k in keys]
    grid_points = list(itertools.product(*values))

    simulations = []
    condition_labels = []
    for point in grid_points:
        cond = dict(zip(keys, point))
        batch_seed = int(rng.integers(0, 2**31))
        sims = simulator.sample(
            sims_per_condition,
            conditions=cond,
            seed=batch_seed,
        )
        simulations.append(sims)
        condition_labels.append(cond)

    return ValidationDataset(
        simulations=simulations,
        condition_labels=condition_labels,
        param_keys=param_keys,
        data_keys=data_keys,
        seed=seed,
    )
```

#### Lifecycle in a study

```
hpo.optimize(...)
  │
  ├── 1. generate_validation_dataset()  ← runs ONCE
  │      (or user passes pre-built ValidationDataset)
  │
  ├── 2. GenericObjective.__init__(validation_data=dataset)
  │
  └── 3. For each trial:
         objective(trial)
           ├── sample hyperparameters
           ├── build & train workflow
           └── run_validation_pipeline(
                 approximator=...,
                 validation_data=self.validation_data,  ← SAME object every trial
               )
```

#### Saving & loading validation datasets

For long-running studies that may be resumed across sessions, the validation dataset
is serialized alongside the Optuna study:

```python
def save_validation_dataset(dataset: ValidationDataset, path: str) -> None:
    """Save validation dataset to disk (numpy .npz + JSON metadata)."""
    ...

def load_validation_dataset(path: str) -> ValidationDataset:
    """Load a previously saved validation dataset."""
    ...
```

This ensures that resuming a study uses exactly the same validation data as the
original session — no silent dataset drift.

### 6.2 What Transfers Directly from bayesflow_rct

The SBC computation logic in `core/validation.py` is already model-agnostic:
- SBC ranks, coverage at all levels (1-99%)
- Calibration error = mean |empirical - nominal| coverage
- RMSE, NRMSE, MAE, bias, contraction
- KS test, chi-squared test, C2ST for rank uniformity

These transfer as-is into `bayesflow_hpo/validation/metrics.py` and
`bayesflow_hpo/validation/sbc_tests.py`.

### 6.3 Changes Needed

1. **Generalize `make_bayesflow_infer_fn()`**: Currently assumes a single `param_key` and
   specific data key layout. Needs to handle multi-parameter posteriors and arbitrary
   adapter configurations. The new version takes `ValidationDataset.param_keys` and
   `ValidationDataset.data_keys` to know which arrays to extract.

2. **`run_validation_pipeline()` takes `ValidationDataset`**: Instead of a simulator +
   condition grid, it takes the pre-built frozen dataset:

   ```python
   def run_validation_pipeline(
       approximator: bf.approximators.Approximator,
       validation_data: ValidationDataset,
       n_posterior_samples: int = 1000,
       coverage_levels: Sequence[float] | None = None,
   ) -> dict[str, Any]:
       """Run SBC validation on a fixed dataset.

       The same ValidationDataset is used for every trial in a study,
       ensuring fair comparison of architectures.
       """
       ...
   ```

3. **Per-parameter calibration error**: Current implementation computes calibration error
   for a single parameter. Need per-parameter metrics and an aggregation strategy
   (mean, max, weighted). Default: mean across parameters.

4. **Condition grid becomes optional**: When `condition_grid=None`, the
   `generate_validation_dataset()` helper produces a single unconditional batch.
   Models without meta-parameters work out of the box.

---

## 7. Migration Plan for bayesflow_rct

### 7.1 Before/After Dependency Graph

**Before:**
```
bayesflow 2.x
└── bayesflow_rct
    ├── core/optimization.py    ← generic HPO code
    ├── core/validation.py      ← generic SBC code
    ├── core/infrastructure.py  ← generic builders
    ├── core/objectives.py      ← generic metrics
    ├── core/results.py         ← generic analysis
    ├── core/utils.py           ← mixed utilities
    └── models/ancova/          ← ANCOVA-specific
```

**After:**
```
bayesflow 2.x
├── bayesflow-hpo              ← new generic package
│   ├── search_spaces/
│   ├── builders/
│   ├── optimization/
│   ├── validation/
│   └── results/
└── bayesflow_rct          ← thin application layer
    ├── models/ancova/model.py  ← simulator, prior, likelihood, adapter spec
    ├── models/ancova/hpo.py    ← ANCOVA-specific search space overrides
    ├── core/threshold.py       ← domain-specific threshold training (if needed)
    ├── core/dashboard.py       ← optional web dashboard (if needed)
    └── plotting/diagnostics.py ← ANCOVA diagnostic plots
```

### 7.2 File-Level Migration Map

| bayesflow_rct file | Destination | Notes |
|------------------------|-------------|-------|
| `core/optimization.py` → `create_study()` | `bayesflow_hpo/optimization/study.py` | Direct move |
| `core/optimization.py` → `OptunaReportCallback` | `bayesflow_hpo/optimization/callbacks.py` | Direct move |
| `core/optimization.py` → `HyperparameterSpace` | Replaced by `bayesflow_hpo/search_spaces/` | Complete rewrite |
| `core/optimization.py` → `sample_hyperparameters()` | `bayesflow_hpo/optimization/sampling.py` | Generalized |
| `core/optimization.py` → `create_optimization_objective()` | `bayesflow_hpo/optimization/objective.py` | Generalized |
| `core/optimization.py` → `cleanup_trial()` | `bayesflow_hpo/optimization/cleanup.py` | Direct move |
| `core/infrastructure.py` → `SummaryNetworkConfig` | Replaced by search space dataclasses | |
| `core/infrastructure.py` → `InferenceNetworkConfig` | Replaced by search space dataclasses | |
| `core/infrastructure.py` → `TrainingConfig` | `bayesflow_hpo/search_spaces/training.py` | Generalized |
| `core/infrastructure.py` → `build_summary_network()` | `bayesflow_hpo/builders/summary.py` | Generalized |
| `core/infrastructure.py` → `build_inference_network()` | `bayesflow_hpo/builders/inference.py` | Generalized |
| `core/infrastructure.py` → `AdapterSpec`, `create_adapter()` | Stays in bayesflow_rct | Model-specific |
| `core/infrastructure.py` → `PriorStandardize` | Stays in bayesflow_rct | ANCOVA-specific transform |
| `core/infrastructure.py` → `save/load_workflow_with_metadata()` | `bayesflow_hpo/results/export.py` | Direct move |
| `core/validation.py` (entire file) | `bayesflow_hpo/validation/` | Split into submodules |
| `core/objectives.py` | `bayesflow_hpo/objectives.py` | Direct move |
| `core/results.py` | `bayesflow_hpo/results/` | Split into submodules |
| `core/utils.py` → `MovingAverageEarlyStopping` | `bayesflow_hpo/optimization/callbacks.py` | Direct move |
| `core/utils.py` → `loguniform_*()` | `bayesflow_hpo/utils.py` | Direct move |
| `core/utils.py` → `sample_t_or_normal()` | Stays in bayesflow_rct | ANCOVA-specific |
| `models/ancova/model.py` | Stays in bayesflow_rct | All ANCOVA-specific |
| `core/threshold.py` | Stays in bayesflow_rct | Domain-specific threshold logic |
| `core/dashboard.py` | Stays in bayesflow_rct | Optional, application-specific |
| `plotting/diagnostics.py` | Stays in bayesflow_rct | ANCOVA-specific plots |

### 7.3 bayesflow_rct After Migration

The ANCOVA package becomes a thin application that composes HPO primitives:

```python
# bayesflow_rct/models/ancova/hpo.py

import bayesflow_hpo as hpo
from .model import create_simulator, create_ancova_adapter

# Path for persisting validation data across sessions
VALIDATION_DATA_DIR = "data/ancova_validation"

def get_or_create_validation_data(
    simulator, seed: int = 42,
) -> hpo.ValidationDataset:
    """Load cached validation data, or generate and save it."""
    try:
        return hpo.load_validation_dataset(VALIDATION_DATA_DIR)
    except FileNotFoundError:
        val_data = hpo.generate_validation_dataset(
            simulator=simulator,
            param_keys=["b_group"],
            data_keys=["outcome", "covariate", "group"],
            condition_grid={
                "N": [30, 100, 300],
                "p_alloc": [0.5, 0.7],
                "prior_df": [3, 10],
                "prior_scale": [0.5, 1.0],
            },
            sims_per_condition=200,
            seed=seed,
        )
        hpo.save_validation_dataset(val_data, VALIDATION_DATA_DIR)
        return val_data


def run_ancova_hpo(n_trials: int = 100, storage: str = "sqlite:///ancova_hpo.db"):
    """Run HPO for the ANCOVA NPE model."""
    simulator = create_simulator()
    adapter = create_ancova_adapter()

    # Generate (or load cached) fixed validation dataset
    val_data = get_or_create_validation_data(simulator)

    # ANCOVA uses narrower ranges than defaults (based on prior experience)
    search_space = hpo.CompositeSearchSpace(
        inference_space=hpo.CouplingFlowSpace(
            depth=hpo.IntDimension("cf_depth", 2, 8),
            subnet_width=hpo.IntDimension("cf_subnet_width", 32, 128, step=16),
        ),
        summary_space=hpo.DeepSetSpace(
            summary_dim=hpo.IntDimension("ds_summary_dim", 4, 16),
            width=hpo.IntDimension("ds_width", 32, 128, step=16),
        ),
    )

    study = hpo.optimize(
        simulator=simulator,
        adapter=adapter,
        inference_conditions=["N", "p_alloc", "prior_df", "prior_scale"],
        validation_data=val_data,
        search_space=search_space,
        n_trials=n_trials,
        storage=storage,
        study_name="ancova_hpo",
    )
    return study
```

---

## 8. pyproject.toml

```toml
[project]
name = "bayesflow-hpo"
version = "0.1.0"
description = "Generic hyperparameter optimization for BayesFlow neural posterior estimation"
requires-python = ">=3.11"
license = {text = "MIT"}
authors = [{name = "..."}]

dependencies = [
    "bayesflow>=2.0.0",
    "optuna>=3.0.0",
    "numpy>=1.24",
    "scipy>=1.10",
    "matplotlib>=3.7",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "ruff>=0.1.0",
    "mypy>=1.0",
]
c2st = [
    "scikit-learn>=1.2",       # Only needed for C2ST SBC test
]
dashboard = [
    "optuna-dashboard>=0.14",  # Web-based study dashboard
]

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 88
target-version = "py311"
```

---

## 9. Implementation Phases

### Phase 1: Foundation (Core infrastructure)
**Goal**: Package skeleton, search spaces for the 2 most common network combos,
validation data pipeline, basic objective.

1. Create repo + package skeleton with pyproject.toml
2. Implement `SearchSpace` protocol and `BaseSearchSpace` with auto-sampling
3. Implement search spaces for **CouplingFlow** and **FlowMatching** (inference).
   Subnet params (widths, activation, dropout) are inline fields — no separate subnet spaces.
4. Implement search space for **DeepSet** (summary)
5. Implement `TrainingSpace` (lr, batch size, decay)
6. Implement `CompositeSearchSpace`
7. Move builders: `build_inference_network()`, `build_summary_network()`, `build_workflow()`
8. Implement `ValidationDataset`, `generate_validation_dataset()`,
   `save_validation_dataset()`, `load_validation_dataset()`
9. Move `create_study()`, `OptunaReportCallback`, `MovingAverageEarlyStopping`, `cleanup_trial()`
10. Implement `GenericObjective` (takes `ValidationDataset` at construction, reuses across trials)
11. Move validation pipeline: `run_validation_pipeline()`, metrics, SBC tests
    (refactored to accept `ValidationDataset` instead of simulator + condition grid)
12. Move results: `get_pareto_trials()`, `trials_to_dataframe()`, plots
13. Move utilities: `get_param_count()`, `normalize_param_count()`, `loguniform_*()`
14. Implement top-level `hpo.optimize()` convenience function
    (accepts either `validation_data=` or `validation_conditions=` + auto-generates)
15. Write tests for all moved code (including validation data serialization round-trip)
16. Write `quickstart.ipynb` example with a simple Gaussian model

### Phase 2: Full Network Coverage
**Goal**: Search spaces for all remaining BayesFlow network types.

17. Add **DiffusionModel** search space + builder
18. Add **ConsistencyModel** search space + builder (requires `total_steps` calculation)
19. Add **StableConsistencyModel** search space + builder
20. Add **SetTransformer** search space + builder
21. Add **TimeSeriesNetwork** search space + builder
22. Add **TimeSeriesTransformer** search space + builder
23. Add **FusionTransformer** search space + builder
24. Implement `NetworkSelectionSpace` (inference network type as HP)
25. Implement `SummarySelectionSpace` (summary network type as HP)
26. Add search space `registry.py` for name-based lookup
27. Tests for all new search spaces + builders
28. Fix silent exception swallowing in `build_workflow()` — log or re-raise
    compile errors instead of catching all exceptions
29. Add input validation to search space `build()` methods — check for missing
    required keys and raise clear errors
30. Generalize `estimate_param_count()` to support all network types (not just
    DeepSet + CouplingFlow/FlowMatching heuristics)

### Phase 3: Advanced Features
**Goal**: Power-user features, documentation, polish.

31. Multi-parameter validation (per-parameter calibration error + aggregation)
32. Memory estimation (predict OOM before training starts)
33. Custom network registration API (for EquivariantIRTSummary etc.)
34. Warm-starting: transfer knowledge from small studies to larger ones
35. Optuna-dashboard integration example
36. Write `custom_summary_network.ipynb` and `multi_objective.ipynb` examples

### Phase 4: Migrate bayesflow_rct + resolve open issues
**Goal**: bayesflow_rct depends on bayesflow-hpo instead of its own core/.

37. Add `bayesflow-hpo` dependency to bayesflow_rct pyproject.toml
38. Replace imports in `models/ancova/model.py` (use bayesflow_hpo builders)
39. Create thin `models/ancova/hpo.py` wrapper (as shown in §7.3)
40. Delete migrated code from `core/` (optimization.py, validation.py, objectives.py, results.py)
41. Keep ANCOVA-specific code: `PriorStandardize`, `sample_t_or_normal()`, adapter spec, threshold
42. Update all notebooks/examples to use new imports
43. Run full bayesflow_rct test suite — verify no regressions
44. Update bayesflow_rct CLAUDE.md and documentation

#### Open issues carried from Phase 3 review (2026-03-04)

None of these are regressions — they were intentional refinements or minor
inconsistencies between the plan tables (§5) and the implementation. They are
now resolved in this pass for 1.0 consistency.

45. **[Resolved] Make example notebooks runnable**: `custom_summary_network.ipynb` and
    `multi_objective.ipynb` are 4-cell API stubs with undefined `simulator`/
    `adapter`. Add a minimal Gaussian model so each notebook executes
    end-to-end. Also create the Phase 1 `quickstart.ipynb` which was never
    written.

46. **[Resolved] Reconcile dimension ranges with plan tables**: Several summary-network
    spaces use `summary_dim` 8–64 step 8 instead of the plan's 4–32.
    `ConsistencyModelSpace.subnet_depth` allows 1–4 (plan: 1–3);
    `DiffusionModelSpace.subnet_width` starts at 32 (plan: 64).
    Recommendation: update the §5 tables to match the code (the wider
    ranges are more practical).

47. **[Resolved] Fix `num_heads` type**: The plan specifies `num_heads` as a
    `CategoricalDimension` with choices `[1, 2, 4, 8]` for SetTransformer,
    TimeSeriesTransformer, and FusionTransformer. The implementation uses
    `IntDimension(step=2, low=2, high=8)`, which can produce 6 (not a
    power-of-2). Switch to `CategoricalDimension` or update the plan.

48. **[Resolved] Adopt `dm_*` prefix in plan tables**: The plan §5.1 tables use `diff_*`
    prefixes; the implementation uses `dm_*`, consistent with the registry
    alias. Update the plan tables.

49. **[Resolved] ConsistencyModel `s0`/`max_time` tweaks**: Implementation uses
    `s0` low=2 (plan: 5) and `cm_max_time` with `log=False` (plan says
    log-scale). Verify whether log-scale sampling is needed and reconcile.

50. **[Resolved] DiffusionModel `prediction_type` extra choice**: The plan lists
    `[velocity, noise, F]`; the implementation adds `"x"` as a fourth
    choice. Confirm this is intentional (it is valid in BayesFlow) and
    update the plan table.

### Phase 5: Complete builder migration + clean up dual code paths
**Goal**: Eliminate the duplicate builder/config infrastructure in
`bayesflow_rct.core.infrastructure` so that `bayesflow-hpo` is the single
source of truth for network construction. After this phase, `core/` contains
only RCT-specific code (`PriorStandardize`, threshold loop, dashboard, utils).

51. **Migrate `model.py` builder imports to `bayesflow_hpo`**: Replace imports of
    `build_inference_network`, `build_summary_network`, `build_workflow`,
    `InferenceNetworkConfig`, `SummaryNetworkConfig`, `WorkflowConfig`,
    `AdapterSpec`, `params_dict_to_workflow_config`, and related helpers in
    `models/ancova/model.py` with their `bayesflow_hpo.builders` equivalents.
    This may require thin adapter functions if the bayesflow_hpo builder API
    differs from the old config-dataclass API.

52. **Migrate workflow metadata helpers**: `get_workflow_metadata()`,
    `save_workflow_with_metadata()`, `load_workflow_with_metadata()` in
    `core/infrastructure.py` are generic utilities. Either move them into
    `bayesflow_hpo` (e.g. `bayesflow_hpo.results.export`) or keep them in
    `core/` if they are RCT-specific. Decide and act.

53. **Migrate `create_adapter()` / `AdapterSpec`**: The declarative adapter
    builder in `core/infrastructure.py` is generic infrastructure. Move it to
    `bayesflow_hpo` or confirm it is ANCOVA-specific (due to
    `PriorStandardize` wiring) and keep it in `core/`.

54. **Migrate `compile_approximator()` and `configure_training_performance()`**:
    These are generic training utilities in `core/infrastructure.py`.
    Determine whether `bayesflow_hpo.optimization` already covers this
    functionality; if not, migrate or delete.

55. **Migrate `generate_validation_data()`**: `core/infrastructure.py` has its
    own `generate_validation_data()` which overlaps with
    `bayesflow_hpo.validation.generate_validation_dataset()`. Remove the
    duplicate and update callers.

56. **Slim down `core/infrastructure.py`**: After items 51–55, the only code
    remaining in `infrastructure.py` should be `PriorStandardize` and
    `create_simulator()` (which is ANCOVA-specific because of
    `sample_t_or_normal`). Verify the file is ≤200 lines or split
    `PriorStandardize` into its own module.

57. **Update `test_core/` tests**: `test_optimization.py` and
    `test_validation.py` still exist in `tests/test_core/`. Verify they pass
    against the new import paths. Delete tests for code that has been fully
    migrated to `bayesflow_hpo` (those are tested in the HPO package). Add
    tests for any new adapter shims created in item 51.

58. **Update notebooks/examples**: Verify all notebooks under
    `bayesflow_rct/examples/` use the new import paths and run without error.

59. **Run full bayesflow_rct test suite**: `pytest tests/ -v` must pass with
    zero failures after the migration.

60. **Final documentation pass**: Update `bayesflow_rct/CLAUDE.md` project
    structure section to reflect the slimmed-down `core/`. Remove references
    to deleted builder code.

---

## 10. Key Design Decisions

### D1: Dataclass-based search spaces (not YAML/JSON config)

**Chosen**: Dataclass with typed fields and defaults.
**Rejected**: YAML/JSON config files.
**Rationale**: Type checking, IDE autocompletion, and the ability to override individual
dimensions via constructor arguments. Config files are opaque and lose all tooling support.

### D2: Dimensions namespaced by network prefix (e.g., `cf_depth`, `ds_width`)

**Rationale**: Optuna requires globally unique parameter names within a study.
When combining inference + summary + training spaces, prefixes prevent collisions.
Also makes parameter importance plots self-documenting.

### D3: build() lives on the SearchSpace, not a separate registry

**Rationale**: Keeps the "what to tune" and "how to construct" co-located. The search
space knows exactly which constructor arguments correspond to which dimensions.
A separate registry would require maintaining parallel mapping logic.

### D4: Single-objective convenience + multi-objective power

The top-level `hpo.optimize()` defaults to multi-objective (calibration + params),
but accepts `directions=["minimize"]` for single-objective (calibration only).
Users who don't care about model size can ignore the second objective.

### D5: Fixed validation dataset, generated once

Validation data is generated **once** before the study starts and reused across all
trials. This eliminates stochastic noise in trial comparison — two architectures that
perform identically in expectation will receive the same score.

Users can either:
- Supply a `condition_grid` and let `hpo.optimize()` call
  `generate_validation_dataset()` internally, or
- Pass a pre-built `ValidationDataset` (for expensive simulators or real data).

If no validation data is provided, the objective falls back to training loss only.
This supports fast prototyping but is not recommended for final architecture selection.

### D6: No tight coupling to bayesflow_calibration_loss

bayesflow-hpo is agnostic to the calibration loss package. Users who want
Lagrangian-calibrated training can pass a `CalibratedContinuousApproximator` as a
custom approximator class — HPO doesn't need to know about it.

### D7: No separate subnet search spaces

**Chosen**: Subnet parameters (widths, activation, dropout) are inline fields on each
inference network's search space.
**Rejected**: Separate `SubnetSearchSpace` objects composed into inference spaces.
**Rationale**: Nobody tunes "the MLP" independently — they tune the CouplingFlow which
*uses* an MLP internally. Different inference networks need different default ranges for
the same underlying MLP parameters (CouplingFlow subnets are typically wider than
FlowMatching subnets). The subnet type is also fixed by the inference network:
CouplingFlow uses `MLP`, FlowMatching uses `TimeMLP`. A separate abstraction would
force an unnecessary seam and require cross-space dependency wiring (which subnet type
goes with which inference type). Keeping everything flat in `CouplingFlowSpace` etc.
is simpler, self-contained, and requires zero coordination logic.

### D8: Default vs. optional dimensions

**Chosen**: Each dimension carries a `default: bool` flag. `BaseSearchSpace.sample()`
only suggests dimensions where `default=True` unless `include_optional=True`.
**Rationale**: Researchers with limited compute budget (the common case) want to tune the
3-5 most impactful parameters per network, not explore a 15-dimensional space where most
dimensions barely matter. The default set always covers **capacity** (width, depth,
num_layers, num_heads) and **regularization** (dropout) — these consistently have the
largest effect on posterior quality. Algorithm-specific knobs (transform type, noise
schedule, ODE solver) have carefully tuned library defaults that rarely need overriding.

Users who *do* want full exploration can set `include_optional=True` on any space, or
promote individual dimensions by overriding them with `default=True`.

**What is NOT default** (and why):
- **Transform type** (CouplingFlow): Affine is fast and sufficient; spline is ~3x slower.
- **Optimal transport** (FlowMatching): 2.5x slower; only worth enabling deliberately.
- **Noise schedule** (Diffusion): EDM is state-of-the-art; cosine is a fallback.
- **Spectral normalization** (DeepSet): Specialized Lipschitz regularization.
- **Number of inducing points** (SetTransformer): Only relevant for very large set sizes.
- **Batch size, LR decay** (Training): Batch size 128-256 is fine for NPE; LR decay 0.95
  is robust. Learning rate is the only training HP worth tuning by default.

---

## 11. Updated Dependency DAG

```
bayesflow 2.x (external)
├── bayesflow-irt              (standalone domain extension)
├── bayesflow-calibration-loss (standalone training enhancement)
├── bayesflow-hpo              (standalone HPO helper)           ← NEW
└── bayesflow_rct          (application)
    ├── depends on: bayesflow-hpo
    └── optionally depends on: bayesflow-calibration-loss
```

---

## 12. Open Questions

1. **Package name**: `bayesflow-hpo` vs `bayesflow-optuna` vs `bayesflow-tuner`?
   → Leaning toward `bayesflow-hpo` (most descriptive, not tied to Optuna in name).

2. **Should network type selection default to on or off?**
   → Default to a single network combo (DeepSet + CouplingFlow). Network type selection
   is opt-in via `NetworkSelectionSpace`.

3. **Should we support single-objective HPO (minimize val_loss only)?**
   → Yes, as a simpler entry point. Multi-objective is the default for power users.

4. **How to handle ConsistencyModel's `total_steps` requirement?**
   → Calculate from `epochs * batches_per_epoch` and inject into the builder automatically.
   User doesn't need to think about it.
