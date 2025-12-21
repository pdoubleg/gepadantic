# GEPA Optimization Interfaces

GEPADANTIC provides three levels of interface for prompt optimization, each offering different levels of control and convenience. This guide demonstrates all three using the Palmer Penguins dataset for species classification.

## Overview

Choose the interface that best fits your needs:

- **High-Level (GepaConfig)**: Declarative configuration object - easiest
- **Mid-Level (optimize_agent_prompts)**: Functional API with explicit parameters - more control  
- **Low-Level (gepa.api.optimize)**: Direct access to GEPA core - maximum flexibility

<details markdown="1">
<summary><strong>Dataset Setup & Helper Functions</strong> (click to expand)</summary>

* First, let's set up our data models and dataset loading function that we'll reuse across all examples:

```python
from typing import Literal
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split

from gepadantic import DataInstWithInput
from gepadantic.data_utils import dataframe_to_dataset

# Define input and output models
class PenguinInput(BaseModel):
    """Input features for Palmer Penguins species classification."""
    bill_length_mm: float = Field(
        description="Length of the penguin's bill (culmen) in millimeters"
    )
    bill_depth_mm: float = Field(
        description="Depth of the penguin's bill (culmen) in millimeters"
    )
    flipper_length_mm: float = Field(
        description="Length of the penguin's flipper in millimeters"
    )
    body_mass_g: float = Field(
        description="Body mass of the penguin in grams"
    )
    sex: Literal["Male", "Female"] = Field(description="Sex of the penguin")
    island: Literal["Torgersen", "Biscoe", "Dream"] = Field(
        description="Island where the penguin was observed"
    )

class SpeciesPrediction(BaseModel):
    """Output prediction for penguin species classification."""
    species: Literal["Adelie", "Chinstrap", "Gentoo"] = Field(
        description="Predicted penguin species"
    )
    confidence: float = Field(
        description="Confidence score between 0 and 1", ge=0.0, le=1.0
    )

def load_penguins_data(n_train: int = 30):
    """Load and prepare Palmer Penguins dataset for GEPA.
    
    Args:
        n_train: Number of samples to use for training (default 30)
    
    Returns:
        List of DataInstWithInput with PenguinInput instances
    """
    import seaborn as sns
    
    # Load dataset
    df = sns.load_dataset("penguins")
    
    # Select features and clean data
    df = df[
        [
            "bill_length_mm", "bill_depth_mm", "flipper_length_mm",
            "body_mass_g", "sex", "island", "species",
        ]
    ].dropna()
    
    # Store species as label for metadata
    df["label"] = df["species"]
    
    # Sample with stratification to maintain class balance
    if len(df) > n_train:
        df, _ = train_test_split(
            df, train_size=n_train, stratify=df["species"], random_state=42
        )
    
    # Convert to PenguinInput instances
    def row_to_penguin_input(row: pd.Series) -> PenguinInput:
        return PenguinInput(
            bill_length_mm=float(row["bill_length_mm"]),
            bill_depth_mm=float(row["bill_depth_mm"]),
            flipper_length_mm=float(row["flipper_length_mm"]),
            body_mass_g=float(row["body_mass_g"]),
            sex=str(row["sex"]),
            island=str(row["island"]),
        )
    
    # Convert DataFrame to dataset
    dataset = dataframe_to_dataset(
        df,
        row_mapper=row_to_penguin_input,
        metadata_cols=["label"],
    )
    
    return dataset

def species_metric(data_inst, output):
    """Evaluate species prediction accuracy.
    
    Args:
        data_inst: Input data instance with metadata containing ground truth
        output: Agent's output (RolloutOutput)
    
    Returns:
        Tuple of (score, feedback) where score is between 0.0 and 1.0
    """
    if not output.success or output.result is None:
        return 0.0, output.error_message or "Agent failed to produce output"
    
    predicted_species = output.result.species
    ground_truth = data_inst.metadata.get("label")
    
    if ground_truth is None:
        return 0.0, "No ground truth label found in metadata"
    
    is_correct = predicted_species == ground_truth
    
    if is_correct:
        return 1.0, None
    else:
        return 0.0, f"Incorrect: predicted '{predicted_species}', actual '{ground_truth}'"
```

</details>

---

## High-Level Interface: GepaConfig

The `GepaConfig` class provides the simplest interface. Just configure everything in one object and call `run_optimization_pipeline()`.

### Example: Config-Based Optimization

```python
from gepadantic import GepaConfig, run_optimization_pipeline
from gepadantic.data_utils import split_dataset

# Load dataset (using helper from above)
dataset = load_penguins_data(n_train=30)

# Split into train/val
trainset, valset = split_dataset(dataset, train_ratio=0.6, shuffle=True)

# Configure optimization
config = GepaConfig(
    # Agent configuration - let GEPA create the agent for you
    agent_model="gpt-4.1-nano",
    agent_instructions=(
        "Classify the penguin species based on the given physical measurements "
        "and location. Consider bill dimensions, flipper length, body mass, "
        "sex, and island to distinguish between Adelie, Chinstrap, and Gentoo."
    ),
    
    # Input/Output types
    input_type=PenguinInput,
    output_type=SpeciesPrediction,
    
    # Data and evaluation
    trainset=trainset,
    valset=valset,
    metric=species_metric,
    
    # Budget - use auto-budget for convenience
    auto="light",  # Options: "light", "medium", "heavy"
    
    # Optimization settings
    reflection_model="gpt-4.1-mini",  # Better model for proposing new prompts
    module_selector="all",
    candidate_selection_strategy="pareto",
    
    # Output
    output_dir="optimization_results",
    save_result=True,
)

# Run optimization - that's it!
result = run_optimization_pipeline(config)

# Access results
print(f"Best score: {result.best_score:.2%}")
print(f"Improvement: {result.improvement_ratio():.2%}")
print(f"Best prompts:\n{result.best_candidate}")
```

### Example: Using Pre-configured Agent with Config

You can also pass a pre-configured `SignatureAgent` if you need to add tools or custom setup. The `input_type`, `output_type` and `agent_instructions` will be inferred from the `SignatureAgent`, so they should be omitted from the `GepaConfig`.

```python
from pydantic_ai import Agent
from gepadantic import SignatureAgent, GepaConfig, run_optimization_pipeline
from gepadantic.lm import get_openai_model

# Create a custom agent with a tool
model = get_openai_model("gpt-4.1-nano")
base_agent = Agent(
    model=model,
    instructions=(
        "Classify penguin species using measurements and your penguin knowledge tool."
    ),
    output_type=SpeciesPrediction,
)

@base_agent.tool
def get_species_info(species: str) -> str:
    """Get information about a penguin species to help with classification.
    
    Args:
        species: The species name (Adelie, Chinstrap, or Gentoo)
    """
    info = {
        "Adelie": "Smaller penguins, found on all three islands",
        "Chinstrap": "Medium-sized, distinctive white face markings",
        "Gentoo": "Largest species, primarily found on Biscoe island",
    }
    return info.get(species, "Unknown species")

# Wrap in SignatureAgent
signature_agent = SignatureAgent(base_agent, input_type=PenguinInput)

# Load and split data
dataset = load_penguins_data(n_train=30)
trainset, valset = split_dataset(dataset, train_ratio=0.6)

# Configure with pre-built agent
config = GepaConfig(
    agent=signature_agent,  # Pass the agent directly
    trainset=trainset,
    valset=valset,
    metric=species_metric,
    auto="light",
    optimize_tools=True,
)

result = run_optimization_pipeline(config)
```

---

## Mid-Level Interface: optimize_agent_prompts

The `optimize_agent_prompts()` function gives you more explicit control without needing a config object.

### Example: Direct Function Call

```python
from pydantic_ai import Agent
from gepadantic import SignatureAgent, optimize_agent_prompts
from gepadantic.lm import get_openai_model
from gepadantic.data_utils import split_dataset

# Load and split data
dataset = load_penguins_data(n_train=30)
trainset, valset = split_dataset(dataset, train_ratio=0.6)

# Create agent explicitly
model = get_openai_model("gpt-4.1-nano")
base_agent = Agent(
    model=model,
    instructions=(
        "Classify penguin species based on physical measurements."
    ),
    output_type=SpeciesPrediction,
)

signature_agent = SignatureAgent(base_agent, input_type=PenguinInput)

# Run optimization with explicit parameters
result = optimize_agent_prompts(
    signature_agent=signature_agent,
    trainset=trainset,
    valset=valset,
    metric=species_metric,
    input_type=PenguinInput,  # Optimize input field descriptions too
    
    # Budget
    auto="light",
    
    # Reflection configuration
    reflection_model="gpt-4.1-mini",
    candidate_selection_strategy="pareto",
    reflection_minibatch_size=3,
    
    # Component selection
    module_selector="all",
    
    # Logging
    display_progress_bar=True,
    track_best_outputs=True,
)

print(f"Optimized prompts: {result.best_candidate}")
```

---

## Low-Level Interface: gepa.api.optimize

For maximum control, use the GEPA core API directly. This requires manual setup of all components.

### Example: Full Manual Setup

```python
import gepa.api
from pydantic_ai import Agent

from gepadantic import SignatureAgent, PydanticAIGEPAAdapter
from gepadantic.lm import get_openai_model
from gepadantic.runner import auto_budget
from gepadantic.data_utils import split_dataset
from gepadantic.components import extract_seed_candidate_with_signature
from gepa.logging.logger import StdOutLogger

# 1. Create agent
model = get_openai_model("gpt-4.1-nano")
base_agent = Agent(
    model=model,
    instructions="Classify penguin species based on measurements.",
    output_type=SpeciesPrediction,
)

signature_agent = SignatureAgent(base_agent, input_type=PenguinInput)

# 2. Load and prepare data
dataset = load_penguins_data(n_train=30)

trainset, valset = split_dataset(dataset, train_ratio=0.6)

# 3. Extract initial candidate prompts
seed_candidate = extract_seed_candidate_with_signature(
    agent=signature_agent,
    input_type=PenguinInput,
)

# 4. Create adapter
adapter = PydanticAIGEPAAdapter(
    agent=signature_agent,
    metric=species_metric,
    reflection_model="gpt-4.1-mini",
)

# 5. Create logger
logger = StdOutLogger()

# 6. Set manual budget
max_metric_calls = 50

# 7. Run GEPA optimization directly
raw_result = gepa.api.optimize(
    adapter=adapter,
    seed_candidate=seed_candidate,
    trainset=list(trainset),
    valset=list(valset),
    
    # Budget
    max_metric_calls=max_metric_calls,
    
    # Reflection configuration
    candidate_selection_strategy="pareto",
    reflection_minibatch_size=3,
    perfect_score=1.0,
    skip_perfect_score=True,
    
    # Component selection
    module_selector="all",
    
    # Merge strategy
    use_merge=True,
    max_merge_invocations=5,
    merge_val_overlap_floor=5,
    
    # Logging
    logger=logger,
    display_progress_bar=True,
    track_best_outputs=True,
    
    # Reproducibility
    seed=0,
    raise_on_exception=True,
)

# 8. Extract results manually
best_candidate = raw_result.best_candidate
best_score = raw_result.val_aggregate_scores[raw_result.best_idx]

print(f"Best score: {best_score:.2%}")
print(f"Optimized prompts: {best_candidate}")
```

---

## Configuration Reference

For detailed configuration options available across all interfaces, see below:

<details markdown="1">
<summary><strong>Configuration Parameters Reference</strong> (click to expand)</summary>

### Required Parameters

#### Core Data & Evaluation

| Parameter | Type | Description |
|-----------|------|-------------|
| `trainset` | `Sequence[DataInstWithInput]` | List of training examples with inputs and metadata |
| `metric` | `Callable` | Function that evaluates agent outputs, returning `(score, feedback)` tuple |

#### Agent Configuration (Pick One)

You must provide **exactly one** of the following options:

**Option 1: Pre-configured Agent**

| Parameter | Type | Description |
|-----------|------|-------------|
| `signature_agent` | `SignatureAgent` | A pre-configured and wrapped PydanticAI agent to optimize. This already contains the input/output types, model, and instructions |

**Option 2: Agent Configuration Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_type` | `type[BaseModel]` | Pydantic model class defining the structured input format |
| `output_type` | `type[BaseModel]` | Pydantic model class defining the expected output format |
| `agent_model` | `str` | Model name (e.g., "gpt-4.1-mini") to create an agent |
| `agent_instructions` | `str` | System instructions for the agent |

#### Budget Configuration (Pick One)

You must provide **exactly one** of the following:

| Parameter | Type | Description |
|-----------|------|-------------|
| `auto` | `"light"` \| `"medium"` \| `"heavy"` | Automatically set budget based on dataset size |
| `max_full_evals` | `int` | Maximum number of full evaluations (complete passes over train+val) |
| `max_metric_calls` | `int` | Maximum number of individual metric evaluations |

### Optional Parameters

#### Dataset & Validation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `valset` | `Sequence[DataInstWithInput]` | `None` | Validation examples. If `None`, trainset is used for validation |

#### Optimization Behavior

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optimize_tools` | `bool` | `True` | Whether to optimize tool descriptions (e.g., output model schema text components) |
| `seed_candidate` | `dict[str, str]` | `None` | Optional initial candidate prompts to start optimization from |
| `reflection_model` | `str` | `None` | Model to use for reflection/mutation. If `None`, uses `agent_model` |
| `reflection_minibatch_size` | `int` | `3` | Number of examples to use for reflection in each proposal |
| `perfect_score` | `int` | `1` | The perfect score value to achieve |
| `skip_perfect_score` | `bool` | `True` | Whether to skip updating if perfect score achieved on minibatch |
| `reflection_sampler` | `ReflectionSampler` | `None` | Optional sampler for reflection records. If `None`, all records are kept |

#### Component & Candidate Selection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module_selector` | `ReflectionComponentSelector` \| `"round_robin"` \| `"all"` | `"all"` | Component selection strategy. `"round_robin"` cycles through components; `"all"` selects all components every iteration |
| `candidate_selection_strategy` | `CandidateSelector` \| `"pareto"` \| `"current_best"` \| `"epsilon_greedy"` | `"pareto"` | Strategy for selecting candidates from the population |

#### Merge Strategy

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_merge` | `bool` | `True` | Whether to use the merge strategy for combining candidates |
| `max_merge_invocations` | `int` | `5` | Maximum number of merge invocations to perform |
| `merge_val_overlap_floor` | `int` | `5` | Minimum number of shared validation IDs required between parents before attempting merge |

#### Stopping Conditions

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `stop_callbacks` | `StopperProtocol` \| `Sequence[StopperProtocol]` | `None` | Custom stopper conditions for early termination |

#### Runtime Display

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `display_progress_bar` | `bool` | `True` | Whether to show progress bar during optimization |
| `track_best_outputs` | `bool` | `True` | Whether to track best outputs for analysis |

#### Caching

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_cache` | `bool` | `True` | Whether to enable caching of metric results |
| `cache_dir` | `str` | `".gepa_cache"` | Directory for cache storage |
| `cache_verbose` | `bool` | `False` | Whether to print cache statistics |

#### Output & Logging

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | `str` \| `Path` | `"optimization_results"` | Directory to save optimization results |
| `save_result` | `bool` | `True` | Whether to automatically save results to JSON |
| `logger` | `LoggerProtocol` | `None` | Custom logger instance. If `None`, uses `StdOutLogger` |

#### MLflow Integration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_mlflow` | `bool` | `False` | Whether to use MLflow for logging |
| `mlflow_tracking_uri` | `str` | `None` | Tracking URI for MLflow |
| `mlflow_experiment_name` | `str` | `None` | Experiment name for MLflow |

#### Reproducibility & Error Handling

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | `int` | `0` | Random seed for reproducibility |
| `raise_on_exception` | `bool` | `True` | Whether to raise exceptions or continue on errors |

</details>

---

## See Also

- [Runner](runner.md) - Detailed API documentation for `optimize_agent_prompts`
- [Adapter](adapter.md) - GEPA adapter implementation details
- [Getting Started](../user-guides/getting-started.md) - Complete tutorial
