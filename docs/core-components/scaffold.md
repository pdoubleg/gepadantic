# Config Based Scaffolding

The Scaffold provides a config-based convenience wrapper for simplified setup and execution of GEPA optimization.

## Overview

The `GepaConfig` class and `run_optimization_pipeline()` function provide a clean, declarative interface for setting up and running GEPA prompt optimization. Instead of manually configuring agents, adapters, and optimization parameters, you can specify everything in a single configuration object.

## Configuration Parameters

### Required Parameters

#### Core Data & Evaluation

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_type` | `type[BaseModel]` | Pydantic model class defining the structured input format |
| `output_type` | `type[BaseModel]` | Pydantic model class defining the expected output format |
| `trainset` | `Sequence[DataInstWithInput]` | List of training examples with inputs and metadata |
| `metric` | `Callable` | Function that evaluates agent outputs, returning `(score, feedback)` tuple |

#### Agent Configuration (Pick One)

You must provide **exactly one** of the following:

| Parameter | Type | Description |
|-----------|------|-------------|
| `agent` | `AbstractAgent` | A pre-configured PydanticAI agent to optimize |
| `agent_model` + `agent_instructions` | `str` + `str` | Model name (e.g., "gpt-4.1-mini") and system instructions to create an agent |

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

## Minimal Example

Here's a complete minimal example for text classification:

```python
from pydantic import BaseModel, Field
from gepadantic import GepaConfig, run_optimization_pipeline, DataInstWithInput

# Define input and output types
class TextInput(BaseModel):
    """Input text to classify."""
    text: str = Field(description="The text to classify")

class CategoryOutput(BaseModel):
    """Classification result."""
    category: str = Field(description="The predicted category: positive or negative")

# Create 10 dummy examples
dummy_data = [
    {"text": "I love this product!", "label": "positive"},
    {"text": "This is terrible and disappointing.", "label": "negative"},
    {"text": "Absolutely amazing experience!", "label": "positive"},
    {"text": "Waste of money, very upset.", "label": "negative"},
    {"text": "Best purchase I've made this year.", "label": "positive"},
    {"text": "Not what I expected, very poor quality.", "label": "negative"},
    {"text": "Highly recommend to everyone!", "label": "positive"},
    {"text": "Complete disaster, avoid at all costs.", "label": "negative"},
    {"text": "Exceeded all my expectations!", "label": "positive"},
    {"text": "Regret buying this, total failure.", "label": "negative"},
]

# Convert to DataInstWithInput format
trainset = [
    DataInstWithInput(
        input=TextInput(text=item["text"]),
        metadata={"label": item["label"]}
    )
    for item in dummy_data
]

# Define metric function
def sentiment_metric(data_inst, output):
    """Evaluate the classification output.
    
    Args:
        data_inst: The input data instance with metadata
        output: The agent's output (RolloutOutput)
        
    Returns:
        tuple: (score, feedback) where score is 1.0 for correct, 0.0 for incorrect
    """
    if output.success and output.result:
        correct = output.result.category == data_inst.metadata["label"]
        if correct:
            return 1.0, None
        else:
            return 0.0, f"Expected {data_inst.metadata['label']}, got {output.result.category}"
    return 0.0, "Failed to produce output"

# Configure and run optimization
config = GepaConfig(
    # Agent configuration
    agent_model="gpt-4.1-nano",
    agent_instructions="Classify the sentiment of the given text as positive or negative.",
    
    # Input/Output types
    input_type=TextInput,
    output_type=CategoryOutput,
    
    # Data
    trainset=trainset,
    
    # Evaluation
    metric=sentiment_metric,
    
    # Budget (use light auto-budget for quick testing)
    auto="light",
    
    # Optional: Use a better model for reflection
    reflection_model="gpt-4.1",
)

# Run optimization
result = run_optimization_pipeline(config)

# Access results
print(f"Best score: {result.best_score}")
print(f"Best candidate prompts: {result.best_candidate}")
print(f"Improvement: {result.improvement_ratio():.2%}")
```

## Output

The `run_optimization_pipeline()` function returns a `GepaOptimizationResult` object containing:

- `best_candidate`: Dict mapping component names to optimized text
- `best_score`: Best validation score achieved
- `num_iterations`: Number of optimization iterations performed
- `num_metric_calls`: Total number of metric evaluations
- `initial_score`: Score of the initial candidate (if available)
- Additional metadata and metrics

Results are automatically saved to JSON in the `output_dir` (default: `optimization_results/`) if `save_result=True`.

## Advanced Usage

### Using Pre-configured Agent

```python
from pydantic_ai import Agent
from gepadantic import SignatureAgent
from gepadantic.lm import get_openai_model

# Create your own agent
model = get_openai_model("gpt-4.1")
base_agent = Agent(
    model=model,
    instructions="Custom instructions here",
    output_type=CategoryOutput,
)
# Add a tool
@base_agent.tool
def get_the_weather(location: str) -> str:
    """Get the weather for a given location"""
    result = get_weather_function(location)
    return str(result)

# Wrap in signature agent
agent = SignatureAgent(base_agent, input_type=TextInput)

config = GepaConfig(
    agent=agent,  # Use pre-configured agent
    input_type=TextInput,
    # ... other params ...
)

# Optimizes tool description
result = run_optimization_pipeline(config)

```

### Budget Control

```python
# Option 1: Auto-budget (recommended for getting started)
config = GepaConfig(
    auto="light",    # Quick testing (fewer iterations)
    # auto="medium", # Balanced optimization
    # auto="heavy",  # Thorough optimization (more compute)
    # ... other params ...
)

# Option 2: Explicit full evaluations
config = GepaConfig(
    max_full_evals=10,  # 10 complete passes over train+val
    # ... other params ...
)

# Option 3: Explicit metric calls
config = GepaConfig(
    max_metric_calls=200,  # 200 individual metric evaluations
    # ... other params ...
)

# Get metric calls before execution
config.estimated_metric_calls
#> 60
```

## See Also

- [Runner](runner.md) - Lower-level optimization interface
- [Adapter](adapter.md) - GEPA adapter implementation details
- [Getting Started](../user-guides/getting-started.md) - Complete tutorial
