# Runner

The `runner` module provides a **lower-level interface** for GEPA optimization. For most use cases, the [config-based scaffolding API](scaffold.md) is preferred as it provides a simpler, more declarative way to configure optimization runs.

!!! warning "When to Use the Runner"

    Use `optimize_agent_prompts` directly when you need:

    - Fine-grained control over GEPA's optimization parameters
    - Custom logging, caching, or callback integration
    - To programmatically configure optimization without config files

    For standard workflows, **prefer the scaffold API** which handles these details automatically.

## How It Works

The `optimize_agent_prompts` function is the main entry point for low-level optimization. Its core responsibility is to **prepare inputs for `gepa.api.optimize`** by:

1. **Extracting the seed candidate** - Pulls the initial prompts from your agent and optional signature
2. **Creating the adapter** - Wraps your agent, metric, and input specification in a `PydanticAIGEPAAdapter`
3. **Configuring the reflection language model** - Sets up the LM used to propose new prompts
4. **Managing caching** - Optionally creates a `CacheManager` for resumable optimization
5. **Calling `gepa.api.optimize`** - Delegates to GEPA's core optimization engine
6. **Packaging results** - Returns a `GepaOptimizationResult` with the optimized prompts

Essentially, it bridges between your pydantic-ai agent and GEPA's optimization algorithm.

## Basic Usage

```python
from gepadantic import optimize_agent_prompts, SignatureAgent
from pydantic_ai import Agent
from pydantic import BaseModel, Field

# Define your input specification
class QueryInput(BaseModel):
    """Answer user queries"""
    query: str = Field(description="The user's question")

# Create a SignatureAgent (required for component optimization)
base_agent = Agent("openai:gpt-4o", system_prompt="You are a helpful assistant.")
agent = SignatureAgent(base_agent, input_type=QueryInput)

# Define your metric
def metric(data_inst, output):
    # Compare output to expected result
    score = 1.0 if output.data == data_inst.expected else 0.0
    feedback = "Correct!" if score == 1.0 else "Incorrect answer."
    return score, feedback

# Optimize - will optimize system_prompt, instructions, and field descriptions
result = optimize_agent_prompts(
    agent=agent,
    trainset=trainset,
    metric=metric,
    valset=valset,
    auto="light",  # Automatically set budget
)

# Use the optimized prompts
with result.apply_best(agent):
    response = agent.run_sync("What is 2+2?")
```

## Key Parameters

### Core Required Parameters

- **`agent`**: A `SignatureAgent` instance wrapping your pydantic-ai agent (required for component optimization)
- **`trainset`**: Training dataset (list of `DataInst` objects)
- **`metric`**: Function that evaluates outputs: `(data_inst, output) -> (score, feedback)`
- **`valset`**: Optional validation set (uses `trainset` if not provided)

### Input Specification

- **`input_type`**: Optional `InputSpec` for optimizing structured input field descriptions alongside agent prompts
- **`seed_candidate`**: Optional initial prompts (extracted from agent by default)

### Budget Control

Choose one approach:

- **`auto`**: Set to `"light"`, `"medium"`, or `"heavy"` for automatic budget sizing
- **`max_metric_calls`**: Explicit total metric call budget
- **`max_full_evals`**: Number of full evaluation passes

### Optimization Strategy

- **`reflection_model`**: Model to use for proposing new prompts (defaults to agent's model)
- **`candidate_selection_strategy`**: `"pareto"` (default), `"current_best"`, or `"epsilon_greedy"`
- **`module_selector`**: `"all"` (default) or `"round_robin"` for component selection
- **`use_merge`**: Enable merge strategy for combining candidates

### Caching & Resumability

- **`enable_cache`**: Enable caching for resumable runs
- **`cache_dir`**: Directory for cache files (defaults to `.gepa_cache`)
- **`cache_verbose`**: Log cache hits/misses

### Logging

- **`logger`**: Custom logger (defaults to `StdOutLogger`)
- **`display_progress_bar`**: Show progress during optimization
- **`run_dir`**: Directory to save results

## GepaOptimizationResult

The `GepaOptimizationResult` object contains all optimization results and metadata:

### Key Attributes

```python
result = optimize_agent_prompts(...)

# Best optimized prompts
result.best_candidate  # dict[str, str] - optimized prompt components
result.best_score      # float - validation score

# Original prompts for comparison
result.original_candidate  # dict[str, str] - initial prompts
result.original_score      # float | None - baseline score

# Optimization metrics
result.num_iterations    # int - optimization iterations performed
result.num_metric_calls  # int - total metric evaluations
result.gepa_usage       # RunUsage - token/cost tracking
```

### Key Methods

**Apply optimized prompts:**

```python
# As context manager
with result.apply_best(agent):
    response = agent.run_sync("Query here")

# With signature
with result.apply_best_to(agent=agent, input_type=input_spec):
    response = agent.run_sync("Query here", ...)
```

**Calculate improvement:**

```python
improvement = result.improvement_ratio()  # Returns (best - original) / original
if improvement:
    print(f"Improved by {improvement*100:.1f}%")
```

**Visualize optimization DAG:**

```python
# Get Graphviz DOT format for the optimization evolution graph
dot_graph = result.graphviz_dag
```

The DAG visualization shows:

- Nodes representing candidate programs (colored by performance)
- Edges showing the evolution/mutation relationships
- Special highlighting for the best program (cyan) and dominator programs (orange)

## Complete Example

```python
from gepadantic import optimize_agent_prompts, SignatureAgent
from pydantic_ai import Agent
from pydantic import BaseModel, Field

# Setup input specification
class MathInput(BaseModel):
    """Solve mathematical problems"""
    problem: str = Field(description="A math problem to solve")

# Setup data model
class MathProblem(BaseModel):
    problem: str
    expected: int

# Create SignatureAgent for component optimization
base_agent = Agent("openai:gpt-4o", system_prompt="Solve math problems step by step.")
agent = SignatureAgent(base_agent, input_type=MathInput)

def metric(data_inst, output):
    try:
        answer = int(output.data)
        correct = answer == data_inst.expected
        score = 1.0 if correct else 0.0
        feedback = "Correct" if correct else f"Expected {data_inst.expected}, got {answer}"
        return score, feedback
    except:
        return 0.0, "Could not parse answer"

trainset = [MathProblem(problem="2+2=?", expected=4), ...]
valset = [MathProblem(problem="3+5=?", expected=8), ...]

# Optimize - will optimize system_prompt, instructions, and field descriptions
result = optimize_agent_prompts(
    agent=agent,
    trainset=trainset,
    valset=valset,
    metric=metric,
    auto="medium",
    enable_cache=True,
    display_progress_bar=True,
)

print(f"Improved from {result.original_score:.2f} to {result.best_score:.2f}")
print(f"Used {result.num_metric_calls} metric calls")

# Use optimized agent
with result.apply_best(agent):
    result = agent.run_sync(MathInput(problem="What is 10+15?"))
```

## See Also

- [Scaffold](scaffold.md): Preferred config-based API for standard workflows
- [Adapter](adapter.md): Understanding the adapter that wraps your agent
- [Getting Started](../user-guides/getting-started.md): Quick start guide
