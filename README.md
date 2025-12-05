# GEPAdantic

<p align="center">
  <img src="https://github.com/pdoubleg/gepadantic/blob/main/assets/gepadantic.png" alt="GEPAdantic" width="500">
</p>

GEPA-driven prompt optimization for [pydantic-ai](https://github.com/pydantic/pydantic-ai) agents.

> [!NOTE]
> There is at least one other repo working on the same thing. See this [issue](https://github.com/pydantic/pydantic-ai/issues/3179) for more info. Unlike the project noted in the issue, which is a full re-write of GEPA, here we rely on the **canonical GEPA api** by simply providing a bridge between the two systems.

## Core Components

### SignatureAgent

Adds `input_type` to pydantic-ai agents, similar to `output_type`. Enables structured inputs and handles text component updates for GEPA optimization.

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from gepadantic import SignatureAgent, get_openai_model

class QueryInput(BaseModel):
    """Answer questions about geography."""
    question: str = Field(description="The geography question")
    context: str = Field(description="Additional context")

class Answer(BaseModel):
    answer: str
    confidence: str

# Create and wrap agent
model = get_openai_model('gpt-4o')
agent = Agent(model, output_type=Answer)
signature_agent = SignatureAgent(agent, input_type=QueryInput)

# Run with structured input
sig = QueryInput(question="What's the capital of France?", context="Current political capital")
result = await signature_agent.run_signature(sig)
```

### Adapter

Bridges pydantic-ai agents with GEPA's optimization API. Handles evaluation, trajectory capture, and reflection for prompt improvement.

```python
from gepadantic.adapter import PydanticAIGEPAAdapter

# Adapter connects your agent to GEPA
adapter = PydanticAIGEPAAdapter(
    agent=signature_agent,
    metric=your_metric_function,
    input_type=QueryInput,
)

# GEPA uses the adapter to evaluate and optimize
# (called internally by the runner)
```

### Runner

Orchestrates the GEPA optimization pipeline. Runs iterative prompt evolution using reflection and evaluation.

```python
from gepadantic import optimize_agent_prompts

result = optimize_agent_prompts(
    agent=signature_agent,
    trainset=train_data,
    valset=val_data,
    metric=metric_fn,
    input_type=QueryInput,
    auto="light",  # or max_metric_calls, max_full_evals
)

print(f"Best score: {result.best_score}")
print(f"Optimized prompts: {result.best_candidate}")
```

### Scaffolding

Config-based convenience wrapper for simplified setup and execution.

```python
from gepadantic import GepaConfig, run_optimization_pipeline

config = GepaConfig(
    agent_model="gpt-4o-mini",
    agent_instructions="You are a helpful assistant",
    input_type=QueryInput,
    output_type=Answer,
    trainset=train_data,
    valset=val_data,
    metric=metric_fn,
    auto="medium",  # light, medium, or heavy
    optimize_tools=True,
)

result = run_optimization_pipeline(config)
```

### Data Utils

Helpers to convert common data formats into GEPA-compatible `DataInstWithInput` instances.

```python
from gepadantic.data_utils import dataframe_to_dataset, json_to_dataset, split_dataset

# From pandas DataFrame
dataset = dataframe_to_dataset(
    df,
    row_mapper=lambda row: QueryInput(question=row['q'], context=row['ctx']),
    metadata_cols=['label', 'difficulty']
)

# From JSON file
dataset = json_to_dataset(
    'data.json',
    input_mapper=lambda d: QueryInput(question=d['question'], context=d['context']),
    metadata_keys=['label']
)

# Split into train/val
trainset, valset = split_dataset(dataset, train_ratio=0.7, shuffle=True)
```

## Example: Before & After Optimization

Here's an example from optimizing a math problem solver. GEPA improved the score from **0.75 → 0.83** by refining prompts across multiple components:

**Before:**

```python
{
  "instructions": "Solve math problems by calling the `run_python` sandbox tool. Write complete Python scripts with all necessary imports and print the final result.",
  "signature:MathProblemInput:problem:desc": "A math problem that needs an exact numeric answer.",
  "tool:run_python:param:code": "Complete Python script to run. Include all required imports and print the final answer."
}
```

**After:**

```python
{
  "instructions": "Solve math problems by writing and executing complete Python scripts using only the Python standard library. Always include all required imports and use print to display the final answer. If a shortcut or algebraic simplification can be applied (e.g., simplifying factorial terms), explain or use it. Ensure the answer is exact and shows the correct final value.",
  "signature:MathProblemInput:problem:desc": "A math problem requiring an exact numeric answer. It may involve factorials (noted as n!), exponents, integer sequences, or sums. State if the answer should be exactly integer, rational, or decimal.",
  "tool:run_python:param:code": "A complete, stand-alone Python script (with all necessary imports) that prints the exact answer to the math problem. Apply algebraic or arithmetic simplifications where possible (for example, simplify factorial ratios or use properties of powers)."
}
```

GEPA discovered that emphasizing "algebraic simplifications" and "exact answers" improved performance on the validation set.

## Quick Start

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from gepadantic import SignatureAgent, GepaConfig, run_optimization_pipeline
from gepadantic.data_utils import prepare_train_val_sets

# 1. Define input/output models
class TaskInput(BaseModel):
    """Classify sentiment of text."""
    text: str = Field(description="Text to analyze")

class TaskOutput(BaseModel):
    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float

# 2. Prepare data
trainset, valset = prepare_train_val_sets(
    data=[{'text': 'Great!', 'label': 'positive'}, ...],
    input_model=TaskInput,
    input_keys=['text'],
    metadata_keys=['label'],
    train_ratio=0.7
)

# 3. Define metric
def metric(data_inst, output):
    if output.success:
        score = 1.0 if output.result.sentiment == data_inst.metadata['label'] else 0.0
        feedback = "Correct" if score == 1.0 else f"Expected {data_inst.metadata['label']}"
        return score, feedback
    return 0.0, "Failed"

# 4. Configure and run optimization
config = GepaConfig(
    agent_model="gpt-4o-mini",
    agent_instructions="Classify sentiment accurately",
    input_type=TaskInput,
    output_type=TaskOutput,
    trainset=trainset,
    valset=valset,
    metric=metric,
    auto="light",
)

result = run_optimization_pipeline(config)

# 5. Use optimized agent
with result.apply_best(agent):
    optimized_result = await agent.run_signature(TaskInput(text="Amazing product!"))
```
