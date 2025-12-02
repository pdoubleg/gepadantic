# GEPAdantic

<p align="center">
  <img src="https://github.com/pdoubleg/gepadantic/blob/main/assets/gepadantic.png" alt="GEPAdantic" width="500">
</p>

GEPA-driven prompt optimization for [pydantic-ai](https://github.com/pydantic/pydantic-ai) agents.

> [!NOTE]
> There is at least one other repo working on the same thing. See this [issue](https://github.com/pydantic/pydantic-ai/issues/3179) for more info. Unlike the project noted in the issue, which is a full re-write of GEPA, here we rely on the **canonical GEPA api** by simply providing a bridge between the two systems.

## About

This library combines pydantic's data validation with GEPA's prompt optimization algorithm. It does this by implementing a GEPA [adapter](https://github.com/gepa-ai/gepa/blob/main/src/gepa/core/adapter.py) allowing any pydantic-ai agent to be plugged into the GEPA optimization api.

## Features

Two main things this library adds to pydantic-ai:

**1. SignatureAgent - Structured Inputs**

Inspired by [DSPy's signatures](https://dspy-docs.vercel.app/docs/building-blocks/signatures), `SignatureAgent` adds `input_type` support to pydantic-ai. Just like pydantic-ai uses `output_type` for structured outputs, SignatureAgent lets you define structured inputs:

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from gepadantic import SignatureAgent

class AnalysisInput(BaseModel):
    """Analyze the provided data and extract insights."""

    data: str = Field(description="The raw data to analyze")
    focus_area: str = Field(description="Which aspect to focus on")
    format: str = Field(description="Output format preference")

class AnalysisOutput(BaseModel):
    """The result of the analysis."""

    takeaways: list[str] = Field(description="A list of takeaways from the analysis")
    summary: str = Field(description="Executive summary of the analysis")

# Create base agent
base_agent = Agent(
    model="openai:gpt-4o",
    output_type=AnalysisOutput,
)

# Wrap with SignatureAgent to add input_type support
agent = SignatureAgent(
    base_agent,
    input_type=AnalysisInput,
)

# Run with structured input
result = await agent.run_signature(
    AnalysisInput(
        data="...",
        focus_area="performance",
        format="bullet points"
    )
)
```

The model docstring becomes system instructions, and field descriptions become input specs.

**2. Optimizable Components**

GEPA can optimize different parts of your agent:

- System prompts
- Signature (i.e., input model) field descriptions
- Tool/output model docstrings, descriptions and parameter docs (set `optimize_tools=True`)

All these text components evolve together using LLM-guided improvements:

```python
from gepadantic import GepaConfig, run_optimization_pipeline

# Define evaluation metric
def metric(input_data, output) -> float:
    # Return 0.0-1.0 score + feedback string
    return score, feedback

config = GepaConfig(
    agent=agent,
    input_type=AnalysisInput,
    output_type=AnalysisOutput,
    trainset=trainset,
    valset=valset,
    metric=metric,
    max_full_evals=10,
    optimize_tools=True,
)

# Optimize agent with SignatureAgent
result = run_optimization_pipeline(config)

# Access all optimized components
print(result.best_candidate.components)
# {
#   "instructions": "...",                           # System prompt
#   "signature:AnalysisInput:instructions": "...",   # Input schema docstring
#   "signature:AnalysisInput:data:desc": "...",      # Field description
#   "signature:AnalysisInput:focus_area:desc": "...",
#   "tool:my_tool:description": "...",               # If optimize_tools=True
#   "tool:my_tool:param_x:description": "...",
#   "tool:final_result:instructions": "...",           # If optimize_tools=True
#   "tool:final_result:param:takeaways:desc": "...",
#   "tool:final_result:param:summary:desc": "...",
#   ...
# }
```