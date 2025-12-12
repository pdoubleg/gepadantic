# Adapter

The `PydanticAIGEPAAdapter` is the bridge that connects pydantic-ai agents to the GEPA optimization engine. It handles evaluation, trajectory capture, reflection, and proposal generation - the core mechanisms that enable GEPA to iteratively improve your prompts.


!!! warning "Automatic Adapter Management"
    When using the `GepaConfig` scaffolding (recommended approach), the adapter is created and managed automatically. You typically only need to work with the adapter directly when you need fine-grained control over the optimization process, which should be rare.
    
    For most use cases, use `run_optimization_pipeline()` with a `GepaConfig` instead of manually creating adapters.


## Core Concept

The adapter implements the [`GEPAAdapter`](https://github.com/gepa-ai/gepa/blob/main/src/gepa/core/adapter.py) protocol, translating between pydantic-ai's agent execution model and GEPA's optimization workflow. It manages:

- **Evaluation**: Running your agent on data instances and computing scores
- **Trajectory Capture**: Recording message history and execution details for reflection
- **Reflection**: Analyzing failures and successes to generate improvement insights
- **Proposal Generation**: Creating new candidate prompts based on reflection
- **Caching**: Optional caching to speed up optimization and reduce API costs

## Basic Usage

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from gepadantic import PydanticAIGEPAAdapter, SignatureAgent

# Define your models
class TaskInput(BaseModel):
    """Classify sentiment of text."""
    text: str = Field(description="Text to analyze")

class TaskOutput(BaseModel):
    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float

# Create your agent
agent = SignatureAgent(
    Agent('openai:gpt-4.1-nano', output_type=TaskOutput),
    input_type=TaskInput
)

# Define a metric function
def metric(data_inst, output):
    """Compute score and provide feedback."""
    if output.success:
        # Check if prediction matches expected label
        is_correct = output.result.sentiment == data_inst.metadata['label']
        score = 1.0 if is_correct else 0.0
        feedback = "Correct classification" if is_correct else f"Expected {data_inst.metadata['label']}, got {output.result.sentiment}"
        return score, feedback
    return 0.0, f"Failed: {output.error_message}"

# Create the adapter
adapter = PydanticAIGEPAAdapter(
    agent=agent,
    metric=metric,
    input_type=TaskInput
)
```

## Key Components

### 1. Agent Execution

The adapter handles two types of agents:

**Regular Agents** (with `DataInstWithPrompt`):

```python
from gepadantic.types import DataInstWithPrompt

# Data instance with direct prompt
data = DataInstWithPrompt(
    case_id="example_1",
    user_prompt=UserPrompt(content="Classify: 'This is amazing!'"),
    metadata={'label': 'positive'}
)

# Adapter runs: agent.run_sync(data.user_prompt.content)
```

**SignatureAgents** (with structured input):

```python
from gepadantic.types import DataInst

# Data instance with structured input
data = DataInst(
    case_id="example_1",
    input=TaskInput(text="This is amazing!"),
    metadata={'label': 'positive'}
)

# Adapter runs: agent.run_signature_sync(data.input)
```

### 2. Metric Function

The metric function is central to optimization - it evaluates each output and provides feedback:

```python
def metric(data_inst: DataInst, output: RolloutOutput) -> tuple[float, str | None]:
    """
    Evaluate agent output and provide feedback.
    
    Args:
        data_inst: The input data instance
        output: The agent's output (includes success flag and result/error)
    
    Returns:
        (score, feedback): Score is 0-1 (higher is better), feedback is optional but recommended
    """
    if not output.success:
        return 0.0, f"Error: {output.error_message}"
    
    # Your evaluation logic here
    score = compute_score(data_inst, output.result)
    feedback = generate_feedback(data_inst, output.result)
    
    return score, feedback
```

**Metric Best Practices:**

- **Return meaningful feedback**: The feedback string helps GEPA understand what went wrong
- **Use normalized scores**: Keep scores between 0 and 1 for consistency
- **Handle failures gracefully**: Check `output.success` before accessing `output.result`

### 3. Trajectory Capture

When `capture_traces=True`, the adapter records execution details:

```python
from gepadantic.types import Trajectory

# Trajectory includes:
trajectory = Trajectory(
    messages=[...],           # All ModelMessages exchanged
    instructions="...",       # Agent instructions used
    final_output=result,      # Final output from agent
    error=None,               # Error message if failed
    usage={...},              # Token usage statistics
    data_inst=data_inst,      # Original input
    metric_feedback="..."     # Feedback from metric function
)
```

This trajectory data powers GEPA's reflection mechanism.

### 4. Reflection and Proposal

The adapter builds reflection datasets from trajectories:

```python
# Reflection record structure
reflection_record = {
    'messages': [...],           # Conversation history
    'final_output': result,      # Agent's output
    'score': 0.85,              # Metric score
    'success': True,            # Whether execution succeeded
    'feedback': "...",          # Metric feedback
    'instructions': "...",      # Instructions that were used
}
```

GEPA uses these records to propose improved prompts via reflection:

```python
# Adapter calls propose_new_texts internally
new_candidate = adapter.propose_new_texts(
    candidate=current_prompts,
    reflective_dataset=reflection_records,
    components_to_update=['instructions', 'signature:TaskInput:text:desc']
)
```

## Additional Features

### Input Type Optimization

When you provide an `input_type`, the adapter optimizes both agent instructions AND field descriptions:

```python
class DetailedInput(BaseModel):
    """Process customer reviews."""
    review_text: str = Field(description="The customer review to analyze")
    product_category: str = Field(description="Category of the product being reviewed")
    rating: int = Field(description="Star rating (1-5)")

adapter = PydanticAIGEPAAdapter(
    agent=agent,
    metric=metric,
    input_type=DetailedInput  # Will optimize all field descriptions
)
```

This creates optimizable components:

- `instructions`: Agent's main instructions
- `signature:DetailedInput:review_text:desc`: Description for review_text field
- `signature:DetailedInput:product_category:desc`: Description for product_category field
- `signature:DetailedInput:rating:desc`: Description for rating field

### Reflection Sampling

For some datasets, you want to control how reflection records are sampled:

```python

def sample_reflections(records, max_records):
    """Custom sampling strategy for reflection."""
    # Sort by score to prioritize edge cases
    sorted_records = sorted(records, key=lambda r: r['score'])
    
    # Take mix of low and high scores
    low_performers = sorted_records[:max_records//2]
    high_performers = sorted_records[-(max_records//2):]
    
    return low_performers + high_performers

adapter = PydanticAIGEPAAdapter(
    agent=agent,
    metric=metric,
    reflection_sampler=sample_reflections
)
```

### Custom Reflection Model

Use a different (often better) model for reflection:

```python

adapter = PydanticAIGEPAAdapter(
    agent=agent,  # Uses gpt-4.1-nano for main task
    metric=metric,
    reflection_model="gpt-5"  # Uses better model for reflection
)
```

### Caching

Enable caching to resume optimization runs and save costs:

```python
from gepadantic.cache import CacheManager, create_cached_metric

cache = CacheManager(cache_dir="./optimization_cache")

cached_metric = create_cached_metric(metric)

adapter = PydanticAIGEPAAdapter(
    agent=agent,
    metric=cached_metric,
    cache_manager=cache
)
```

The cache stores:

- Agent runs (keyed by candidate + input)
- Metric evaluations (keyed by candidate + input + output)

This means if GEPA re-evaluates the same candidate, no API calls are made.

## Evaluation Workflow

Here's how the adapter evaluates a batch:

```python
# 1. Apply candidate prompts to agent
candidate = {
    'instructions': 'Classify sentiment accurately...',
    'signature:TaskInput:text:desc': 'Customer review text to classify'
}

# 2. Evaluate batch
results = adapter.evaluate(
    batch=[data1, data2, data3],
    candidate=candidate,
    capture_traces=True  # Needed for reflection
)

# 3. Results contain:
results.outputs      # List[RolloutOutput] - agent outputs
results.scores       # List[float] - metric scores
results.trajectories # List[Trajectory] - execution traces (if captured)
```

The process for each data instance:

1. **Check cache** (if enabled) for this candidate + input
2. **Run agent** with applied candidate
3. **Compute metric** score and feedback
4. **Capture trajectory** (if requested)
5. **Store in cache** (if enabled)

## Integration with Runner

The adapter is typically used through the `optimize_agent_prompts` function:

```python
from gepadantic import optimize_agent_prompts

result = optimize_agent_prompts(
    adapter=adapter,
    trainset=train_data,
    valset=val_data,
    # GEPA config options...
)

# The function orchestrates the optimization loop:
# 1. Evaluate current candidate on training batch
# 2. Build reflection dataset from trajectories
# 3. Propose new candidate via reflection
# 4. Validate new candidate on validation set
# 5. Keep best candidate and repeat
```

## Error Handling

The adapter handles errors gracefully:

```python
# If agent execution fails:
try:
    result = agent.run_signature_sync(data.input)
except Exception as e:
    # Adapter creates error output
    output = RolloutOutput.from_error(e)
    trajectory = Trajectory(messages=[], final_output=None, error=str(e))
    score = 0.0  # Failed executions get score 0
```

Errors don't crash optimization - they're treated as low-scoring examples that reflection learns from.

## Usage Tracking

The adapter tracks token usage across all operations:

```python
# After optimization
total_usage = adapter.gepa_usage

print(f"Total requests: {total_usage.input_tokens}")
print(f"Total responses: {total_usage.output_tokens}")
print(f"Total cost: ${total_usage.requests}")
```

This includes:

- Agent runs (main task)
- Reflection calls (proposal generation)


## Example: Complete Workflow

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from gepadantic import (
    SignatureAgent,
    PydanticAIGEPAAdapter,
    optimize_agent_prompts,
    prepare_train_val_sets
)

# 1. Define models
class ReviewInput(BaseModel):
    """Analyze product reviews."""
    review: str = Field(description="Customer review text")

class Sentiment(BaseModel):
    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float

# 2. Create agent
base_agent = Agent(
    'openai:gpt-4o-mini',
    system_prompt="Classify sentiment of product reviews",
    output_type=Sentiment
)
agent = SignatureAgent(base_agent, input_type=ReviewInput)

# 3. Prepare data
data = [
    {'review': 'Great product!', 'label': 'positive'},
    {'review': 'Terrible quality', 'label': 'negative'},
    # ... more examples
]

trainset, valset = prepare_train_val_sets(
    data=data,
    input_model=ReviewInput,
    input_keys=['review'],
    metadata_keys=['label']
)

# 4. Define metric
def metric(data_inst, output):
    if output.success:
        correct = output.result.sentiment == data_inst.metadata['label']
        score = 1.0 if correct else 0.0
        feedback = "Correct" if correct else f"Expected {data_inst.metadata['label']}"
        return score, feedback
    return 0.0, f"Error: {output.error_message}"

# 5. Create adapter
adapter = PydanticAIGEPAAdapter(
    agent=agent,
    metric=metric,
    input_type=ReviewInput
)

# 6. Run optimization
result = optimize_agent_prompts(
    adapter=adapter,
    trainset=trainset,
    valset=valset,
    gepa_config={'n': 10}  # Generate 10 candidates
)

# 7. Use best candidate
print(f"Best score: {result.best_score}")
print(f"Best candidate: {result.best_candidate}")

# Apply to agent for inference
with result.apply_best(agent):
    new_result = await agent.run_signature(
        ReviewInput(review="Amazing value for money!")
    )
```

## See Also

- [SignatureAgent](signature-agent.md): Structured inputs for agents
- [Runner](runner.md): Orchestrating the optimization loop
- [Scaffold](scaffold.md): Config-based setup wrapper
