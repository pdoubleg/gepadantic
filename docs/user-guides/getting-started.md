# Getting Started

Welcome to GEPAdantic! This guide will help you get up and running with GEPA-driven prompt optimization for your structured input and output.

## Installation

Install GEPAdantic via:

```bash
uv add gepadantic
```

### Requirements

GEPAdantic requires:

- Python 3.10+
- pydantic-ai
- GEPA
- Pydantic v2

## Your First Optimization

Let's walk through a simple example: optimizing a sentiment classification agent.

### 1. Define Your Models

First, define the input and output structure using Pydantic:

```python
from pydantic import BaseModel, Field

class ReviewInput(BaseModel):
    """Classify sentiment of product reviews."""
    review: str = Field(description="The product review text to analyze")

class SentimentOutput(BaseModel):
    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float = Field(description="Confidence score 0-1")
```

### 2. Prepare Your Data

Create a dataset with labeled examples:

```python
from gepadantic.data_utils import prepare_train_val_sets

# Raw data as list of dictionaries
data = [
    {'review': 'This product is amazing!', 'label': 'positive'},
    {'review': 'Terrible quality, very disappointed', 'label': 'negative'},
    {'review': 'It works as expected', 'label': 'neutral'},
    {'review': 'Best purchase ever!', 'label': 'positive'},
    {'review': 'Broke after one day', 'label': 'negative'},
    # ... add more examples
]

# Convert to GEPA format and split
trainset, valset = prepare_train_val_sets(
    data=data,
    input_model=ReviewInput,
    input_keys=['review'],  # Maps to ReviewInput fields
    metadata_keys=['label'],  # Stored as metadata for evaluation
    train_ratio=0.7  # 70% training, 30% validation
)
```

### 3. Define Your Metric

The metric function evaluates how well your agent performs:

```python
def metric(data_inst, output):
    """
    Evaluate sentiment classification.
    
    Returns:
        (score, feedback): Score 0-1, feedback string for GEPA
    """
    # Check if agent execution succeeded
    if not output.success:
        return 0.0, f"Agent failed: {output.error_message}"
    
    # Compare prediction to ground truth
    predicted = output.result.sentiment
    expected = data_inst.metadata['label']
    
    if predicted == expected:
        return 1.0, f"Correct: {predicted}"
    else:
        return 0.0, f"Wrong: expected {expected}, got {predicted}"
```

### 4. Run Optimization

Use the high-level scaffold API for quick setup:

```python
from gepadantic import GepaConfig, run_optimization_pipeline

config = GepaConfig(
    agent_model="openai:gpt-4o-mini",
    agent_instructions="You are a sentiment analysis expert. Classify reviews accurately.",
    input_type=ReviewInput,
    output_type=SentimentOutput,
    trainset=trainset,
    valset=valset,
    metric=metric,
    auto="light"  # Preset: 3 rounds, batch size 10
)

# Run the optimization
result = run_optimization_pipeline(config)

# View results
print(f"Initial score: {result.initial_score:.2f}")
print(f"Best score: {result.best_score:.2f}")
print(f"Improvement: {result.best_score - result.initial_score:.2f}")
```

### 5. Use the Optimized Agent

Apply the best candidate to your agent for production use:

```python
# Get the optimized agent
optimized_agent = result.optimized_agent

# Or apply to existing agent
with result.apply_best(your_agent):
    result = await your_agent.run_signature(
        ReviewInput(review="Great value for the price!")
    )
    print(result.data)
```

## Understanding the Results

After optimization, you'll get a detailed result object:

```python
# Initial candidate (before optimization)
print(result.initial_candidate)
# {
#   'instructions': 'You are a sentiment analysis expert...',
#   'signature:ReviewInput:review:desc': 'The product review text to analyze'
# }

# Best candidate (after optimization)
print(result.best_candidate)
# {
#   'instructions': 'Analyze product reviews and classify sentiment as positive, negative, or neutral. Consider context and tone carefully...',
#   'signature:ReviewInput:review:desc': 'Customer review text requiring sentiment classification (positive/negative/neutral)'
# }

# Scores by round
for i, score in enumerate(result.validation_scores):
    print(f"Round {i}: {score:.2f}")
```

## Common Patterns

### Pattern: Start Simple, Then Expand

Begin with a small dataset and "light" preset:

```python
# Quick experiment with 10 examples
config = GepaConfig(
    agent_model="gpt-4o-mini",
    input_type=InputModel,
    output_type=OutputModel,
    trainset=trainset[:10],
    valset=valset[:10],
    metric=metric,
    auto="light"  # Fast iteration
)
```

Then scale up once you're happy with the setup:

```python
# Full optimization with all data
config = GepaConfig(
    agent_model="gpt-4o",  # Upgrade to better model
    input_type=InputModel,
    output_type=OutputModel,
    trainset=trainset,  # Full dataset
    valset=valset,
    metric=metric,
    auto="heavy"  # More thorough optimization
)
```

### Pattern: Iterative Development with Caching

Enable caching to speed up experimentation:

```python
from gepadantic.cache import CacheManager

cache = CacheManager(cache_dir="./optimization_cache")

config = GepaConfig(
    agent_model="gpt-4o-mini",
    input_type=InputModel,
    output_type=OutputModel,
    trainset=trainset,
    valset=valset,
    metric=metric,
    cache_manager=cache,  # Enable caching
    auto="light"
)

# First run: makes API calls
result1 = run_optimization_pipeline(config)

# Adjust and rerun: uses cache for repeated evaluations
config.agent_instructions = "Updated instructions..."
result2 = run_optimization_pipeline(config)
```

### Pattern: Custom Metric with Multiple Criteria

Combine multiple evaluation criteria:

```python
def comprehensive_metric(data_inst, output):
    """Evaluate on multiple criteria."""
    if not output.success:
        return 0.0, "Failed execution"
    
    # Criterion 1: Correctness
    is_correct = output.result.sentiment == data_inst.metadata['label']
    correctness_score = 1.0 if is_correct else 0.0
    
    # Criterion 2: Confidence calibration
    confidence = output.result.confidence
    if is_correct:
        # Reward high confidence on correct predictions
        confidence_score = confidence
    else:
        # Penalize high confidence on wrong predictions
        confidence_score = 1.0 - confidence
    
    # Combine scores (weighted average)
    final_score = 0.8 * correctness_score + 0.2 * confidence_score
    
    # Provide detailed feedback
    feedback = f"Correct: {is_correct}, Confidence: {confidence:.2f}"
    
    return final_score, feedback
```

## Troubleshooting

### "No improvement in validation score"

If you're not seeing improvements:

1. **Check your metric**: Make sure it's returning meaningful scores and feedback
2. **Increase batch size**: Small batches may not capture the distribution well
3. **Add more rounds**: Complex tasks may need more iterations
4. **Improve data quality**: Ensure your training data is representative and labeled correctly

### "Agent fails on some inputs"

The adapter handles failures gracefully:

```python
def robust_metric(data_inst, output):
    """Handle both success and failure cases."""
    if not output.success:
        # Provide specific feedback about the failure
        error_msg = output.error_message or "Unknown error"
        return 0.0, f"Failed: {error_msg}"
    
    # Normal evaluation
    # ...
```

### "Optimization is too slow"

To speed up optimization:

1. **Use caching**: Avoid re-evaluating the same candidates
2. **Reduce batch size**: Smaller batches mean faster iterations
3. **Use a faster model**: Try `gpt-4o-mini` instead of `gpt-4o`
4. **Limit validation set**: Use a smaller validation set for faster checks

```python
config = GepaConfig(
    agent_model="gpt-4o-mini",  # Faster model
    trainset=trainset,
    valset=valset[:20],  # Smaller validation set
    metric=metric,
    cache_manager=cache,  # Enable caching
    max_rounds=3,  # Fewer rounds
    batch_size=5  # Smaller batches
)
```

## Working with String Inputs

While structured inputs (Pydantic models) are powerful, sometimes you need to work with simple string prompts. GEPAdantic fully supports `input_type=str` for these cases.

### When to Use String Inputs

Use `input_type=str` when:

- You have unstructured text data (documents, medical records, transcripts)
- Your input doesn't fit neatly into fields
- You want to optimize just the agent instructions, not input field descriptions
- You're migrating from a traditional prompt-based workflow

### Basic String Input Example

```python
from gepadantic import GepaConfig, run_optimization_pipeline
from gepadantic.schema import DataInstWithPrompt
from pydantic_ai.messages import UserPromptPart

# Define output structure
class Summary(BaseModel):
    key_points: list[str] = Field(description="Main points from the text")
    sentiment: str = Field(description="Overall sentiment")

# Create dataset with string inputs
dataset = []
for idx, text in enumerate(documents):
    data_inst = DataInstWithPrompt(
        user_prompt=UserPromptPart(content=text),
        metadata={"expected_sentiment": labels[idx]},
        case_id=f"doc-{idx}"
    )
    dataset.append(data_inst)

# Split into train/val
trainset = dataset[:70]
valset = dataset[70:]

# Define metric
def metric(data_inst, output):
    if not output.success:
        return 0.0, f"Failed: {output.error_message}"
    
    predicted = output.result.sentiment
    expected = data_inst.metadata["expected_sentiment"]
    return 1.0 if predicted == expected else 0.0, None

# Run optimization with str input
config = GepaConfig(
    agent_model="openai:gpt-4o-mini",
    agent_instructions="Analyze the document and extract key information.",
    input_type=str,  # Use string inputs
    output_type=Summary,
    trainset=trainset,
    valset=valset,
    metric=metric,
    auto="light"
)

result = run_optimization_pipeline(config)
```

### Creating String Input Datasets

The key is using `DataInstWithPrompt` instead of `DataInstWithInput`:

```python
from gepadantic.schema import DataInstWithPrompt
from pydantic_ai.messages import UserPromptPart

def create_string_dataset(
    texts: list[str],
    labels: list[dict],
) -> list[DataInstWithPrompt]:
    """Convert raw text data to GEPA format."""
    dataset = []
    
    for idx, text in enumerate(texts):
        data_inst = DataInstWithPrompt(
            user_prompt=UserPromptPart(content=text),
            metadata=labels[idx],  # Store labels/metadata for evaluation
            case_id=f"item-{idx}"
        )
        dataset.append(data_inst)
    
    return dataset

# Example usage
texts = [
    "Patient presented with severe headache and nausea...",
    "Customer called regarding billing issue with last month's charge...",
    "The quarterly report shows a 15% increase in revenue..."
]

labels = [
    {"category": "medical", "urgency": "high"},
    {"category": "billing", "urgency": "medium"},
    {"category": "financial", "urgency": "low"}
]

dataset = create_string_dataset(texts, labels)
```

### Real-World Example: Medical Records

Here's a complete example analyzing medical records:

```python
from pydantic import BaseModel, Field
from gepadantic import GepaConfig, run_optimization_pipeline
from gepadantic.schema import DataInstWithPrompt
from pydantic_ai.messages import UserPromptPart

# Output structure
class MedicalChronology(BaseModel):
    """Chronological summary of medical events."""
    events: list[str] = Field(description="List of medical events in order")
    date_of_service: str = Field(description="Date of the medical service")

# Load your medical records
medical_records = [
    {
        "text": "Patient John Doe presented on 2024-01-15 with complaints of...",
        "date_of_service": "2024-01-15",
        "expected_events": ["Initial consultation", "Blood work ordered"]
    },
    # ... more records
]

# Create dataset
def create_medical_dataset(records):
    dataset = []
    for idx, record in enumerate(records):
        data_inst = DataInstWithPrompt(
            user_prompt=UserPromptPart(content=record["text"]),
            metadata={
                "date_of_service": record["date_of_service"],
                "expected_events": record["expected_events"]
            },
            case_id=f"record-{idx}"
        )
        dataset.append(data_inst)
    return dataset

dataset = create_medical_dataset(medical_records)
trainset = dataset[:int(len(dataset) * 0.7)]
valset = dataset[int(len(dataset) * 0.7):]

# Metric function
def medical_metric(data_inst, output):
    if not output.success:
        return 0.0, f"Failed: {output.error_message}"
    
    # Check if date matches
    predicted_date = output.result.date_of_service
    expected_date = data_inst.metadata["date_of_service"]
    
    if predicted_date != expected_date:
        return 0.0, f"Wrong date: expected {expected_date}, got {predicted_date}"
    
    # Check event extraction
    predicted_events = output.result.events
    expected_events = data_inst.metadata["expected_events"]
    
    # Simple overlap metric
    overlap = len(set(predicted_events) & set(expected_events))
    score = overlap / len(expected_events) if expected_events else 0.0
    
    return score, f"Found {overlap}/{len(expected_events)} events"

# Run optimization
config = GepaConfig(
    agent_model="openai:gpt-4o-mini",
    agent_instructions="Extract medical events from the record in chronological order.",
    input_type=str,  # String input for unstructured medical text
    output_type=MedicalChronology,
    trainset=trainset,
    valset=valset,
    metric=medical_metric,
    auto="medium"
)

result = run_optimization_pipeline(config)
print(f"Optimized instructions: {result.best_candidate['instructions']}")
```

### String Input vs Structured Input

| Feature | String Input (`str`) | Structured Input (`BaseModel`) |
|---------|---------------------|-------------------------------|
| **Input Format** | Raw text string | Pydantic model with fields |
| **Field Descriptions** | N/A | Each field has optimizable description |
| **Use Case** | Unstructured data | Multi-field structured data |
| **GEPA Optimizes** | Agent instructions only | Instructions + field descriptions |
| **Type Safety** | Basic (string) | Full Pydantic validation |
| **Dataset Type** | `DataInstWithPrompt` | `DataInstWithInput` |

### Key Differences

**With string inputs:**
- Only agent instructions are optimized
- Input is passed directly to the agent
- No field-level descriptions to optimize
- Simpler for unstructured text

**With structured inputs:**
- Agent instructions AND field descriptions are optimized
- Input is formatted with XML tags for each field
- Each field description can be improved by GEPA
- Better for multi-field data

### Migration Path

You can start with string inputs and migrate to structured inputs later:

```python
# Phase 1: Start with strings
config = GepaConfig(
    input_type=str,
    # ... other config
)

# Phase 2: Migrate to structured inputs when ready
class StructuredInput(BaseModel):
    """Process medical records."""
    text: str = Field(description="Medical record text")
    patient_id: str = Field(description="Patient identifier")
    date: str = Field(description="Date of service")

config = GepaConfig(
    input_type=StructuredInput,  # Now using structured input
    # ... other config
)
```

## Tips for Success

1. **Start with good baseline prompts**: GEPA optimization will evolve the initial seed prompt
2. **Provide detailed feedback**: The quality of metric feedback directly impacts optimization
3. **Use validation sets**: They help prevent overfitting to training data
4. **Monitor token usage**: Use `config.estimated_metric_calls` before kicking off runs to see the expected usage. Use `max_metric_calls` to set an explicit cap. For advanced usage limits see pydantic-ai [UsageLimits](https://ai.pydantic.dev/api/models/#pydantic_ai.models.UsageLimits)
5. **Experiment with batch sizes**: Balance between cost and quality. For example, changing the default `reflection_minibatch_size` for greater batch diversity.
