# Data Utils

Helper functions to convert common data formats into GEPA-compatible datasets for optimization.

## Overview

The `gepadantic.data_utils` module provides utilities to transform data from common formats (pandas DataFrames, JSON files, Python dictionaries) into the structured format required for GEPA optimization. This guide explains the core data models and shows practical examples of loading and preparing data.

## Core Data Models

### DataInstWithInput

`DataInstWithInput[InputModelT]` is the primary data container for GEPA optimization. Each instance represents a single training or validation example.

**Structure:**

- `input: InputModelT` - Your Pydantic model containing the structured input fields
- `message_history: list[ModelMessage] | None` - Optional conversation history for multi-turn scenarios. Only applicable to multi-turn workflows.
- `metadata: dict[str, Any]` - Additional data (labels, ground truth, etc.) **used by metrics**
- `case_id: str` - Unique identifier for tracking this example

!!! note "Data Instance with Input"

    > The optimizer runs your agent on each `DataInstWithInput`, producing **model outputs** that are **accessible to your metric** function using the **`metadata`** attribute.

### RolloutOutput

`RolloutOutput[OutputT]` wraps the result of executing your agent on a single input. This output is produced internally by the `PydanticAIGEPAAdapter`.

**Structure:**

- `result: OutputT | None` - The agent's output (your result model or structured output)
- `success: bool` - Whether execution completed without errors
- `error_message: str | None` - Error details if execution failed

!!! warning "Metric Data"

    > Your metric function should receive both the `RolloutOutput` (what the agent produced) and the original `DataInstWithInput` (including metadata with ground truth) to compute a score.

### Metric Function Signature

Your metric function should follow this pattern:

```python
from gepadantic.schema import DataInstWithInput, RolloutOutput

def my_metric(
    output: RolloutOutput[YourOutputModel],
    data_inst: DataInstWithInput[YourInputModel]
) -> float:
    """
    Args:
        output: The agent's execution result
        data_inst: The original input with metadata containing ground truth
    
    Returns:
        Score (higher is better, typically 0-1 or 0-100)
    """
    if not output.success or output.result is None:
        return 0.0
    
    # Access ground truth from metadata
    ground_truth = data_inst.metadata["label"]
    prediction = output.result.predicted_label
    
    # Compute and return score
    return 1.0 if prediction == ground_truth else 0.0
```

## Loading Data

### From a List of Dictionaries

The simplest approach when you already have Python dictionaries:

```python
from pydantic import BaseModel, Field
from gepadantic.data_utils import create_dataset_from_dicts

# Define your input model
class SentimentInput(BaseModel):
    text: str = Field(description="Text to analyze")
    context: str = Field(description="Additional context", default="")

# Your data
data = [
    {
        "text": "This product exceeded my expectations!",
        "context": "Product review",
        "label": "positive",
        "confidence": 0.95
    },
    {
        "text": "Disappointed with the customer service.",
        "context": "Service feedback",
        "label": "negative",
        "confidence": 0.88
    },
    {
        "text": "It's okay, nothing special.",
        "context": "Product review",
        "label": "neutral",
        "confidence": 0.72
    }
]

# Convert to dataset
dataset = create_dataset_from_dicts(
    data=data,
    input_model=SentimentInput,
    input_keys=["text", "context"],  # Fields for your input model
    metadata_keys=["label", "confidence"],  # Fields for metric evaluation
    case_id_key=None  # Will auto-generate: "item-0", "item-1", etc.
)

print(f"Created {len(dataset)} examples")
print(f"First input: {dataset[0].input}")
print(f"First metadata: {dataset[0].metadata}")

# Output:
# Created 3 examples
# First input: text='This product exceeded my expectations!' context='Product review'
# First metadata: {'label': 'positive', 'confidence': 0.95}
```

### From a JSON File

Load data from a JSON file containing an array of objects:

```python
from pydantic import BaseModel, Field
from gepadantic.data_utils import json_to_dataset

class QuestionInput(BaseModel):
    question: str = Field(description="User's question")
    domain: str = Field(description="Question domain")

# JSON file content (questions.json):
# [
#   {
#     "question": "What is Python?",
#     "domain": "programming",
#     "answer": "A high-level programming language",
#     "difficulty": "easy"
#   },
#   {
#     "question": "Explain recursion with an example",
#     "domain": "programming",
#     "answer": "A function that calls itself...",
#     "difficulty": "medium"
#   }
# ]

def dict_to_input(item: dict) -> QuestionInput:
    """Custom mapper for complex transformations"""
    return QuestionInput(
        question=item["question"],
        domain=item["domain"]
    )

dataset = json_to_dataset(
    json_path="questions.json",
    input_mapper=dict_to_input,
    metadata_keys=["answer", "difficulty"],
    case_id_key="question"  # Use question text as case ID
)

print(f"Loaded {len(dataset)} questions")
print(f"Case ID: {dataset[0].case_id}")
```

### From a Pandas DataFrame

Convert tabular data with a custom row mapper:

```python
import pandas as pd
from pydantic import BaseModel, Field
from gepadantic.data_utils import dataframe_to_dataset

class ClassificationInput(BaseModel):
    text: str = Field(description="Input text")
    category: str = Field(description="Text category")

# Create DataFrame
df = pd.DataFrame({
    "id": [1, 2, 3],
    "text": [
        "The stock market rose today",
        "New smartphone released",
        "Team wins championship"
    ],
    "category": ["finance", "technology", "sports"],
    "true_label": ["finance", "tech", "sports"],
    "source": ["news_api", "blog", "news_api"]
})

def row_to_input(row) -> ClassificationInput:
    """Convert DataFrame row to input model"""
    return ClassificationInput(
        text=row["text"],
        category=row["category"]
    )

dataset = dataframe_to_dataset(
    df=df,
    row_mapper=row_to_input,
    metadata_cols=["true_label", "source"],
    case_id_col="id"
)

print(f"Converted {len(dataset)} rows")
print(f"Sample metadata: {dataset[0].metadata}")
```

## Splitting Data for Training and Validation

### Manual Split

Split an existing dataset:

```python
from gepadantic.data_utils import split_dataset

# Assume we have a dataset with 100 examples
trainset, valset = split_dataset(
    dataset=dataset,
    train_ratio=0.8,  # 80% training, 20% validation
    shuffle=True,
    random_seed=42  # For reproducibility
)

print(f"Training: {len(trainset)}, Validation: {len(valset)}")
```

### Combined Creation and Split

Create and split in one step:

```python
from pydantic import BaseModel, Field
from gepadantic.data_utils import prepare_train_val_sets

class TaskInput(BaseModel):
    instruction: str = Field(description="Task instruction")
    input_text: str = Field(description="Input to process")

data = [
    {"instruction": "Summarize", "input_text": "Long article...", "label": "summary"},
    {"instruction": "Translate", "input_text": "Hello world", "label": "Hola mundo"},
    # ... more examples ...
]

trainset, valset = prepare_train_val_sets(
    data=data,
    input_model=TaskInput,
    input_keys=["instruction", "input_text"],
    metadata_keys=["label"],
    train_ratio=0.7,
    shuffle=True,
    random_seed=42
)

print(f"Training: {len(trainset)}, Validation: {len(valset)}")
```

## Complete Example: Data Preparation for Optimization

Here's a full workflow from raw data to optimization-ready datasets:

```python
from pydantic import BaseModel, Field
from gepadantic.data_utils import prepare_train_val_sets
from gepadantic.schema import DataInstWithInput, RolloutOutput

# 1. Define your input model
class SentimentInput(BaseModel):
    text: str = Field(description="Text to classify")

# 2. Define your output model
class SentimentOutput(BaseModel):
    sentiment: str = Field(description="Predicted sentiment: positive, negative, or neutral")
    confidence: float = Field(description="Confidence score 0-1")

# 3. Prepare your raw data
raw_data = [
    {"text": "I love this product!", "label": "positive"},
    {"text": "Terrible experience", "label": "negative"},
    {"text": "It's okay", "label": "neutral"},
    {"text": "Best purchase ever!", "label": "positive"},
    {"text": "Not worth the price", "label": "negative"},
    {"text": "Average quality", "label": "neutral"},
]

# 4. Create train/val splits
trainset, valset = prepare_train_val_sets(
    data=raw_data,
    input_model=SentimentInput,
    input_keys=["text"],
    metadata_keys=["label"],  # Ground truth for metric
    train_ratio=0.67,  # 4 training, 2 validation
    shuffle=True,
    random_seed=42
)

# 5. Define your metric function
def accuracy_metric(
    output: RolloutOutput[SentimentOutput],
    data_inst: DataInstWithInput[SentimentInput]
) -> float:
    """Calculate accuracy: 1.0 for correct, 0.0 for incorrect"""
    if not output.success or output.result is None:
        return 0.0
    
    ground_truth = data_inst.metadata["label"]
    prediction = output.result.sentiment
    
    return 1.0 if prediction == ground_truth else 0.0

# 6. Use with optimizer
# from gepadantic import GepaAgent
# 
# agent = GepaAgent(
#     result_type=SentimentOutput,
#     # ... other config ...
# )
# 
# optimized_agent = agent.optimize(
#     trainset=trainset,
#     valset=valset,
#     metric=accuracy_metric,
#     num_candidates=5,
#     max_iterations=3
# )
```

## Key Takeaways

1. **DataInstWithInput** contains your input model and metadata (ground truth)
2. **RolloutOutput** contains the agent's output after execution
3. **Metric functions** compare outputs against metadata to compute scores
4. **Helper functions** transform common formats (DataFrame, JSON, dicts) into `DataInstWithInput` instances
5. **Metadata is crucial** - it stores ground truth labels, expected outputs, or any data needed for evaluation

## See Also

- [Getting Started](../user-guides/getting-started.md)
- [Optimization Guide](../user-guides/optimization.md)
- [API Reference](../api/index.md)
