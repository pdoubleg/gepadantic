# SignatureAgent

The `SignatureAgent` is a wrapper around pydantic-ai agents that enables **structured inputs** with field descriptions that GEPA can optimize. It's inspired by DSPy's [Signature](https://dspy.ai/learn/programming/signatures/) concept but adapted for pydantic-ai's architecture.

## Core Concept

Instead of passing raw strings to your agent, `SignatureAgent` allows you to define a Pydantic model that represents your input structure. Each field in the model can have a description that explains what that field represents. GEPA can then optimize these descriptions to improve your agent's performance.

## Basic Usage

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from gepadantic import SignatureAgent

# Define your input model
class QueryInput(BaseModel):
    """Answer questions about geography."""
    question: str = Field(description="The geography question to answer")
    context: str = Field(description="Additional context if needed")

class Answer(BaseModel):
    answer: str
    confidence: str

# Create base agent
agent = Agent('openai:gpt-4o', output_type=Answer)

# Wrap with SignatureAgent
signature_agent = SignatureAgent(agent, input_type=QueryInput)

# Run with structured input
query = QueryInput(
    question="What's the capital of France?",
    context="Focus on current political capital"
)
result = await signature_agent.run_signature(query)
```

## How Input Models Work

### 1. Input Type Alongside Output Type

Just as pydantic-ai agents have an `output_type` to structure responses, `SignatureAgent` adds an `input_type` to structure requests:

```python
# Traditional pydantic-ai approach
agent = Agent(model, output_type=Answer)
result = await agent.run("What's the capital of France?")

# SignatureAgent approach
signature_agent = SignatureAgent(agent, input_type=QueryInput, output_type=Answer)
sig = QueryInput(question="What's the capital of France?")
result = await signature_agent.run_signature(sig)
```

### 2. Structured Inputs vs Raw Strings

When you use a `SignatureAgent`, your input is a Pydantic model instance instead of a raw string. This provides:

- **Type safety**: Fields are validated according to their type annotations
- **Clarity**: Each piece of information has a named field
- **Optimizability**: GEPA can learn better descriptions for each field

Example with multiple fields:

```python
class EmailInput(BaseModel):
    """Analyze customer support emails."""
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Email body text")
    sender_tier: str = Field(description="Customer tier (bronze, silver, gold)")
    previous_interactions: int = Field(description="Number of previous support tickets")

# All fields are provided as structured data
email = EmailInput(
    subject="Problem with billing",
    body="I was charged twice for my subscription...",
    sender_tier="gold",
    previous_interactions=3
)
```

## How `append_instructions` Works

By default (`append_instructions=True`), the `SignatureAgent` appends schema information to the agent's system instructions. This helps the model understand what each input field represents. DSPy follows a similar pattern in their `dspy.InputField` construct.

### Default Behavior

When you create a `SignatureAgent` with `append_instructions=True` (the default):

```python
signature_agent = SignatureAgent(
    agent,
    input_type=QueryInput,
    append_instructions=True  # This is the default
)
```

The agent's system instructions are augmented with information about the input schema.

### What Gets Appended

The system instructions are built from several parts:

1. **Model Docstring**: The docstring of your input model becomes the main task description
2. **Field Descriptions**: Each field's description is included with its type information
3. **XML-Formatted Schema**: The structure is presented in a clear, readable format

**Example - Input Model:**

```python
class TaskInput(BaseModel):
    """Classify support tickets by urgency and category."""
    ticket_text: str = Field(description="The full text of the support ticket")
    customer_tier: str = Field(description="Customer subscription level")
    response_time: str = Field(description="Expected response time requirement")
```

**Generated System Instructions:**

```
Classify support tickets by urgency and category.

Inputs
- `<ticket_text>` (str): The full text of the support ticket
- `<customer_tier>` (str): Customer subscription level
- `<response_time>` (str): Expected response time requirement
```

### Before/After Comparison

**Without `append_instructions`:**

```python
# Agent only sees its base instructions
agent = Agent(
    model='openai:gpt-4o',
    instructions="You are a helpful assistant"
)
# System prompt: "You are a helpful assistant"
```

**With `append_instructions` (default):**

```python
# Agent sees base instructions + signature schema
signature_agent = SignatureAgent(
    agent,
    input_type=TaskInput,
    append_instructions=True
)
# System prompt combines:
# 1. Base instructions: "You are a helpful assistant"
# 2. Signature instructions: "Classify support tickets by urgency and category."
# 3. Field schemas: "Inputs\n- `<ticket_text>` (str): The full text..."
```

### Example Showing Generated Instructions

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from gepadantic import SignatureAgent

class QuestionInput(BaseModel):
    """Answer questions using the provided context."""
    question: str = Field(description="The question to answer")
    context: str = Field(description="Relevant context for answering")
    constraints: str = Field(description="Any constraints on the answer format")

agent = Agent(
    'openai:gpt-4o',
    instructions="You are a precise question-answering system.",
    output_type=str
)

signature_agent = SignatureAgent(agent, input_type=QuestionInput)

# The actual system instructions sent to the LLM will be:
# """
# You are a precise question-answering system.
#
# Answer questions using the provided context.
#
# Inputs
# - `<question>` (str): The question to answer
# - `<context>` (str): Relevant context for answering
# - `<constraints>` (str): Any constraints on the answer format
# """
```

## GEPA Component Updates

During optimization, GEPA can modify the text components extracted from your signature. Here's how it works:

### Component Naming Convention

Each optimizable piece of text has a unique key:

- `"signature:{ModelName}:instructions"` - The model's docstring
- `"signature:{ModelName}:{field_name}:desc"` - Each field's description

**Example:**

```python
class EmailInput(BaseModel):
    """Analyze customer emails."""
    subject: str = Field(description="Email subject")
    body: str = Field(description="Email body")

# Extracted components:
{
    "signature:EmailInput:instructions": "Analyze customer emails.",
    "signature:EmailInput:subject:desc": "Email subject",
    "signature:EmailInput:body:desc": "Email body"
}
```

### How GEPA Evolves Components

GEPA uses reflection to analyze agent performance and propose better text:

1. **Initial Extraction**: Components are extracted from your original model
2. **Evaluation**: The agent runs on training data with current components
3. **Reflection**: GEPA analyzes failures and successes in the execution traces
4. **Proposal**: New component text is generated to address weaknesses
5. **Testing**: The new candidate is evaluated on validation data
6. **Selection**: Better candidates are kept; poor ones are discarded

**Example Evolution:**

```python
# Initial component
{
    "signature:EmailInput:subject:desc": "Email subject"
}

# After GEPA optimization (learned that urgency matters)
{
    "signature:EmailInput:subject:desc": "Email subject line, which may indicate urgency level"
}

# Further refinement (learned about priority keywords)
{
    "signature:EmailInput:subject:desc": "Email subject line. Look for urgency indicators like 'URGENT', 'ASAP', or 'Critical'"
}
```

### Component Extraction and Application

**Extraction** (getting initial components):

```python
from gepadantic.components import extract_seed_candidate_with_signature

# Extract all optimizable components
components = extract_seed_candidate_with_signature(
    agent=agent,
    input_type=QueryInput
)

# Result:
{
    "instructions": "Original agent instructions...",
    "signature:QueryInput:instructions": "Answer questions about geography.",
    "signature:QueryInput:question:desc": "The geography question to answer",
    "signature:QueryInput:context:desc": "Additional context if needed"
}
```

**Application** (using optimized components):

```python
# Apply optimized components during inference
optimized_components = {
    "signature:QueryInput:question:desc": "The specific geography question requiring a factual answer with confidence assessment"
}

with signature_agent.input_spec.apply_candidate(optimized_components):
    result = await signature_agent.run_signature(query)
```

## Practical Examples

### Example 1: Multi-Field Classification

```python
class SupportTicketInput(BaseModel):
    """Classify support tickets for routing and prioritization."""
    ticket_text: str = Field(description="Full support ticket text")
    customer_name: str = Field(description="Name of the customer")
    product: str = Field(description="Product the ticket is about")
    previous_tickets: int = Field(description="Number of previous tickets from this customer")

class TicketClassification(BaseModel):
    urgency: str = Field(description="low, medium, high, or critical")
    category: str = Field(description="billing, technical, feature_request, or other")
    assigned_team: str = Field(description="Team to handle this ticket")

agent = Agent('openai:gpt-4o', output_type=TicketClassification)
signature_agent = SignatureAgent(agent, input_type=SupportTicketInput)

# Use it
ticket = SupportTicketInput(
    ticket_text="My payment failed but I was still charged!",
    customer_name="Alice Smith",
    product="Premium Subscription",
    previous_tickets=2
)

result = await signature_agent.run_signature(ticket)
print(result.data)  # TicketClassification(urgency='high', category='billing', ...)
```

### Example 2: Seeing the Actual Prompts

You can inspect what prompt is actually sent to the LLM:

```python
# The user content (XML-formatted)
user_content = signature_agent._prepare_user_content(ticket)
# Result:
# <ticket_text>My payment failed but I was still charged!</ticket_text>
# <customer_name>Alice Smith</customer_name>
# <product>Premium Subscription</product>
# <previous_tickets>2</previous_tickets>

# The system instructions
system_instructions = signature_agent._prepare_system_instructions(ticket)
# Result includes:
# "Classify support tickets for routing and prioritization.
#
# Inputs
# - `<ticket_text>` (str): Full support ticket text
# - `<customer_name>` (str): Name of the customer
# ..."
```

### Example 3: Tool Optimization

When using tools, you can also optimize tool descriptions:

```python
from pydantic_ai import RunContext

async def search_database(ctx: RunContext, query: str) -> str:
    """Search the customer database for relevant information."""
    # Implementation here
    pass

agent = Agent('openai:gpt-4o', tools=[search_database])

# Enable tool optimization
signature_agent = SignatureAgent(
    agent,
    input_type=QueryInput,
    optimize_tools=True  # This is the key
)

# Now GEPA can optimize:
# - "tool:search_database:description" - The tool's docstring
# - "tool:search_database:param:query" - The parameter description
```

## Key Points

1. **Type Safety**: Input models are validated Pydantic models with full type checking
2. **Automatic Schema Generation**: Field descriptions are automatically formatted into clear instructions
3. **Optimization Target**: Each field description becomes a component GEPA can improve
4. **Flexible**: Can be combined with regular pydantic-ai features like tools, dependencies, and streaming
5. **Backward Compatible**: The wrapped agent still works normally; `SignatureAgent` just adds capabilities

## API Reference

For complete API details, see:

::: gepadantic.signature_agent.SignatureAgent
    options:
      show_root_heading: true
      show_source: false
      