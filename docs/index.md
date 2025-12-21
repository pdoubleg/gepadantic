# GEPAdantic

Pydantic interface for GEPA prompt optimization

!!! note
    There is at least one other repo working on the same thing. See this [issue](https://github.com/pydantic/pydantic-ai/issues/3179) for more info. Unlike the project noted in the issue, which is a full re-write of GEPA, here we rely on the **canonical GEPA API** by simply providing a bridge between the two systems.


GEPAdantic brings the power of [GEPA (Genetic Pareto)](https://github.com/gepa-ai/gepa) to pydantic-ai agents. GEPA is an evolutionary optimization algorithm that automatically improves your prompts through iterative refinement, learning from execution traces and feedback.

> If you can define your input and output as pydantic models, have some data and a metric, you can use GEPAdantic to optimize your prompts.

### Key Features

- **🎯 Automatic Prompt Optimization**: Let GEPA **evolve** your agent instructions and field descriptions.
- **🔧 Structured Data**: Use Pydantic models for type-safe, validated inputs with `SignatureAgent`, and pydantic-ai for structured output.
- **📊 Data-Driven**: Learn from your training data to optimize prompts for your specific use case.
- **🔄 Iterative Refinement**: GEPA uses evolutionary strategies to continuously improve a `seed_candidate`. The seed candidate is either inferred automatically, user defined, or even the results from another optimization run.
- **📈 Tool Optimization**: Optimize tool (and structured output model) descriptions and parameter schemas alongside prompts.

## What is GEPA?

[**GEPA**](https://github.com/gepa-ai/gepa) is an evolutionary optimizer that evolves multiple text components of a complex system to optimize them towards a given metric. GEPA can leverage rich textual feedback obtained from the system's execution environment, evaluation, and the system's own execution traces to iteratively improve performance.

## How GEPAdantic Works with GEPA

The `PydanticAIGEPAAdapter` plugs into the canonical GEPA optimization API and is responsible for:

1. **Evaluating proposed candidates**: Given a candidate consisting of proposed text components, and a minibatch of inputs sampled from the train/val sets, evaluate and return execution scores, also capturing the system traces.
2. **Extract Traces for Reflection**: Given the execution traces obtained from executing a proposed candidate, and a named component being optimized, return the textual content from the traces relevant to the named component.

At each iteration, GEPA proposes new candidates using:

1. **Reflective mutation**: Mutates the current candidate using rich textual feedback
2. **Merge**: Combines 2 candidates on the Pareto frontier

GEPA tracks the Pareto frontier of performance across candidates, leveraging candidates that work well on subsets of inputs to improve overall system performance.

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
    agent_model="gpt-4o-mini", # <- student model
    agent_instructions="Classify sentiment accurately",
    reflection_model="gpt-5" # <- better teacher model
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

## Core Components

| Component | Description |
|-----------|-------------|
| [SignatureAgent](core-components/signature-agent.md) | Adds `input_type` to pydantic-ai agents, enabling structured inputs with field descriptions that GEPA can optimize. |
| [Adapter](core-components/adapter.md) | Bridges pydantic-ai agents with GEPA's optimization API, handling evaluation, trajectory capture, and reflection. |
| [Runner](core-components/runner.md) | Orchestrates the GEPA optimization pipeline through iterative prompt evolution. |
| [Scaffold](core-components/scaffold.md) | Config-based convenience wrapper for simplified setup and execution. |
| [Data Utils](core-components/data-utils.md) | Helpers to convert common data formats into GEPA-compatible datasets. |

## Example Results

Here's an example from optimizing the Palmer Penguins classification problem. GEPA improved the score from **0.68 → 0.94** by refining prompts:

<details markdown="1">
<summary><b>Before Optimization (Score: 0.68)</b></summary>

#### Agent Instructions

```
Classify the penguin species based on the given physical measurements and location. 
Consider bill dimensions, flipper length, body mass, sex, and island to distinguish 
between Adelie, Chinstrap, and Gentoo penguins.
```

#### Output Tool Description

```
Output prediction for penguin species classification.
```

#### Output Tool Parameters

**`species`**

```
Predicted penguin species
```

**`confidence`**

```
Confidence score between 0 and 1
```

**`reasoning`**

```
Detailed explanation of the prediction based on the features
```

#### Input Schema Instructions

```
Input features for Palmer Penguins species classification.
```

#### Input Schema Field Descriptions

**`bill_length_mm`**

```
Length of the penguin's bill (culmen) in millimeters
```

**`bill_depth_mm`**

```
Depth of the penguin's bill (culmen) in millimeters
```

**`flipper_length_mm`**

```
Length of the penguin's flipper in millimeters
```

**`body_mass_g`**

```
Body mass of the penguin in grams
```

**`sex`**

```
Sex of the penguin
```

**`island`**

```
Island where the penguin was observed
```

</details>

<details markdown="1">
<summary><b>After Optimization (Score: 0.94)</b></summary>

#### Agent Instructions

```
Task: classify a penguin from the Palmer Penguins dataset into one of three species: 
Adelie, Chinstrap, or Gentoo. Inputs: bill_length_mm (mm), bill_depth_mm (mm), 
flipper_length_mm (mm), body_mass_g (g), sex (Male or Female), island (Torgersen, 
Biscoe, Dream). Use measurement patterns and island as a strong but not absolute prior.

Guiding domain knowledge:
- Typical distinguishing patterns (relative): Gentoo are the largest: longest flippers 
  and highest body mass; Chinstrap often have longer bill_length but shallower 
  bill_depth than Adelie; Adelie have shorter bills, relatively deeper bills, smaller 
  body mass and shorter flippers.
- Island priors (strong): Torgersen → primarily Adelie; Dream → primarily Chinstrap; 
  Biscoe → primarily Gentoo. Treat island as a prior (adds weight) but override it 
  when multiple numeric features strongly disagree.
- Sex effect: males tend to be heavier and have slightly longer flippers; interpret 
  body_mass_g and flipper_length_mm with a small sex adjustment rather than as 
  absolute determinants.

Decision strategy (recommended and repeatable):
1) Validate input units and ranges (mm for lengths, g for mass).
2) Standardize/normalize features against expected dataset-scale values (centering 
   is optional in heuristic rules).
3) Compute a per-species score composed of:
   (a) normalized distance from feature prototypes (bill_length, bill_depth, 
       flipper_length, body_mass) with weights emphasizing flipper_length and 
       body_mass for Gentoo, and bill_depth and bill_length for Adelie/Chinstrap;
   (b) an island prior boost for the species associated with that island;
   (c) a small sex-based mass adjustment.
4) Select species with highest score.
5) Compute confidence from the relative score gap and absolute agreement with island 
   prior (scale to [0,1]).
6) Produce a clear reasoning output that cites which features contributed most, shows 
   the normalized comparisons or thresholds used, and notes any conflict between 
   island prior and measurements.

Output requirements: return JSON with keys: species (one of [Adelie, Chinstrap, 
Gentoo]), confidence (float 0.0–1.0), reasoning (text explaining the decision and 
the main feature contributions). If features are missing or outside reasonable ranges, 
state that explicitly and lower confidence.
```

#### Output Tool Description

```
Return the final prediction for the penguin species classification as JSON with three 
fields: species, confidence, and reasoning. species must be exactly one of: 'Adelie', 
'Chinstrap', 'Gentoo'. confidence must be a float between 0 and 1. reasoning must 
explain which features (bill length, bill depth, flipper length, body mass, sex, 
island) drove the prediction, reference any heuristics or thresholds used, and note 
conflicts (e.g., island prior vs measurements). If the prediction is uncertain, 
include what additional measurements or checks would increase confidence.
```

#### Output Tool Parameters

**`species`**

```
Predicted penguin species — string; allowed values: 'Adelie', 'Chinstrap', 'Gentoo'. 
Choose exactly one. Base the choice on a weighted combination of: (a) closeness of 
bill_length_mm and bill_depth_mm to species patterns, (b) flipper_length_mm and 
body_mass_g (heaviest weight for Gentoo), and (c) island prior (Torgersen→Adelie, 
Dream→Chinstrap, Biscoe→Gentoo) which should be treated as strong but not absolute.
```

**`confidence`**

```
Confidence score — float in [0,1]. Compute from: normalized score difference between 
top and second species (primary), agreement with island prior (additive boost if 
island matches selected species), and absence/presence of conflicting features 
(penalize conflicts). Examples: large clear margin → >0.8, moderate margin or 
island-feature conflict → 0.4–0.8, very ambiguous or missing/out-of-range features 
→ <0.4. Document how the score was derived in reasoning.
```

**`reasoning`**

```
Detailed textual explanation of the prediction. Must include: 1) the input feature 
values and units; 2) which features contributed most and whether they pulled toward 
this species (explicit comparisons, e.g., 'bill_depth (18.9 mm) is deeper than 
typical Chinstrap and closer to Adelie'); 3) island prior and whether it supported 
or conflicted with the measurement-based decision; 4) a short description of the 
scoring/heuristic used (weights and any normalization); 5) how the confidence score 
was computed from relative scores and conflicts; 6) if confidence is low, state what 
additional data or checks would help (e.g., verify measurement units, check 
age/juvenile status, photo to confirm facial markings). Keep reasoning concise but 
specific (3–6 short sentences or numbered points).
```

#### Input Schema Instructions

```
Inputs expected for Palmer Penguins species classification. Provide a single record 
with fields: bill_length_mm (float, mm), bill_depth_mm (float, mm), flipper_length_mm 
(float, mm), body_mass_g (float, g), sex (literal 'Male' or 'Female'), island 
(literal 'Torgersen', 'Biscoe', or 'Dream'). All numeric values must be positive; 
if values are missing or obviously out of range (e.g., bill_length_mm < 20 or > 70, 
flipper_length_mm < 130 or > 260, body_mass_g < 2000 or > 8000), the assistant 
should flag this and lower confidence.
```

#### Input Schema Field Descriptions

**`bill_length_mm`**

```
Bill (culmen) length in millimeters (float). Typical dataset-scale guidance: expect 
roughly 30–60 mm; values below ~30 or above ~60 are unusual and should reduce 
confidence. Bill_length is important for distinguishing Chinstrap (generally longer) 
from Adelie (generally shorter) and helps identify Gentoo when combined with shallow 
bill_depth and large mass.
```

**`bill_depth_mm`**

```
Bill (culmen) depth in millimeters (float). Typical guidance: expect roughly 13–21 mm; 
higher bill_depth tends to indicate Adelie, while Chinstrap typically has shallower 
bills than Adelie. Use bill_depth together with bill_length for pairwise 
differentiation (e.g., short+deep → Adelie; long+shallow → Chinstrap).
```

**`flipper_length_mm`**

```
Flipper length in millimeters (float). Typical guidance: expect roughly 170–240 mm. 
Gentoo routinely have the longest flippers (strong indicator for Gentoo when combined 
with large body mass); short flippers lean toward Adelie. Use flipper_length as a 
high-weight feature for Gentoo detection.
```

**`body_mass_g`**

```
Body mass in grams (float). Typical guidance: expect roughly 2500–7000 g. Gentoo tend 
to have the highest body_mass_g; Adelie are smallest on average; Chinstrap intermediate. 
Apply a small sex adjustment (males slightly heavier) rather than treating sex as a 
separate predictor.
```

**`sex`**

```
Sex of the penguin: literal 'Male' or 'Female'. Use sex mainly to adjust interpretation 
of body_mass_g and (to a lesser extent) flipper_length_mm: if 'Male', allow slightly 
higher expected mass/flipper values before favoring Gentoo. Do not use sex alone to 
determine species.
```

**`island`**

```
Island where the penguin was observed: one of the literal values 'Torgersen', 'Biscoe', 
or 'Dream'. Treat island as a strong prior: Torgersen strongly favors Adelie, Dream 
strongly favors Chinstrap, Biscoe strongly favors Gentoo. However, if multiple numeric 
features strongly contradict the island prior, allow the measurement-based score to 
override the prior and reduce reported confidence accordingly.
```

</details>

## Installation

```bash
uv add gepadantic
```

## API Reference

Browse the [API Reference](api/index.md) for detailed documentation of all classes and functions.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
