"""Configuration-based scaffolding for GEPA optimization setup.

This module provides a simplified interface for setting up GEPA prompt optimization.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypeVar

from gepa.logging.logger import LoggerProtocol, StdOutLogger
from gepa.proposer.reflective_mutation.base import (
    CandidateSelector,
    ReflectionComponentSelector,
)
from gepa.utils import StopperProtocol
from pydantic import BaseModel
from pydantic_ai import Agent

from .lm import get_openai_model
from .adapter import ReflectionSampler
from .runner import (
    AUTO_RUN_SETTINGS,
    GepaOptimizationResult,
    auto_budget,
    optimize_agent_prompts,
)
from .signature_agent import SignatureAgent
from .schema import DataInstWithInput, RolloutOutput

# Type variables
InputModelT = TypeVar("InputModelT", bound=BaseModel)
OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


@dataclass(kw_only=True)
class GepaConfig:
    """Configuration for GEPA optimization setup.

    This class captures all parameters needed to run GEPA optimization,
    providing a single point of entry for optimization runs.

    GEPA is an evolutionary optimizer that evolves (multiple) text components of a 
    complex system to optimize them towards a given metric. GEPA can also leverage 
    rich textual feedback obtained from the system's execution environment, evaluation,
    and the system's own execution traces to iteratively improve the system's performance.

    ## Concepts

    - **System**: A harness that uses text components to perform a task. Each text 
      component of the system to be optimized is a named component of the system.
    - **Candidate**: A mapping from component names to component text. A concrete 
      instantiation of the system is realized by setting the text of each system 
      component to the text provided by the candidate mapping.
    - **DataInst**: An (uninterpreted) data type over which the system operates.
    - **RolloutOutput**: The output of the system on a DataInst.

    Each execution of the system produces a RolloutOutput, which can be evaluated to 
    produce a score. The execution of the system also produces a trajectory, which 
    consists of the operations performed by different components of the system, 
    including the text of the components that were executed.

    ## Adapter Implementation

    Here we use a custom PydanticAIGEPAAdapter that plugs into the canonical GEPA 
    optimization API. The adapter is responsible for:

    1. **Evaluating a proposed candidate on a batch of inputs**:
        * The adapter receives a candidate proposed by GEPA, along with a batch of 
            inputs selected from the training/validation set.
        * The adapter instantiates the system with the texts proposed in the candidate.
        * The adapter then evaluates the candidate on the batch of inputs, and returns 
            the scores.
        * The adapter should also capture relevant information from the execution of 
            the candidate, like system and evaluation traces.

    2. **Identifying textual information relevant to a component of the candidate**:
        * Given the trajectories captured during the execution of the candidate, GEPA 
            selects a component of the candidate to update.
        * The adapter receives the candidate, the batch of inputs, and the trajectories 
            captured during the execution of the candidate.
        * The adapter is responsible for identifying the textual information relevant 
            to the component to update.
        * This information is used by GEPA to reflect on the performance of the 
            component, and propose new component texts.

    ## Optimization Strategies

    At each iteration, GEPA proposes a new candidate using one of the following strategies:

    1. **Reflective mutation**: GEPA proposes a new candidate by mutating the current 
       candidate, leveraging rich textual feedback.
    2. **Merge**: GEPA proposes a new candidate by merging 2 candidates that are on 
       the Pareto frontier.

    GEPA also tracks the Pareto frontier of performance achieved by different candidates 
    on the validation set. This way, it can leverage candidates that work well on a 
    subset of inputs to improve the system's performance on the entire validation set, 
    by evolving from the Pareto frontier.

    Example:
        ```python
        from pydantic import BaseModel, Field
        from gepadantic import GepaConfig, run_optimization_pipeline, split_dataset

        class MyInput(BaseModel):
                text: str = Field(description="Input text")
                
        class MyOutput(BaseModel):
            category: str = Field(description="Classification category")
            
        def my_metric(data_inst, output):
            if output.success and output.result:
                return 1.0 if output.result.category == data_inst.metadata["label"] else 0.0, None
            return 0.0, "Failed to produce output"
            
        # Split dataset into train and validation
        trainset, valset = split_dataset(dataset, train_ratio=0.7)
        
        config = GepaConfig(
            agent_model="gpt-4.1-nano",
            reflection_model="gpt-4.1",
            agent_instructions="Classify the input text",
            input_type=MyInput,
            output_type=MyOutput,
            trainset=trainset,
            valset=valset,
            metric=my_metric,
            auto="light",
        )
        result = run_optimization_pipeline(config)
        ```
    """

    # Required configuration
    input_type: type[BaseModel]
    """Pydantic model class defining the structured input format."""

    output_type: type[BaseModel]
    """Pydantic model class defining the expected output format."""

    trainset: Sequence[DataInstWithInput[Any]]
    """List of DataInstWithInput instances for training."""

    metric: Callable[
        [DataInstWithInput[Any], RolloutOutput[Any]], tuple[float, str | None]
    ]
    """Function that evaluates agent outputs, returning (score, feedback)."""

    # Core agent configuration
    agent: SignatureAgent[Any, Any] | None = None
    """The pre-configured SignatureAgent to optimize. Can be provided instead of agent_model. Useful for complex agents, e.g. with tools, mcp servers, etc."""

    agent_model: str | None = None
    """The model name or Model instance for the agent (e.g., "gpt-4.1-mini")."""

    agent_instructions: str | None = None
    """System instructions for the agent."""

    # Dataset and evaluation
    valset: Sequence[DataInstWithInput[Any]] | None = None
    """Optional list of DataInstWithInput instances for validation. If None, trainset is used."""

    # Budget configuration (exactly one must be set)
    auto: Literal["light", "medium", "heavy"] | None = None
    """Automatically set the budget based on the dataset size. Can be 'light', 'medium', or 'heavy'."""

    max_full_evals: int | None = None
    """Maximum number of full evaluations (budget)."""

    max_metric_calls: int | None = None
    """Maximum number of metric evaluations (budget)."""

    # Optimize tools (e.g., output model schema text components)
    optimize_tools: bool = True
    """Whether to optimize tool descriptions (default: True)."""

    # Optimization parameters
    seed_candidate: dict[str, str] | None = None
    """Optional initial candidate prompts to start optimization from."""

    reflection_model: str | None = None
    """Model to use for reflection/mutation (default: None, uses agent_model)."""

    reflection_minibatch_size: int = 3
    """Number of examples to use for reflection in each proposal (default: 3)."""

    perfect_score: int = 1
    """The perfect score value to achieve (default: 1)."""

    skip_perfect_score: bool = True
    """Whether to skip updating if perfect score achieved on minibatch (default: True)."""

    # Sampling and selection options
    reflection_sampler: ReflectionSampler | None = None
    """Optional custom sampler for reflection records."""

    module_selector: ReflectionComponentSelector | Literal["round_robin", "all"] = "all"
    """Component selection strategy. Can be a ReflectionComponentSelector instance or a string ('round_robin', 'all'). Defaults to 'all'. \
    The 'round_robin' strategy cycles through components in order. The 'all' strategy selects all components for modification in every GEPA iteration."""

    candidate_selection_strategy: (
        CandidateSelector | Literal["pareto", "current_best", "epsilon_greedy"]
    ) = "pareto"
    """Strategy for selecting candidates - "pareto", "current_best", "epsilon_greedy". (default: "pareto")."""

    # Merge options
    use_merge: bool = True
    """Whether to use the merge strategy for combining candidates (default: False)."""

    max_merge_invocations: int = 5
    """Maximum number of merge invocations to perform (default: 5)."""

    merge_val_overlap_floor: int = 5
    """Minimum number of shared validation ids required between parents before attempting a merge subsample. \
    Only relevant when using `val_evaluation_policy` other than `full_eval`."""

    # Stopping options
    stop_callbacks: StopperProtocol | Sequence[StopperProtocol] | None = None
    """Stopper conditions for stopping optimization (default: None). If None, optimization will run until the budget, e.g. max_metric_calls, is exhausted."""

    # Runtime options
    display_progress_bar: bool = True
    """Whether to show progress bar during optimization (default: True)."""

    track_best_outputs: bool = True
    """Whether to track best outputs for analysis (default: True)."""

    # Caching options
    enable_cache: bool = True
    """Whether to enable caching of metric results (default: True)."""

    cache_dir: str = ".gepa_cache"
    """Directory for cache storage (default: "~/.gepa_cache")."""

    cache_verbose: bool = False
    """Whether to print cache statistics (default: False)."""

    # Output/logging options
    output_dir: str | Path = "optimization_results"
    """Directory to save optimization results (default: "optimization_results")."""

    save_result: bool = True
    """Whether to automatically save results to JSON (default: True)."""

    logger: LoggerProtocol | None = None
    """LoggerProtocol instance for tracking progress (default: None)."""

    # MLFlow options
    use_mlflow: bool = False
    """Whether to use MLflow for logging (default: False)."""

    mlflow_tracking_uri: str | None = None
    """Tracking URI for MLflow (default: None)."""

    mlflow_experiment_name: str | None = None
    """Experiment name for MLflow (default: None)."""

    # Reproducibility
    seed: int = 0
    """Random seed for reproducibility (default: 0)."""

    raise_on_exception: bool = (True,)
    """Whether to raise exceptions or continue on errors (default: True)."""

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if len(self.trainset) == 0:
            raise ValueError("trainset cannot be empty")

        # Validate that exactly one budget option is set
        assert (self.max_metric_calls is not None) + (
            self.max_full_evals is not None
        ) + (self.auto is not None) == 1, (
            "Exactly one of max_metric_calls, max_full_evals, auto must be set. "
            f"You set max_metric_calls={self.max_metric_calls}, "
            f"max_full_evals={self.max_full_evals}, "
            f"auto={self.auto}."
        )
        # Validate that exactly one of agent or agent_model is set
        assert (self.agent is not None) + (self.agent_model is not None) == 1, (
            "Exactly one of agent or agent_model must be set. "
            f"You set agent={self.agent}, "
            f"agent_model={self.agent_model}."
        )
        if self.logger is None:
            self.logger = StdOutLogger()

    @property
    def estimated_metric_calls(self) -> int:
        if self.max_metric_calls is not None:
            metric_calls = self.max_metric_calls
        elif self.auto is not None:
            metric_calls = auto_budget(
                num_candidates=AUTO_RUN_SETTINGS[self.auto]["n"],
                valset_size=len(self.valset)
                if self.valset is not None
                else len(self.trainset),
            )
        elif self.max_full_evals is not None:
            metric_calls = self.max_full_evals * (
                len(self.trainset)
                + (len(self.valset) if self.valset is not None else 0)
            )
        else:
            raise ValueError("No budget set")
        self.logger.log(
            f"GEPA needs approx {metric_calls} metric calls of the program. This amounts to {metric_calls / len(self.trainset) if self.valset is None else metric_calls / (len(self.trainset) + len(self.valset)):.2f} full evals on the {'train' if self.valset is None else 'train+val'} set."
        )
        return metric_calls


def run_optimization_pipeline(config: GepaConfig) -> GepaOptimizationResult:
    """Set up and run GEPA optimization based on configuration.

    This function orchestrates the complete optimization workflow:
    1. Validates the configuration
    2. Uses provided train and validation sets
    3. Creates the agent with specified configuration, or uses provided agent
    4. Wraps agent with SignatureAgent for structured input support
    5. Runs GEPA optimization
    6. Optionally saves results to disk

    Args:
        config: GepaConfig instance with all optimization parameters.

    Returns:
        GepaOptimizationResult containing the best candidate, scores, and metadata.

    Raises:
        ValueError: If configuration is invalid.
        RuntimeError: If optimization fails.
    """

    # Use provided trainset and valset
    trainset = config.trainset
    valset = config.valset

    # Print dataset info
    if valset is not None:
        print(f"Dataset: {len(trainset)} training, {len(valset)} validation examples")
    else:
        print(
            f"Dataset: {len(trainset)} training examples (valset=None, will use trainset for validation)"
        )

    # Create the base agent
    if config.agent is None:
        model = get_openai_model(config.agent_model)
        agent = Agent(
            model=model,
            instructions=config.agent_instructions,
            output_type=config.output_type,
        )
    else:
        agent = config.agent

    # Wrap with SignatureAgent for structured input support (if not already wrapped)
    if not isinstance(agent, SignatureAgent):
        signature_agent = SignatureAgent(
            agent,
            input_type=config.input_type,
            optimize_tools=config.optimize_tools,
        )

    # Run optimization
    print("Starting GEPA optimization...")
    result = optimize_agent_prompts(
        agent=signature_agent,
        seed_candidate=config.seed_candidate,
        trainset=trainset,
        valset=valset,
        module_selector=config.module_selector,
        metric=config.metric,
        input_type=config.input_type,
        max_metric_calls=config.max_metric_calls,
        max_full_evals=config.max_full_evals,
        auto=config.auto,
        reflection_model=config.reflection_model,
        reflection_minibatch_size=config.reflection_minibatch_size,
        perfect_score=config.perfect_score,
        skip_perfect_score=config.skip_perfect_score,
        candidate_selection_strategy=config.candidate_selection_strategy,
        use_merge=config.use_merge,
        max_merge_invocations=config.max_merge_invocations,
        merge_val_overlap_floor=config.merge_val_overlap_floor,
        stop_callbacks=config.stop_callbacks,
        display_progress_bar=config.display_progress_bar,
        track_best_outputs=config.track_best_outputs,
        enable_cache=config.enable_cache,
        cache_dir=config.cache_dir,
        cache_verbose=config.cache_verbose,
        logger=config.logger,
        use_mlflow=config.use_mlflow,
        mlflow_tracking_uri=config.mlflow_tracking_uri,
        mlflow_experiment_name=config.mlflow_experiment_name,
        seed=config.seed,
        raise_on_exception=config.raise_on_exception,
        reflection_sampler=config.reflection_sampler,
    )

    # Save result if requested
    if config.save_result:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"optimization_{timestamp}.json"

        result_dict = result.model_dump()
        with open(output_file, "w") as f:
            json.dump(result_dict, f, indent=2)

        print(f"\n✅ Optimization result saved to: {output_file}")

    # Print summary
    print(f"   Best score: {result.best_score:.4f}")
    print(f"   Iterations: {result.num_iterations}")
    print(f"   Metric calls: {result.num_metric_calls}")

    improvement = result.improvement_ratio()
    if improvement is not None:
        print(f"   Improvement: {improvement:.2%}")

    return result
