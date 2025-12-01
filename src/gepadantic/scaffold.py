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
from pydantic_ai.agent import AbstractAgent

from .lm import get_openai_model
from .runner import (
    AUTO_RUN_SETTINGS,
    GepaOptimizationResult,
    auto_budget,
    optimize_agent_prompts,
)
from .signature_agent import SignatureAgent
from .types import DataInstWithInput, RolloutOutput

# Type variables
InputModelT = TypeVar("InputModelT", bound=BaseModel)
OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


@dataclass(kw_only=True)
class GepaConfig:
    """Configuration for GEPA optimization setup.

    This class captures all parameters needed to run GEPA optimization,
    providing a clean interface for users to configure their optimization runs.

    Args:
        agent: The agent to optimize. Can be provided instead of agent_model and agent_instructions.
        agent_model: The model name or Model instance for the agent (e.g., "gpt-4.1-mini").
        agent_instructions: System instructions for the agent.
        input_type: Pydantic model class defining the structured input format.
        output_type: Pydantic model class defining the expected output format.
        trainset: List of DataInstWithInput instances for training.
        metric: Function that evaluates agent outputs, returning (score, feedback).
        valset: Optional list of DataInstWithInput instances for validation. If None, trainset is used.
        auto: Automatically set the budget based on the dataset size. Can be 'light', 'medium', or 'heavy'.
        max_full_evals: Maximum number of full evaluations (budget).
        max_metric_calls: Maximum number of metric evaluations (budget).
        optimize_tools: Whether to optimize tool descriptions (default: True).
        seed_candidate: Optional initial candidate prompts to start optimization from.
        reflection_model: Model to use for reflection/mutation (default: None, uses agent_model).
        reflection_minibatch_size: Number of examples to use for reflection in each proposal (default: 3).
        perfect_score: The perfect score value to achieve (default: 1).
        skip_perfect_score: Whether to skip updating if perfect score achieved on minibatch (default: True).
        module_selector: Which components to optimize - "all", "round_robin", etc. (default: "round_robin").
        candidate_selection_strategy: Strategy for selecting candidates - "pareto", "current_best", "epsilon_greedy", etc. (default: "pareto").
        use_merge: Whether to use the merge strategy for combining candidates (default: False).
        max_merge_invocations: Maximum number of merge invocations to perform (default: 5).
        merge_val_overlap_floor: Minimum number of validation examples to overlap between merge candidates (default: 5).
        stop_callbacks: Stopper conditions for stopping optimization (default: None).
        display_progress_bar: Whether to show progress bar during optimization (default: True).
        track_best_outputs: Whether to track best outputs for analysis (default: True).
        enable_cache: Whether to enable caching of metric results (default: True).
        cache_dir: Directory for cache storage (default: ".gepa_cache").
        cache_verbose: Whether to print cache statistics (default: False).
        output_dir: Directory to save optimization results (default: "optimization_results").
        save_result: Whether to automatically save results to JSON (default: True).
        logger: LoggerProtocol instance for tracking progress (default: None).
        use_mlflow: Whether to use MLflow for logging (default: False).
        mlflow_tracking_uri: Tracking URI for MLflow (default: None).
        mlflow_experiment_name: Experiment name for MLflow (default: None).

    Example:

    ```python
        >>> from pydantic import BaseModel, Field
        >>> from gepadantic.scaffold import GepaConfig, run_optimization_pipeline
        >>> from gepadantic.data_utils import split_dataset
        >>>
        >>> class MyInput(BaseModel):
        ...     text: str = Field(description="Input text")
        >>>
        >>> class MyOutput(BaseModel):
        ...     category: str = Field(description="Classification category")
        >>>
        >>> def my_metric(data_inst, output):
        ...     if output.success and output.result:
        ...         return 1.0 if output.result.category == data_inst.metadata["label"] else 0.0, None
        ...     return 0.0, "Failed to produce output"
        >>>
        >>> # Split dataset into train and validation
        >>> trainset, valset = split_dataset(my_dataset, train_ratio=0.7)
        >>>
        >>> config = GepaConfig(
        ...     agent_model="gpt-4.1-mini",
        ...     agent_instructions="Classify the input text",
        ...     input_type=MyInput,
        ...     output_type=MyOutput,
        ...     trainset=trainset,
        ...     valset=valset,
        ...     metric=my_metric,
        ...     auto="light",
        ... )
        >>> result = run_optimization_pipeline(config)
    ```
    """

    # Required configuration
    input_type: type[BaseModel]
    output_type: type[BaseModel]
    trainset: Sequence[DataInstWithInput[Any]]
    metric: Callable[
        [DataInstWithInput[Any], RolloutOutput[Any]], tuple[float, str | None]
    ]

    # Core agent configuration
    agent: AbstractAgent[Any, Any] | None = None
    agent_model: str | None = None
    agent_instructions: str | None = None

    # Dataset and evaluation
    valset: Sequence[DataInstWithInput[Any]] | None = None

    # Budget configuration (exactly one must be set)
    auto: Literal["light", "medium", "heavy"] | None = None
    max_full_evals: int | None = None
    max_metric_calls: int | None = None

    # Optimize tools (e.g., output model schema text components)
    optimize_tools: bool = True

    # Optimization parameters
    seed_candidate: dict[str, str] | None = None
    reflection_model: str | None = None
    reflection_minibatch_size: int = 3
    perfect_score: int = 1
    skip_perfect_score: bool = True

    # Component selection options
    module_selector: ReflectionComponentSelector | Literal["round_robin", "all"] = "all"
    candidate_selection_strategy: (
        CandidateSelector | Literal["pareto", "current_best", "epsilon_greedy"]
    ) = "pareto"

    # Merge options
    use_merge: bool = True
    max_merge_invocations: int = 5
    merge_val_overlap_floor: int = 5

    # Stopping options
    stop_callbacks: StopperProtocol | Sequence[StopperProtocol] | None = None

    # Runtime options
    display_progress_bar: bool = True
    track_best_outputs: bool = True

    # Caching options
    enable_cache: bool = True
    cache_dir: str = ".gepa_cache"
    cache_verbose: bool = False

    # Output/logging options
    output_dir: str | Path = "optimization_results"
    save_result: bool = True
    logger: LoggerProtocol | None = None

    # MLFlow options
    use_mlflow: bool = False
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str | None = None

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
    3. Creates the agent with specified configuration
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

    Example:

    ```python
        >>> from gepadantic.data_utils import split_dataset
        >>>
        >>> # Split your dataset
        >>> trainset, valset = split_dataset(my_dataset, train_ratio=0.7)
        >>>
        >>> config = GepaConfig(
        ...     agent_model="gpt-4.1-mini",
        ...     agent_instructions="Classify sentiment as positive, negative, or neutral",
        ...     input_type=SentimentInput,
        ...     output_type=SentimentOutput,
        ...     trainset=trainset,
        ...     valset=valset,
        ...     metric=sentiment_metric,
        ...     auto="light",
        ... )
        >>> result = run_optimization_pipeline(config)
        >>> print(f"Best score: {result.best_score:.4f}")
        >>> print(f"Improvement: {result.improvement_ratio():.2%}")
        >>>
        >>> # Apply best candidate to agent
        >>> with result.apply_best(agent):
        ...     output = agent.run_sync("This is great!")
    ```
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
    # Wrap with SignatureAgent for structured input support
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
