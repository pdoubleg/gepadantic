"""Complete example demonstrating GEPA scaffolding with the Facility Support Analyzer dataset.

This example shows how to use the scaffolding system to optimize a facility support
email classification task using the Facility Support Analyzer dataset released by Meta.

The task involves classifying facility support emails into:
- Urgency (low, medium, high)
- Sentiment (positive, neutral, negative)
- Categories (multi-label from a controlled taxonomy)

This example is based on the DSPy tutorial and demonstrates how GEPA can optimize
structured information extraction for enterprise tasks.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Literal

import numpy as np
import requests
from pydantic import BaseModel, Field
from pydantic_ai import Agent

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gepadantic.lm import get_openai_model
from src.gepadantic.scaffold import GepaConfig, run_optimization_pipeline
from src.gepadantic.signature_agent import SignatureAgent
from src.gepadantic.types import DataInstWithInput, InputModelT, RolloutOutput


# Step 1: Define input and output models
class EmailInput(BaseModel):
    """Input model for email classification."""

    input: str = Field(description="The email content to classify")


class FacilitySupportAnalyzer(BaseModel):
    """Analyzes facility support requests."""

    urgency: Literal["low", "medium", "high"] = Field(
        description="The urgency of the request"
    )
    sentiment: Literal["positive", "neutral", "negative"] = Field(
        description="The sentiment of the request"
    )
    categories: list[
        Literal[
            "emergency_repair_services",
            "routine_maintenance_requests",
            "quality_and_safety_concerns",
            "specialized_cleaning_services",
            "general_inquiries",
            "sustainability_and_environmental_practices",
            "training_and_support_requests",
            "cleaning_services_scheduling",
            "customer_feedback_and_complaints",
            "facility_management_issues",
        ]
    ] = Field(description="The categories of the request")


# Step 2: Load and prepare the dataset
def load_facility_support_data() -> list[DataInstWithInput[InputModelT]]:
    """Load data from Meta's Facility Support Analyzer dataset.

    Returns:
        List of DataInstWithInput instances with email content and expected outputs
    """
    url = "https://raw.githubusercontent.com/meta-llama/llama-prompt-ops/refs/heads/main/use-cases/facility-support-analyzer/dataset.json"

    print("📥 Downloading Facility Support Analyzer dataset from Meta...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        dataset = json.loads(response.text)
        print(f"✅ Successfully loaded {len(dataset)} examples")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        raise

    prepared_dataset: list[DataInstWithInput[InputModelT]] = []

    for idx, item in enumerate(dataset):
        if not isinstance(item, dict):
            raise ValueError(f"Item at index {idx} is not a dict: {type(item)}")

        # Convert dict to input model
        try:
            input_instance = EmailInput(input=item["fields"]["input"])
        except Exception as e:
            raise ValueError(f"Error mapping item {idx} to input model: {e}") from e

        # Extract metadata with expected output
        metadata = {
            "expected_output": json.loads(item["answer"]),
        }

        case_id = f"item-{idx}"

        # Create DataInstWithInput
        data_inst = DataInstWithInput[InputModelT](
            input=input_instance,
            message_history=None,
            metadata=metadata,
            case_id=case_id,
        )
        prepared_dataset.append(data_inst)

    return prepared_dataset


# Step 3: Define scoring functions for each component
def score_urgency(gold_urgency: str, pred_urgency: str) -> float:
    """Compute score for the urgency module.

    Args:
        gold_urgency: Ground truth urgency level
        pred_urgency: Predicted urgency level

    Returns:
        Score of 1.0 if correct, 0.0 otherwise
    """
    return 1.0 if gold_urgency == pred_urgency else 0.0


def score_sentiment(gold_sentiment: str, pred_sentiment: str) -> float:
    """Compute score for the sentiment module.

    Args:
        gold_sentiment: Ground truth sentiment
        pred_sentiment: Predicted sentiment

    Returns:
        Score of 1.0 if correct, 0.0 otherwise
    """
    return 1.0 if gold_sentiment == pred_sentiment else 0.0


def score_categories(
    gold_categories: dict[str, bool], pred_categories: list[str]
) -> float:
    """Compute score for the categories module.

    Uses match/mismatch logic for category accuracy.

    Args:
        gold_categories: Ground truth categories as dict with boolean values
        pred_categories: Predicted categories as list of strings

    Returns:
        Score between 0.0 and 1.0 representing category accuracy
    """
    correct = 0
    for k, v in gold_categories.items():
        if v and k in pred_categories:
            correct += 1
        elif not v and k not in pred_categories:
            correct += 1
    score = correct / len(gold_categories)
    return score


# Step 4: Define feedback functions for GEPA
def feedback_urgency(gold_urgency: str, pred_urgency: str) -> tuple[str, float]:
    """Generate feedback for the urgency module.

    Args:
        gold_urgency: Ground truth urgency level
        pred_urgency: Predicted urgency level

    Returns:
        Tuple of (feedback_string, score)
    """
    score = 1.0 if gold_urgency == pred_urgency else 0.0
    if gold_urgency == pred_urgency:
        feedback = f"You correctly classified the urgency of the message as `{gold_urgency}`. This message is indeed of `{gold_urgency}` urgency."
    else:
        feedback = f"You incorrectly classified the urgency of the message as `{pred_urgency}`. The correct urgency is `{gold_urgency}`. Think about how you could have reasoned to get the correct urgency label."
    return feedback, score


def feedback_sentiment(gold_sentiment: str, pred_sentiment: str) -> tuple[str, float]:
    """Generate feedback for the sentiment module.

    Args:
        gold_sentiment: Ground truth sentiment
        pred_sentiment: Predicted sentiment

    Returns:
        Tuple of (feedback_string, score)
    """
    score = 1.0 if gold_sentiment == pred_sentiment else 0.0
    if gold_sentiment == pred_sentiment:
        feedback = f"You correctly classified the sentiment of the message as `{gold_sentiment}`. This message is indeed `{gold_sentiment}`."
    else:
        feedback = f"You incorrectly classified the sentiment of the message as `{pred_sentiment}`. The correct sentiment is `{gold_sentiment}`. Think about how you could have reasoned to get the correct sentiment label."
    return feedback, score


def feedback_categories(
    gold_categories: dict[str, bool], pred_categories: list[str]
) -> tuple[str, float]:
    """Generate feedback for the categories module.

    Args:
        gold_categories: Ground truth categories as dict with boolean values
        pred_categories: Predicted categories as list of strings

    Returns:
        Tuple of (feedback_string, score)
    """
    correctly_included = [
        k for k, v in gold_categories.items() if v and k in pred_categories
    ]
    incorrectly_included = [
        k for k, v in gold_categories.items() if not v and k in pred_categories
    ]
    incorrectly_excluded = [
        k for k, v in gold_categories.items() if v and k not in pred_categories
    ]
    correctly_excluded = [
        k for k, v in gold_categories.items() if not v and k not in pred_categories
    ]

    # Recompute category accuracy
    score = (len(correctly_included) + len(correctly_excluded)) / len(gold_categories)

    if score == 1.0:
        fb_text = f"The category classification is perfect. You correctly identified that the message falls under the following categories: `{repr(correctly_included)}`."
    else:
        fb_text = f"The category classification is not perfect. You correctly identified that the message falls under the following categories: `{repr(correctly_included)}`.\n"
        if incorrectly_included:
            fb_text += f"However, you incorrectly identified that the message falls under the following categories: `{repr(incorrectly_included)}`. The message DOES NOT fall under these categories.\n"
        if incorrectly_excluded:
            prefix = "Additionally, " if incorrectly_included else "However, "
            fb_text += f"{prefix}you didn't identify the following categories that the message actually falls under: `{repr(incorrectly_excluded)}`.\n"
        fb_text += "Think about how you could have reasoned to get the correct category labels."
    return fb_text, score


# Step 5: Define the main metric with feedback
def metric_with_feedback(
    data_inst: DataInstWithInput[EmailInput],
    output: RolloutOutput[FacilitySupportAnalyzer],
) -> tuple[float, str | None]:
    """Evaluate facility support request analysis with feedback.

    This metric checks if the predicted urgency, sentiment, and categories match
    the ground truth. It also provides feedback on the reasoning process for each module.

    Args:
        data_inst: Input data instance with metadata containing ground truth
        output: Agent's output to evaluate

    Returns:
        Tuple of (score, feedback) where score is between 0.0 and 1.0
    """
    if not output.success or output.result is None:
        return 0.0, output.error_message or "Agent failed to produce output"

    # Extract predicted values
    pred_urgency = output.result.urgency
    pred_sentiment = output.result.sentiment
    pred_categories = output.result.categories

    # Extract ground truth from metadata
    gold_urgency = data_inst.metadata["expected_output"]["urgency"]
    gold_sentiment = data_inst.metadata["expected_output"]["sentiment"]
    gold_categories = data_inst.metadata["expected_output"]["categories"]

    # Get feedback and scores for each component
    fb_urgency, score_urgency = feedback_urgency(gold_urgency, pred_urgency)
    fb_sentiment, score_sentiment = feedback_sentiment(gold_sentiment, pred_sentiment)
    fb_categories, score_categories = feedback_categories(
        gold_categories, pred_categories
    )

    # Calculate final score as average of all module scores
    final_score = (score_urgency + score_sentiment + score_categories) / 3.0

    # Combine feedback into a single string
    feedback = (
        f"Urgency: {fb_urgency}\nSentiment: {fb_sentiment}\nCategories: {fb_categories}"
    )

    return final_score, feedback


# Step 6: Evaluation helper for test set
def evaluate_agent_on_testset(
    agent: SignatureAgent,
    test_set: list[DataInstWithInput[EmailInput]],
    description: str = "Agent",
) -> dict[str, Any]:
    """Evaluate an agent on a test set and return detailed metrics.

    Args:
        agent: The SignatureAgent to evaluate
        test_set: List of test data instances
        description: Description of the agent being evaluated (for display)

    Returns:
        Dictionary containing scores, predictions, and per-component accuracy
    """
    print(f"\n📊 Evaluating {description} on test set ({len(test_set)} examples)...")

    scores = []
    urgency_correct = 0
    sentiment_correct = 0
    category_scores = []

    predictions = []

    for data_inst in test_set:
        try:
            result = agent.run_signature_sync(data_inst.input)

            # Extract predictions
            pred_urgency = result.output.urgency
            pred_sentiment = result.output.sentiment
            pred_categories = result.output.categories

            # Extract ground truth
            gold_urgency = data_inst.metadata["expected_output"]["urgency"]
            gold_sentiment = data_inst.metadata["expected_output"]["sentiment"]
            gold_categories = data_inst.metadata["expected_output"]["categories"]

            # Calculate component scores
            urgency_score = score_urgency(gold_urgency, pred_urgency)
            sentiment_score = score_sentiment(gold_sentiment, pred_sentiment)
            categories_score = score_categories(gold_categories, pred_categories)

            # Track correct predictions
            if urgency_score == 1.0:
                urgency_correct += 1
            if sentiment_score == 1.0:
                sentiment_correct += 1
            category_scores.append(categories_score)

            # Overall score
            overall_score = (urgency_score + sentiment_score + categories_score) / 3.0
            scores.append(overall_score)

            predictions.append(
                {
                    "pred_urgency": pred_urgency,
                    "gold_urgency": gold_urgency,
                    "pred_sentiment": pred_sentiment,
                    "gold_sentiment": gold_sentiment,
                    "pred_categories": pred_categories,
                    "gold_categories": gold_categories,
                    "score": overall_score,
                }
            )

        except Exception as e:
            print(f"❌ Error evaluating example: {e}")
            scores.append(0.0)
            predictions.append(
                {
                    "error": str(e),
                    "score": 0.0,
                }
            )

    # Calculate metrics
    mean_score = np.mean(scores) if scores else 0.0
    urgency_accuracy = urgency_correct / len(test_set) if test_set else 0.0
    sentiment_accuracy = sentiment_correct / len(test_set) if test_set else 0.0
    category_accuracy = np.mean(category_scores) if category_scores else 0.0

    return {
        "mean_score": mean_score,
        "urgency_accuracy": urgency_accuracy,
        "sentiment_accuracy": sentiment_accuracy,
        "category_accuracy": category_accuracy,
        "predictions": predictions,
        "scores": scores,
    }


# Step 7: Main optimization pipeline
def main():
    """Run the GEPA optimization for Facility Support Analyzer."""
    
    SAMPLE_SIZE = 30

    print("\n" + "=" * 70)
    print("DATASET PREPARATION")
    print("=" * 70)

    # Load the full dataset
    full_dataset = load_facility_support_data()
    full_dataset = full_dataset[:SAMPLE_SIZE]
    
    print("\n📊 Dataset Statistics:")
    print(f"   └─ Total Examples: {len(full_dataset)}")

    # Shuffle and split the dataset
    import random

    random.Random(42).shuffle(full_dataset)

    # Split: 33% train, 33% val, 34% test
    train_size = int(len(full_dataset) * 0.33)
    val_size = int(len(full_dataset) * 0.33)

    train_set = full_dataset[:train_size]
    val_set = full_dataset[train_size : train_size + val_size]
    test_set = full_dataset[train_size + val_size :]

    print("\n🔀 Dataset Split:")
    print(f"   └─ Training Set: {len(train_set)} ({len(train_set)/len(full_dataset):.1%})")
    print(f"   └─ Validation Set: {len(val_set)} ({len(val_set)/len(full_dataset):.1%})")
    print(f"   └─ Test Set: {len(test_set)} ({len(test_set)/len(full_dataset):.1%})")

    # Show sample email
    print("\n📧 Sample Email (Training Set):")
    print("-" * 70)
    sample_email = train_set[0].input.input
    print(sample_email[:300] + "..." if len(sample_email) > 300 else sample_email)
    print("-" * 70)
    print(f"Expected Output: {train_set[0].metadata['expected_output']}")
    print("=" * 70)

    # Configure the optimization
    reflection_model = "gpt-4.1"
    agent_model = "gpt-4.1-nano"

    DEFAULT_INSTRUCTIONS = """
Read the following facility support request and analyze it to determine the urgency, sentiment, and categories.
"""

    config = GepaConfig(
        # Agent configuration
        agent_model=agent_model,
        agent_instructions=DEFAULT_INSTRUCTIONS,
        input_type=EmailInput,
        output_type=FacilitySupportAnalyzer,
        # Data and evaluation
        trainset=train_set,
        valset=val_set,
        metric=metric_with_feedback,
        # Budget
        max_full_evals=3,
        # Optimization parameters
        module_selector="all",
        candidate_selection_strategy="pareto",
        optimize_tools=True,
        use_merge=False,
        # LLM for reflection
        reflection_model=reflection_model,
        # Display options
        display_progress_bar=True,
        track_best_outputs=True,
        # Caching for faster iterations
        enable_cache=False,
        cache_dir=".gepa_cache",
        # Output settings
        output_dir="optimization_results",
        save_result=True,
    )

    print("\n" + "=" * 70)
    print("OPTIMIZATION CONFIGURATION")
    print("=" * 70)
    print("\n🎯 Task: Facility Support Email Classification")
    print("\n🤖 Models:")
    print(f"   └─ Agent Model: {config.agent_model}")
    print(f"   └─ Reflection Model: {config.reflection_model}")
    print("\n📚 Dataset Configuration:")
    print(f"   └─ Training Set: {len(config.trainset)} emails")
    print(
        f"   └─ Validation Set: {len(config.valset) if config.valset else 0} emails"
    )
    print(f"   └─ Test Set: {len(test_set)} emails (for final evaluation)")
    print("\n⚙️  Optimization Settings:")
    print(f"   └─ Max Full Evaluations: {config.max_full_evals}")
    print(f"   └─ Estimated Metric Calls: {config.estimated_metric_calls}")
    print(f"   └─ Module Selector: {config.module_selector}")
    print(f"   └─ Candidate Selection: {config.candidate_selection_strategy}")
    print(f"   └─ Optimize Tools: {config.optimize_tools}")
    print(f"   └─ Use Merge: {config.use_merge}")
    print(f"   └─ Cache Enabled: {config.enable_cache}")
    print("=" * 70 + "\n")

    # Evaluate baseline agent on test set
    print("\n" + "=" * 70)
    print("PRE-OPTIMIZATION BASELINE EVALUATION")
    print("=" * 70)

    baseline_agent_obj = Agent(
        model=get_openai_model(config.agent_model),
        instructions=config.agent_instructions,
        output_type=FacilitySupportAnalyzer,
        retries=5,
    )

    baseline_agent = SignatureAgent(
        baseline_agent_obj,
        input_type=EmailInput,
    )

    baseline_results = evaluate_agent_on_testset(
        baseline_agent, test_set, description="Baseline Agent"
    )

    print("\n📊 Baseline Test Set Performance:")
    print("-" * 70)
    print(f"{'Metric':<30} {'Score':<15}")
    print("-" * 70)
    print(f"{'Overall Score':<30} {baseline_results['mean_score']:<15.3f}")
    print(f"{'Urgency Accuracy':<30} {baseline_results['urgency_accuracy']:<15.3f}")
    print(f"{'Sentiment Accuracy':<30} {baseline_results['sentiment_accuracy']:<15.3f}")
    print(f"{'Category Accuracy':<30} {baseline_results['category_accuracy']:<15.3f}")
    print("-" * 70)

    # Show a few sample predictions
    print("\n📋 Sample Baseline Predictions (First 3):")
    for i, pred in enumerate(baseline_results["predictions"][:3], 1):
        if "error" in pred:
            print(f"\n❌ Example {i}: Error - {pred['error']}")
            continue

        print(f"\n--- Example {i} (Score: {pred['score']:.2f}) ---")
        print(f"Urgency: {pred['pred_urgency']} (actual: {pred['gold_urgency']})")
        print(f"Sentiment: {pred['pred_sentiment']} (actual: {pred['gold_sentiment']})")
        print(f"Categories: {pred['pred_categories'][:3]}...")

    print("=" * 70 + "\n")

    # Run the optimization
    print("=" * 70)
    print("RUNNING GEPA OPTIMIZATION")
    print("=" * 70 + "\n")

    result = run_optimization_pipeline(config)

    # Display optimization results
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)

    print("\n📈 Performance Metrics:")
    print(f"   └─ Best Validation Score: {result.best_score:.4f}")

    if result.original_score is not None:
        print(f"   └─ Original Validation Score: {result.original_score:.4f}")
        improvement = result.improvement_ratio()
        if improvement is not None:
            improvement_symbol = (
                "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
            )
            print(f"   └─ {improvement_symbol} Improvement: {improvement:+.2%}")

    print("\n🔄 Optimization Statistics:")
    print(f"   └─ Iterations Completed: {result.num_iterations}")
    print(f"   └─ Metric Evaluations: {result.num_metric_calls}")

    print("\n💰 GEPA Token Usage:")
    print(f"   └─ Input Tokens: {result.gepa_usage.input_tokens:,}")
    print(f"   └─ Output Tokens: {result.gepa_usage.output_tokens:,}")
    print(
        f"   └─ Total Tokens: {result.gepa_usage.input_tokens + result.gepa_usage.output_tokens:,}"
    )
    print(f"   └─ API Calls: {result.gepa_usage.requests}")

    print("\n🔧 Optimized Components:")
    for component_name, component_value in result.best_candidate.items():
        print(f"\n   {component_name}:")
        # Show first 200 chars of each component
        value_str = str(component_value)
        if len(value_str) > 200:
            print(f"      {value_str[:200]}...")
        else:
            for line in value_str.split("\n")[:5]:  # Show first 5 lines
                print(f"      {line}")

    print("\n" + "=" * 70)

    # Check if optimization made improvements
    improvement = result.improvement_ratio()
    has_improvement = improvement is not None and improvement > 0

    if not has_improvement:
        print("\n" + "=" * 70)
        print("⚠️  NO IMPROVEMENT DETECTED")
        print("=" * 70)
        print("\nThe optimization did not find a better configuration than the baseline.")
        print("Skipping post-optimization evaluation since the model has not changed.")
        print("\nℹ️  The baseline test results above represent the final performance.")

        optimized_results = baseline_results

    else:
        # Evaluate optimized agent on test set
        print("\n" + "=" * 70)
        print("POST-OPTIMIZATION TEST EVALUATION")
        print("=" * 70)

        # Create optimized agent
        optimized_agent_obj = Agent(
            model=get_openai_model(config.agent_model),
            instructions=config.agent_instructions,
            output_type=FacilitySupportAnalyzer,
            retries=5,
        )

        # Apply optimized configuration and evaluate
        with result.apply_best_to(agent=optimized_agent_obj, input_type=EmailInput):
            optimized_agent = SignatureAgent(
                optimized_agent_obj,
                input_type=EmailInput,
                optimize_tools=True,
            )

            optimized_results = evaluate_agent_on_testset(
                optimized_agent, test_set, description="Optimized Agent"
            )

        print("\n📊 Optimized Test Set Performance:")
        print("-" * 70)
        print(f"{'Metric':<30} {'Score':<15}")
        print("-" * 70)
        print(f"{'Overall Score':<30} {optimized_results['mean_score']:<15.3f}")
        print(f"{'Urgency Accuracy':<30} {optimized_results['urgency_accuracy']:<15.3f}")
        print(
            f"{'Sentiment Accuracy':<30} {optimized_results['sentiment_accuracy']:<15.3f}"
        )
        print(
            f"{'Category Accuracy':<30} {optimized_results['category_accuracy']:<15.3f}"
        )
        print("-" * 70)

        # Show a few sample predictions
        print("\n📋 Sample Optimized Predictions (First 3):")
        for i, pred in enumerate(optimized_results["predictions"][:3], 1):
            if "error" in pred:
                print(f"\n❌ Example {i}: Error - {pred['error']}")
                continue

            print(f"\n--- Example {i} (Score: {pred['score']:.2f}) ---")
            print(f"Urgency: {pred['pred_urgency']} (actual: {pred['gold_urgency']})")
            print(
                f"Sentiment: {pred['pred_sentiment']} (actual: {pred['gold_sentiment']})"
            )
            print(f"Categories: {pred['pred_categories'][:3]}...")

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print("\n📊 Dataset Configuration:")
    print(f"   └─ Total Examples: {len(full_dataset)}")
    print(
        f"   └─ Training: {len(train_set)} | Validation: {len(val_set)} | Test: {len(test_set)}"
    )

    if has_improvement:
        print("\n📈 Performance Comparison (Test Set):")
        print(f"   {'Metric':<30} {'Baseline':<15} {'Optimized':<15} {'Improvement':<15}")
        print(f"   {'-'*75}")
        print(
            f"   {'Overall Score':<30} {baseline_results['mean_score']:<15.3f} {optimized_results['mean_score']:<15.3f} {optimized_results['mean_score'] - baseline_results['mean_score']:+.3f}"
        )
        print(
            f"   {'Urgency Accuracy':<30} {baseline_results['urgency_accuracy']:<15.3f} {optimized_results['urgency_accuracy']:<15.3f} {optimized_results['urgency_accuracy'] - baseline_results['urgency_accuracy']:+.3f}"
        )
        print(
            f"   {'Sentiment Accuracy':<30} {baseline_results['sentiment_accuracy']:<15.3f} {optimized_results['sentiment_accuracy']:<15.3f} {optimized_results['sentiment_accuracy'] - baseline_results['sentiment_accuracy']:+.3f}"
        )
        print(
            f"   {'Category Accuracy':<30} {baseline_results['category_accuracy']:<15.3f} {optimized_results['category_accuracy']:<15.3f} {optimized_results['category_accuracy'] - baseline_results['category_accuracy']:+.3f}"
        )
    else:
        print("\n📈 Performance (Test Set - No Improvement):")
        print(f"   {'Metric':<30} {'Value':<15}")
        print(f"   {'-'*45}")
        print(f"   {'Overall Score':<30} {baseline_results['mean_score']:<15.3f}")
        print(f"   {'Urgency Accuracy':<30} {baseline_results['urgency_accuracy']:<15.3f}")
        print(
            f"   {'Sentiment Accuracy':<30} {baseline_results['sentiment_accuracy']:<15.3f}"
        )
        print(
            f"   {'Category Accuracy':<30} {baseline_results['category_accuracy']:<15.3f}"
        )

    print("\n💰 Optimization Cost:")
    print(
        f"   └─ Total Tokens: {result.gepa_usage.input_tokens + result.gepa_usage.output_tokens:,}"
    )
    print(f"   └─ API Calls: {result.gepa_usage.requests}")
    print(f"   └─ Iterations: {result.num_iterations}")

    print("\n" + "=" * 70)
    print("✅ EXAMPLE COMPLETE!")
    print("=" * 70 + "\n")

    return result


if __name__ == "__main__":
    # Run the example
    result = main()

