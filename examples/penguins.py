"""Complete example demonstrating GEPA scaffolding with the Palmer Penguins dataset.

This example shows how to use the scaffolding system to optimize a species
classification task using the Palmer Penguins dataset with penguin measurements.

The example includes metrics:
- Accuracy
- Precision, Recall, and F1-Score (per class and macro-averaged)
- Confusion Matrix
- Performance comparison between baseline and optimized agents
"""

from __future__ import annotations

import os
import sys
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gepadantic.data_utils import dataframe_to_dataset, split_dataset
from src.gepadantic.lm import get_openai_model
from src.gepadantic.scaffold import GepaConfig, run_optimization_pipeline
from src.gepadantic.types import DataInstWithInput, RolloutOutput
from src.gepadantic.signature_agent import SignatureAgent


# Step 1: Define input and output models
class PenguinInput(BaseModel):
    """Input features for Palmer Penguins species classification."""

    bill_length_mm: float = Field(
        description="Length of the penguin's bill (culmen) in millimeters"
    )
    bill_depth_mm: float = Field(
        description="Depth of the penguin's bill (culmen) in millimeters"
    )
    flipper_length_mm: float = Field(
        description="Length of the penguin's flipper in millimeters"
    )
    body_mass_g: float = Field(
        description="Body mass of the penguin in grams"
    )
    sex: Literal["Male", "Female"] = Field(description="Sex of the penguin")
    island: Literal["Torgersen", "Biscoe", "Dream"] = Field(
        description="Island where the penguin was observed"
    )


class SpeciesPrediction(BaseModel):
    """Output prediction for penguin species classification."""

    species: Literal["Adelie", "Chinstrap", "Gentoo"] = Field(
        description="Predicted penguin species"
    )
    confidence: float = Field(
        description="Confidence score between 0 and 1", ge=0.0, le=1.0
    )
    reasoning: str = Field(
        description="Detailed explanation of the prediction based on the features"
    )


# Step 2: Load and prepare the Palmer Penguins dataset
def load_penguins_data(
    n_train: int = 30, n_holdout: int = 15
) -> tuple[
    list[DataInstWithInput[PenguinInput]], list[DataInstWithInput[PenguinInput]]
]:
    """Load and prepare Palmer Penguins dataset for GEPA with holdout test set.

    This function uses stratified sampling to ensure balanced class distributions
    in both training and holdout sets. It leverages the existing data_utils helpers
    to convert the DataFrame directly into DataInstWithInput format.

    Args:
        n_train: Number of samples to use for training/validation (default 30)
        n_holdout: Number of samples to hold out for final testing (default 15)

    Returns:
        Tuple of (training_dataset, holdout_dataset) as DataInstWithInput lists
        with balanced species class distributions
    """
    # Load Palmer Penguins dataset from seaborn
    try:
        import seaborn as sns

        df = sns.load_dataset("penguins")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure seaborn is installed: pip install seaborn")
        raise

    # Select features and clean data
    df = df[
        [
            "bill_length_mm",
            "bill_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
            "sex",
            "island",
            "species",
        ]
    ].copy()

    # Drop rows with missing values
    df = df.dropna()

    # Store species as label for metadata
    df["label"] = df["species"]

    # Perform stratified sampling to maintain class balance
    # This ensures both train and holdout sets have representative distributions
    # First, limit to the total samples we need
    total_needed = n_train + n_holdout
    if len(df) > total_needed:
        # Sample with stratification to get the subset we need
        df_subset, _ = train_test_split(
            df,
            train_size=total_needed,
            stratify=df["species"],
            random_state=42,
        )
    else:
        df_subset = df

    # Now split into train and holdout with stratification
    df_train, df_holdout = train_test_split(
        df_subset,
        train_size=n_train,
        test_size=min(n_holdout, len(df_subset) - n_train),
        stratify=df_subset["species"],
        random_state=42,
    )

    # Define a mapper function to convert DataFrame rows to PenguinInput
    def row_to_penguin_input(row: pd.Series) -> PenguinInput:
        """Convert a DataFrame row to a PenguinInput instance.

        Args:
            row: pandas Series representing a single penguin observation

        Returns:
            PenguinInput instance with the penguin's features
        """
        return PenguinInput(
            bill_length_mm=float(row["bill_length_mm"]),
            bill_depth_mm=float(row["bill_depth_mm"]),
            flipper_length_mm=float(row["flipper_length_mm"]),
            body_mass_g=float(row["body_mass_g"]),
            sex=str(row["sex"]),
            island=str(row["island"]),
        )

    # Convert DataFrames to datasets using our helper function
    train_dataset = dataframe_to_dataset(
        df_train,
        row_mapper=row_to_penguin_input,
        metadata_cols=["label"],  # Only include the species label in metadata
    )

    holdout_dataset = dataframe_to_dataset(
        df_holdout,
        row_mapper=row_to_penguin_input,
        metadata_cols=["label"],
    )

    return train_dataset, holdout_dataset


# Step 3: Helper function for classification metrics
def calculate_classification_metrics(
    predictions: list[str], actuals: list[str], label_name: str = "species"
) -> dict[str, float]:
    """Calculate precision, recall, F1, and confusion matrix for multi-class classification.
    
    Args:
        predictions: List of predicted labels (species names)
        actuals: List of actual labels (species names)
        label_name: Name of the label being predicted (for display)
    
    Returns:
        Dictionary containing accuracy, precision, recall, F1, and confusion matrix
    """
    # Get unique classes
    classes = sorted(list(set(actuals + predictions)))
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    # Convert to numeric labels
    y_pred = [class_to_idx[p] for p in predictions]
    y_true = [class_to_idx[a] for a in actuals]
    
    # Calculate metrics
    accuracy = sum(p == t for p, t in zip(y_pred, y_true)) / len(y_pred) if y_pred else 0.0
    
    # Calculate precision, recall, F1 for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=list(range(len(classes)))
    )
    
    # Calculate macro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    
    result = {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "confusion_matrix": cm,
        "classes": classes,
    }
    
    # Add per-class metrics
    for idx, cls in enumerate(classes):
        result[f"precision_{cls}"] = precision[idx]
        result[f"recall_{cls}"] = recall[idx]
        result[f"f1_{cls}"] = f1[idx]
        result[f"support_{cls}"] = support[idx]
    
    return result


def print_classification_report(
    metrics: dict[str, float], title: str = "Classification Metrics"
) -> None:
    """Print a detailed classification report.
    
    Args:
        metrics: Dictionary of metrics from calculate_classification_metrics
        title: Title for the report section
    """
    print(f"\n{title}:")
    print("-" * 50)
    print(f"{'Metric':<20} {'Value':<15}")
    print("-" * 50)
    print(f"{'Accuracy':<20} {metrics['accuracy']:<15.3f}")
    print(f"{'Precision (Macro)':<20} {metrics['precision_macro']:<15.3f}")
    print(f"{'Recall (Macro)':<20} {metrics['recall_macro']:<15.3f}")
    print(f"{'F1-Score (Macro)':<20} {metrics['f1_macro']:<15.3f}")
    print("-" * 50)
    
    # Print per-class metrics
    classes = metrics.get("classes", [])
    if classes:
        print("\nPer-Class Metrics:")
        print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 60)
        for cls in classes:
            precision = metrics.get(f"precision_{cls}", 0.0)
            recall = metrics.get(f"recall_{cls}", 0.0)
            f1 = metrics.get(f"f1_{cls}", 0.0)
            support = metrics.get(f"support_{cls}", 0)
            print(f"{cls:<15} {precision:<12.3f} {recall:<12.3f} {f1:<12.3f} {support:<10.0f}")
    
    # Print confusion matrix
    cm = metrics.get("confusion_matrix")
    if cm is not None and classes:
        print("\nConfusion Matrix:")
        print("                 Predicted")
        # Print header
        header = "Actual           " + "  ".join([f"{cls[:8]:>8}" for cls in classes])
        print(header)
        # Print rows
        for i, cls in enumerate(classes):
            row = f"{cls[:15]:<15}  " + "  ".join([f"{cm[i][j]:>8}" for j in range(len(classes))])
            print(row)
    print("-" * 50)


# Step 4: Define evaluation metric
def species_metric(
    data_inst: DataInstWithInput[PenguinInput],
    output: RolloutOutput[SpeciesPrediction],
) -> tuple[float, str | None]:
    """Evaluate species prediction accuracy.

    This metric checks if the predicted species matches the ground truth.
    It also considers confidence calibration as a bonus.

    Args:
        data_inst: Input data instance with metadata containing ground truth.
        output: Agent's output to evaluate.

    Returns:
        Tuple of (score, feedback) where score is between 0.0 and 1.0.
    """
    # Check if the agent execution was successful
    if not output.success or output.result is None:
        return 0.0, output.error_message or "Agent failed to produce output"

    # Extract predicted species
    predicted_species = output.result.species
    confidence = output.result.confidence
    reasoning = output.result.reasoning

    # Extract ground truth from metadata
    ground_truth = data_inst.metadata.get("label")

    if ground_truth is None:
        return 0.0, "No ground truth label found in metadata"
    
    # Check if prediction is correct
    is_correct = predicted_species == ground_truth

    if is_correct:
        # Correct prediction: base score + confidence bonus
        score = 0.7 + (0.3 * confidence)
        feedback = f"✅ Correct: predicted '{predicted_species}', actual '{ground_truth}' (confidence: {confidence:.2f}).\n"
    else:
        # Incorrect prediction: small score if low confidence
        score = 0.3 * (1 - confidence)
        feedback = f"❌ Incorrect: predicted '{predicted_species}', but actual was '{ground_truth}' (confidence: {confidence:.2f}).\n"
        feedback += f"Your reasoning: {reasoning}\n"
        feedback += "Reflect on what aspects of the penguin's measurements might have led to this error. "
        feedback += f"Consider the distinguishing features between {predicted_species} and {ground_truth} penguins."

    return score, feedback


# Step 5: Main optimization pipeline
def main():
    """Run the GEPA optimization for Palmer Penguins species classification."""

    print("\n" + "=" * 70)
    print("DATASET PREPARATION")
    print("=" * 70)
    
    # First, load the raw dataset to show original distribution before sampling
    # This helps users understand the baseline class distribution and sample size
    print("\n📊 Loading Raw Palmer Penguins Dataset...")
    try:
        import seaborn as sns
        df_raw = sns.load_dataset("penguins")
        
        # Clean the data the same way as in load_penguins_data
        df_clean = df_raw[
            [
                "bill_length_mm",
                "bill_depth_mm",
                "flipper_length_mm",
                "body_mass_g",
                "sex",
                "island",
                "species",
            ]
        ].copy()
        df_clean = df_clean.dropna()
        
        # Show raw dataset statistics
        total_raw = len(df_clean)
        raw_species_counts = df_clean["species"].value_counts().sort_index()
        
        print("\n📈 Raw Dataset (after removing missing values):")
        print(f"   └─ Total Penguins: {total_raw}")
        for species, count in raw_species_counts.items():
            print(f"   └─ {species}: {count} ({count/total_raw:.1%})")
        
    except Exception as e:
        print(f"   ⚠️  Could not load raw dataset: {e}")
    
    print("\n" + "-" * 70)

    # Load the data with train/holdout split - now returns datasets directly
    n_train = 15
    n_holdout = 15
    train_dataset, holdout_dataset = load_penguins_data(n_train=n_train, n_holdout=n_holdout)

    # Calculate total dataset size
    total_samples = len(train_dataset) + len(holdout_dataset)
    
    print(f"\n📊 Sampled Subset for This Run: {total_samples} penguins")
    print(f"   └─ Training Pool: {len(train_dataset)} ({len(train_dataset)/total_samples:.1%})")
    print(f"   └─ Holdout Test: {len(holdout_dataset)} ({len(holdout_dataset)/total_samples:.1%})")
    print("   └─ Sampling Strategy: Stratified (maintains class balance)")

    # Show species statistics for training pool
    train_species_counts = {}
    for data_inst in train_dataset:
        label = data_inst.metadata.get("label")
        train_species_counts[label] = train_species_counts.get(label, 0) + 1

    # Show species statistics for holdout
    holdout_species_counts = {}
    for data_inst in holdout_dataset:
        label = data_inst.metadata.get("label")
        holdout_species_counts[label] = holdout_species_counts.get(label, 0) + 1

    print("\n📈 Training Pool Class Distribution:")
    for label in sorted(train_species_counts.keys()):
        count = train_species_counts[label]
        print(f"   └─ {label}: {count} ({count/len(train_dataset):.1%})")
    
    print("\n📈 Holdout Test Class Distribution:")
    for label in sorted(holdout_species_counts.keys()):
        count = holdout_species_counts[label]
        print(f"   └─ {label}: {count} ({count/len(holdout_dataset):.1%})")

    # Split the training dataset into train/val using our helper
    train_ratio = 0.60
    trainset, valset = split_dataset(
        train_dataset, train_ratio=train_ratio, shuffle=True, random_seed=1
    )

    print(f"\n🔀 Training Pool Split (ratio={train_ratio}):")
    print(f"   └─ Training Set: {len(trainset)} ({len(trainset)/len(train_dataset):.1%})")
    print(f"   └─ Validation Set: {len(valset)} ({len(valset)/len(train_dataset):.1%})")
    
    # Calculate split class distributions
    train_split_counts = {}
    for data_inst in trainset:
        label = data_inst.metadata.get("label")
        train_split_counts[label] = train_split_counts.get(label, 0) + 1
    
    val_split_counts = {}
    for data_inst in valset:
        label = data_inst.metadata.get("label")
        val_split_counts[label] = val_split_counts.get(label, 0) + 1
    
    print("\n   Training Set Distribution:")
    for label in sorted(train_split_counts.keys()):
        count = train_split_counts[label]
        print(f"      └─ {label}: {count} ({count/len(trainset):.1%})")
    
    print("\n   Validation Set Distribution:")
    for label in sorted(val_split_counts.keys()):
        count = val_split_counts[label]
        print(f"      └─ {label}: {count} ({count/len(valset):.1%})")
    
    print("=" * 70)
    

    # Configure the optimization
    reflection_model = "gpt-5-mini"
    agent_model = "gpt-4.1-nano"

    config = GepaConfig(
        # Agent configuration
        agent_model=agent_model,
        agent_instructions=(
            "Classify the penguin species based on the given physical measurements and location. "
            "Consider bill dimensions, flipper length, body mass, sex, and island to distinguish "
            "between Adelie, Chinstrap, and Gentoo penguins."
        ),
        input_type=PenguinInput,
        output_type=SpeciesPrediction,
        # Data and evaluation
        trainset=trainset,
        valset=valset,
        metric=species_metric,
        # Budget
        max_full_evals=3,
        # Optimization parameters
        module_selector="all",
        candidate_selection_strategy="pareto",
        optimize_tools=True,
        use_merge=True,
        # LLM for reflection
        reflection_model=reflection_model,
        reflection_minibatch_size=3,
        # Display options
        display_progress_bar=True,
        track_best_outputs=True,
        # Caching for faster iterations
        enable_cache=True,
        cache_dir=".gepa_cache",
        cache_verbose=True,
        # Output settings
        output_dir="optimization_results",
        save_result=True,
    )

    print("\n" + "=" * 70)
    print("OPTIMIZATION CONFIGURATION")
    print("=" * 70)
    print("\n🎯 Task: Palmer Penguins Species Classification")
    print("\n🤖 Models:")
    print(f"   └─ Agent Model: {config.agent_model}")
    print(f"   └─ Reflection Model: {config.reflection_model}")
    print("\n📚 Dataset Configuration:")
    print(f"   └─ Training Set: {len(config.trainset)} penguins")
    print(f"   └─ Validation Set: {len(config.valset) if config.valset else 0} penguins")
    print(f"   └─ Holdout Test: {len(holdout_dataset)} penguins (for final evaluation)")
    print("\n⚙️  Optimization Settings:")
    print(f"   └─ Max Full Evaluations: {config.max_full_evals}")
    print(f"   └─ Max Metric Calls: {config.estimated_metric_calls}")
    print(f"   └─ Module Selector: {config.module_selector}")
    print(f"   └─ Candidate Selection: {config.candidate_selection_strategy}")
    print(f"   └─ Optimize Tools: {config.optimize_tools}")
    print(f"   └─ Use Merge: {config.use_merge}")
    print(f"   └─ Cache Enabled: {config.enable_cache}")
    print("=" * 70 + "\n")

    # Run eval on holdout set using baseline agent
    print("\n" + "=" * 70)
    print("PRE-OPTIMIZATION BASELINE EVALUATION")
    print("=" * 70)
    print(f"\nEvaluating baseline agent on holdout set ({len(holdout_dataset)} penguins)...")
    baseline_correct_predictions = 0
    baseline_total_predictions = 0
    baseline_results_table = []

    # Create and configure agent
    baseline_agent = Agent(
        model=get_openai_model(config.agent_model),
        instructions=config.agent_instructions,
        output_type=SpeciesPrediction,
    )

    baseline_signature_agent = SignatureAgent(
        baseline_agent,
        input_type=PenguinInput,
    )

    for i, data_inst in enumerate(holdout_dataset, 1):
        # Input is already a PenguinInput instance
        test_input = data_inst.input

        # Get ground truth from metadata
        actual = data_inst.metadata.get("label")
        test_result = baseline_signature_agent.run_signature_sync(test_input)
        predicted = test_result.output.species
        confidence = test_result.output.confidence
        reasoning = test_result.output.reasoning

        # Check if correct
        is_correct = predicted == actual
        if is_correct:
            baseline_correct_predictions += 1
        baseline_total_predictions += 1

        # Store result
        baseline_results_table.append(
            {
                "case": i,
                "bill_length": test_input.bill_length_mm,
                "bill_depth": test_input.bill_depth_mm,
                "flipper_length": test_input.flipper_length_mm,
                "body_mass": test_input.body_mass_g,
                "sex": test_input.sex,
                "island": test_input.island,
                "predicted": predicted,
                "actual": actual,
                "confidence": confidence,
                "correct": is_correct,
                "reasoning": reasoning,
            }
        )

    # Calculate baseline classification metrics
    baseline_predictions = [row["predicted"] for row in baseline_results_table]
    baseline_actuals = [row["actual"] for row in baseline_results_table]
    baseline_metrics = calculate_classification_metrics(baseline_predictions, baseline_actuals)
    
    # Print baseline results
    print("\n📋 Baseline Prediction Details:")
    print("-" * 110)
    print(
        f"{'#':<4} {'Island':<10} {'Sex':<7} {'Bill L':<8} {'Bill D':<8} {'Predicted':<12} {'Actual':<12} {'Conf':<6} {'Result':<8}"
    )
    print("-" * 110)
    for row in baseline_results_table:
        result_symbol = "✓" if row["correct"] else "✗"
        print(
            f"{row['case']:<4} {row['island']:<10} {row['sex']:<7} {row['bill_length']:<8.1f} {row['bill_depth']:<8.1f} "
            f"{row['predicted']:<12} {row['actual']:<12} {row['confidence']:<6.2f} {result_symbol:<8}"
        )
    print("-" * 110)
    baseline_accuracy = baseline_correct_predictions / baseline_total_predictions
    print(
        f"📊 BASELINE ACCURACY: {baseline_accuracy:.2%} ({baseline_correct_predictions}/{baseline_total_predictions})"
    )
    print("-" * 110)
    
    # Print detailed classification metrics
    print_classification_report(baseline_metrics, "📊 Baseline Classification Report")
    print("=" * 70 + "\n")

    # Run the optimization
    print("=" * 70)
    print("RUNNING GEPA OPTIMIZATION")
    print("=" * 70 + "\n")
    
    result = run_optimization_pipeline(config)

    # Display results
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)
    
    print("\n📈 Performance Metrics:")
    print(f"   └─ Best Validation Score: {result.best_score:.4f}")
    
    if result.original_score is not None:
        print(f"   └─ Original Validation Score: {result.original_score:.4f}")
        improvement = result.improvement_ratio()
        if improvement is not None:
            improvement_symbol = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
            print(f"   └─ {improvement_symbol} Improvement: {improvement:+.2%}")

    print("\n🔄 Optimization Statistics:")
    print(f"   └─ Iterations Completed: {result.num_iterations}")
    print(f"   └─ Metric Evaluations: {result.num_metric_calls}")
    
    print("\n💰 GEPA Token Usage:")
    print(f"   └─ Input Tokens: {result.gepa_usage.input_tokens:,}")
    print(f"   └─ Output Tokens: {result.gepa_usage.output_tokens:,}")
    print(f"   └─ Total Tokens: {result.gepa_usage.input_tokens + result.gepa_usage.output_tokens:,}")
    print(f"   └─ API Calls: {result.gepa_usage.requests}")

    print("\n🔧 Optimized Components:")
    for component_name, component_value in result.best_candidate.items():
        print(f"\n   {component_name}:")
        # Indent the component value
        for line in str(component_value).split('\n'):
            print(f"      {line}")

    print("\n" + "=" * 70)

    # Check if optimization made any improvements
    improvement = result.improvement_ratio()
    has_improvement = improvement is not None and improvement > 0
    
    if not has_improvement:
        print("\n" + "=" * 70)
        print("⚠️  NO IMPROVEMENT DETECTED")
        print("=" * 70)
        print("\nThe optimization did not find a better configuration than the baseline.")
        print("Skipping holdout evaluation since the model has not changed.")
        print("\nℹ️  The baseline holdout results above represent the final performance.")
        
        # Use baseline metrics as the "optimized" metrics for the summary
        optimized_accuracy = baseline_accuracy
        optimized_metrics = baseline_metrics
        optimized_predictions = baseline_predictions
        accuracy_improvement = 0.0
        results_table = baseline_results_table  # Use baseline results for the detailed examples
        
    else:
        # Test the optimized agent on holdout set
        print("\n" + "=" * 70)
        print("POST-OPTIMIZATION HOLDOUT EVALUATION")
        print("=" * 70)
        print(f"\nEvaluating optimized agent on holdout test set ({len(holdout_dataset)} penguins)...")

        # Create and configure agent
        test_agent = Agent(
            model=get_openai_model(config.agent_model),
            instructions=config.agent_instructions,
            output_type=SpeciesPrediction,
        )


        # Track results
        correct_predictions = 0
        total_predictions = 0
        results_table = []

        # Apply optimized configuration and test on holdout set
        with result.apply_best_to(agent=test_agent, input_type=PenguinInput):
            
            test_signature_agent = SignatureAgent(
                test_agent,
                input_type=PenguinInput,
                optimize_tools=True,
            )
            for i, data_inst in enumerate(holdout_dataset, 1):
                # Input is already a PenguinInput instance
                test_input = data_inst.input

                # Get ground truth from metadata
                actual = data_inst.metadata.get("label")

                # Run prediction
                try:
                    test_result = test_signature_agent.run_signature_sync(test_input)
                    predicted = test_result.output.species
                    confidence = test_result.output.confidence
                    reasoning = test_result.output.reasoning

                    # Check if correct
                    is_correct = predicted == actual
                    if is_correct:
                        correct_predictions += 1
                    total_predictions += 1

                    # Store result
                    results_table.append(
                        {
                            "case": i,
                            "bill_length": test_input.bill_length_mm,
                            "bill_depth": test_input.bill_depth_mm,
                            "flipper_length": test_input.flipper_length_mm,
                            "body_mass": test_input.body_mass_g,
                            "sex": test_input.sex,
                            "island": test_input.island,
                            "predicted": predicted,
                            "actual": actual,
                            "confidence": confidence,
                            "correct": is_correct,
                            "reasoning": reasoning,
                        }
                    )

                except Exception as e:
                    print(f"\n⚠️  Error on test case {i}: {e}")
                    results_table.append(
                        {
                            "case": i,
                            "bill_length": test_input.bill_length_mm,
                            "bill_depth": test_input.bill_depth_mm,
                            "flipper_length": test_input.flipper_length_mm,
                            "body_mass": test_input.body_mass_g,
                            "sex": test_input.sex,
                            "island": test_input.island,
                            "predicted": "ERROR",
                            "actual": actual,
                            "confidence": 0.0,
                            "correct": False,
                            "reasoning": str(e),
                        }
                    )

        # Calculate optimized classification metrics
        optimized_predictions = [row["predicted"] for row in results_table if row["predicted"] != "ERROR"]
        optimized_actuals = [row["actual"] for row in results_table if row["predicted"] != "ERROR"]
        
        if optimized_predictions:  # Only calculate if we have valid predictions
            optimized_metrics = calculate_classification_metrics(optimized_predictions, optimized_actuals)
        
        # Print results table
        print("\n📋 Optimized Agent Prediction Details:")
        print("-" * 110)
        print(
            f"{'#':<4} {'Island':<10} {'Sex':<7} {'Bill L':<8} {'Bill D':<8} {'Predicted':<12} {'Actual':<12} {'Conf':<6} {'Result':<8}"
        )
        print("-" * 110)

        for row in results_table:
            result_symbol = "✓" if row["correct"] else "✗"
            print(
                f"{row['case']:<4} {row['island']:<10} {row['sex']:<7} {row['bill_length']:<8.1f} {row['bill_depth']:<8.1f} "
                f"{row['predicted']:<12} {row['actual']:<12} {row['confidence']:<6.2f} {result_symbol:<8}"
            )

        print("-" * 110)

        # Calculate and display accuracy
        if total_predictions > 0:
            optimized_accuracy = correct_predictions / total_predictions
            print(
                f"📊 OPTIMIZED ACCURACY: {optimized_accuracy:.2%} ({correct_predictions}/{total_predictions})"
            )
            
            # Compare with baseline
            if baseline_total_predictions > 0:
                accuracy_improvement = optimized_accuracy - baseline_accuracy
                improvement_symbol = "📈" if accuracy_improvement > 0 else "📉" if accuracy_improvement < 0 else "➡️"
                print(
                    f"{improvement_symbol} IMPROVEMENT vs BASELINE: {accuracy_improvement:+.2%} "
                    f"({baseline_accuracy:.2%} → {optimized_accuracy:.2%})"
                )
        
        print("-" * 110)
        
        # Print detailed classification metrics
        if optimized_predictions:
            print_classification_report(optimized_metrics, "📊 Optimized Classification Report")

    # Show a few detailed examples
    print("\n" + "=" * 70)
    if has_improvement:
        print("SAMPLE DETAILED PREDICTIONS - OPTIMIZED (First 5 Cases)")
    else:
        print("SAMPLE DETAILED PREDICTIONS - BASELINE (First 5 Cases)")
    print("=" * 70)

    for i, row in enumerate(results_table[:5], 1):  # Show first 5
        result_symbol = "✓ CORRECT" if row["correct"] else "✗ INCORRECT"
        print(f"\n--- Case {row['case']} ({result_symbol}) ---")
        print(
            f"Penguin: {row['island']} island, {row['sex']}, "
            f"bill {row['bill_length']:.1f}mm x {row['bill_depth']:.1f}mm, "
            f"flipper {row['flipper_length']:.0f}mm, {row['body_mass']:.0f}g"
        )
        print(
            f"Predicted: {row['predicted'].upper()} (confidence: {row['confidence']:.2%})"
        )
        print(f"Actual: {row['actual'].upper()}")
        print(
            f"Reasoning: {row['reasoning'][:150]}..."
            if len(row["reasoning"]) > 150
            else f"Reasoning: {row['reasoning']}"
        )

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print("\n📊 Dataset Configuration:")
    print(f"   └─ Total Samples: {total_samples}")
    print(f"   └─ Training: {len(trainset)} | Validation: {len(valset)} | Holdout: {len(holdout_dataset)}")
    
    if has_improvement:
        print("\n📈 Performance Comparison (Holdout Test Set):")
        print(f"   {'Metric':<25} {'Baseline':<15} {'Optimized':<15} {'Improvement':<15}")
        print(f"   {'-'*70}")
        print(f"   {'Accuracy':<25} {baseline_accuracy:<15.3f} {optimized_accuracy:<15.3f} {accuracy_improvement:+.3f}")
        
        if optimized_predictions:
            # Show comparison between baseline and optimized
            f1_improvement = optimized_metrics['f1_macro'] - baseline_metrics['f1_macro']
            precision_improvement = optimized_metrics['precision_macro'] - baseline_metrics['precision_macro']
            recall_improvement = optimized_metrics['recall_macro'] - baseline_metrics['recall_macro']
            
            print(f"   {'Precision (Macro)':<25} {baseline_metrics['precision_macro']:<15.3f} {optimized_metrics['precision_macro']:<15.3f} {precision_improvement:+.3f}")
            print(f"   {'Recall (Macro)':<25} {baseline_metrics['recall_macro']:<15.3f} {optimized_metrics['recall_macro']:<15.3f} {recall_improvement:+.3f}")
            print(f"   {'F1-Score (Macro)':<25} {baseline_metrics['f1_macro']:<15.3f} {optimized_metrics['f1_macro']:<15.3f} {f1_improvement:+.3f}")
    else:
        print("\n📈 Performance (Holdout Test Set - No Improvement):")
        print(f"   {'Metric':<25} {'Value':<15}")
        print(f"   {'-'*40}")
        print(f"   {'Accuracy':<25} {baseline_accuracy:<15.3f}")
        
        if optimized_predictions:
            # Show only baseline metrics (no comparison)
            print(f"   {'Precision (Macro)':<25} {baseline_metrics['precision_macro']:<15.3f}")
            print(f"   {'Recall (Macro)':<25} {baseline_metrics['recall_macro']:<15.3f}")
            print(f"   {'F1-Score (Macro)':<25} {baseline_metrics['f1_macro']:<15.3f}")
    
    print("\n💰 Optimization Cost:")
    print(f"   └─ Total Tokens: {result.gepa_usage.input_tokens + result.gepa_usage.output_tokens:,}")
    print(f"   └─ API Calls: {result.gepa_usage.requests}")
    print(f"   └─ Iterations: {result.num_iterations}")
    
    print("\n" + "=" * 70)
    print("✅ EXAMPLE COMPLETE!")
    print("=" * 70 + "\n")

    return result


if __name__ == "__main__":
    # Run the example
    result = main()
