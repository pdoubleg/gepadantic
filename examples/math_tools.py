"""Complete example demonstrating GEPA scaffolding with a math problems solving task.

This example shows how to optimize a task using an agent that can call a tool (Python sandbox).
"""

from __future__ import annotations

import os
import sys

from pydantic import BaseModel, Field
from pydantic_ai import Agent

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gepadantic.data_utils import split_dataset
from src.gepadantic.scaffold import GepaConfig, run_optimization_pipeline
from src.gepadantic.signature_agent import SignatureAgent
from src.gepadantic.types import DataInstWithInput, RolloutOutput
from utils import run_python_tool

TASK_LLM = "gpt-4.1-nano"
REFLECTION_LLM = "gpt-4.1"


class MathProblemInput(BaseModel):
    """Input model for math problems requiring exact numeric answers.

    Attributes:
        problem: A math problem that needs an exact numeric answer.
    """

    problem: str = Field(
        description="A math problem that needs an exact numeric answer."
    )


class MathProblemOutput(BaseModel):
    """The solved value and the code that produced it.

    Attributes:
        explanation: Two sentences max summarizing how the code solves the problem.
        expression: The complete Python script used to compute the answer.
        answer: Numeric answer rounded only if necessary.
    """

    explanation: str = Field(
        description="Two sentences max summarizing how the code solves the problem."
    )
    expression: str = Field(
        description="The complete Python script used to compute the answer (can be multi-line with imports)."
    )
    answer: float = Field(description="Numeric answer rounded only if necessary.")


# Create dataset conforming to DataInstWithInput structure
dataset: list[DataInstWithInput[MathProblemInput]] = [
    DataInstWithInput(
        input=MathProblemInput(problem="Compute 100 choose 5."),
        message_history=None,
        case_id="comb-100-5",
        metadata=dict(
            expected_answer=75287520.0,
            tolerance=1e-09,
            feedback="Use the combinatorics function from the math module to compute binomial coefficients directly.",
            ideal_expression="math.comb(100, 5)",
            expected_output=MathProblemOutput(
                answer=75287520.0,
                expression="math.comb(100, 5)",
                explanation="Use the combinatorics function from the math module to compute binomial coefficients directly.",
            ).model_dump(),
        ),
    ),
    DataInstWithInput(
        input=MathProblemInput(
            problem="Compute the sum of the digits of 2 raised to the 200th power."
        ),
        message_history=None,
        case_id="digit-sum-2-200",
        metadata=dict(
            expected_answer=256.0,
            tolerance=1e-09,
            feedback="Convert the large integer to a string first, then sum each digit character after converting back to int.",
            ideal_expression="sum(int(d) for d in str(2 ** 200))",
            expected_output=MathProblemOutput(
                answer=256.0,
                expression="sum(int(d) for d in str(2 ** 200))",
                explanation="Convert the large integer to a string first, then sum each digit character after converting back to int.",
            ).model_dump(),
        ),
    ),
    DataInstWithInput(
        input=MathProblemInput(
            problem="Multiply the primes [2, 3, 5, 7, 11, 13, 17, 19, 23, 29] together."
        ),
        message_history=None,
        case_id="primorial-product",
        metadata=dict(
            expected_answer=6469693230.0,
            tolerance=1e-09,
            feedback="Use the product aggregation function from the math module to multiply list elements.",
            ideal_expression="math.prod([2, 3, 5, 7, 11, 13, 17, 19, 23, 29])",
            expected_output=MathProblemOutput(
                answer=6469693230.0,
                expression="math.prod([2, 3, 5, 7, 11, 13, 17, 19, 23, 29])",
                explanation="Use the product aggregation function from the math module to multiply list elements.",
            ).model_dump(),
        ),
    ),
    DataInstWithInput(
        input=MathProblemInput(problem="Sum all integers between 10 and 20."),
        message_history=None,
        case_id="range-ambiguity-between",
        metadata=dict(
            expected_answer=135.0,
            tolerance=1e-09,
            feedback="The phrase 'between A and B' typically excludes both endpoints. Verify whether the count matches your interpretation.",
            ideal_expression="sum(range(11, 20))",
            expected_output=MathProblemOutput(
                answer=135.0,
                expression="sum(range(11, 20))",
                explanation="The phrase 'between A and B' typically excludes both endpoints. Verify whether the count matches your interpretation.",
            ).model_dump(),
        ),
    ),
    DataInstWithInput(
        input=MathProblemInput(
            problem="Calculate the average of squares from 5 through 15."
        ),
        message_history=None,
        case_id="range-ambiguity-from-through",
        metadata=dict(
            expected_answer=110.0,
            tolerance=1e-09,
            feedback="The phrase 'from A through B' indicates inclusive bounds. Check that your range includes both endpoints.",
            ideal_expression="sum(n**2 for n in range(5, 16)) / len(range(5, 16))",
            expected_output=MathProblemOutput(
                answer=110.0,
                expression="sum(n**2 for n in range(5, 16)) / len(range(5, 16))",
                explanation="The phrase 'from A through B' indicates inclusive bounds. Check that your range includes both endpoints.",
            ).model_dump(),
        ),
    ),
    DataInstWithInput(
        input=MathProblemInput(
            problem="Find the product of all even numbers up to 12."
        ),
        message_history=None,
        case_id="implicit-inclusive-up-to",
        metadata=dict(
            expected_answer=46080.0,
            tolerance=1e-09,
            feedback="The phrase 'up to N' is ambiguous—it may include or exclude N. Verify against the expected result which interpretation is correct.",
            ideal_expression="math.prod(range(2, 13, 2))",
            expected_output=MathProblemOutput(
                answer=46080.0,
                expression="math.prod(range(2, 13, 2))",
                explanation="The phrase 'up to N' is ambiguous—it may include or exclude N. Verify against the expected result which interpretation is correct.",
            ).model_dump(),
        ),
    ),
    DataInstWithInput(
        input=MathProblemInput(
            problem="Approximate the square root of 50 to the nearest integer."
        ),
        message_history=None,
        case_id="rounding-specification",
        metadata=dict(
            expected_answer=7.0,
            tolerance=1e-09,
            feedback="Use the rounding function explicitly when the problem requests rounding to a specific precision.",
            ideal_expression="round(math.sqrt(50))",
            expected_output=MathProblemOutput(
                answer=7.0,
                expression="round(math.sqrt(50))",
                explanation="Use the rounding function explicitly when the problem requests rounding to a specific precision.",
            ).model_dump(),
        ),
    ),
    DataInstWithInput(
        input=MathProblemInput(problem="What is 100 divided by 7, rounded down?"),
        message_history=None,
        case_id="floor-vs-truncate",
        metadata=dict(
            expected_answer=14.0,
            tolerance=1e-09,
            feedback="Rounded down means floor division. Use the appropriate math function for floor operations.",
            ideal_expression="math.floor(100 / 7)",
            expected_output=MathProblemOutput(
                answer=14.0,
                expression="math.floor(100 / 7)",
                explanation="Rounded down means floor division. Use the appropriate math function for floor operations.",
            ).model_dump(),
        ),
    ),
    DataInstWithInput(
        input=MathProblemInput(
            problem="Sum integers greater than 5 and less than or equal to 15."
        ),
        message_history=None,
        case_id="mixed-boundaries",
        metadata=dict(
            expected_answer=105.0,
            tolerance=1e-09,
            feedback="Pay attention to strict inequalities (>) versus inclusive inequalities (≤). Translate each bound correctly.",
            ideal_expression="sum(range(6, 16))",
            expected_output=MathProblemOutput(
                answer=105.0,
                expression="sum(range(6, 16))",
                explanation="Pay attention to strict inequalities (>) versus inclusive inequalities (≤). Translate each bound correctly.",
            ).model_dump(),
        ),
    ),
    DataInstWithInput(
        input=MathProblemInput(
            problem="Find the sum of all primes less than 50, then multiply that sum by the largest prime less than 50."
        ),
        message_history=None,
        case_id="conditional-prime-product",
        metadata=dict(
            expected_answer=15416.0,
            tolerance=1e-09,
            feedback="Break the problem into steps: first identify all primes in the range, then compute the sum and find the maximum, then multiply them.",
            ideal_expression="(lambda primes: sum(primes) * max(primes))([n for n in range(2, 50) if all(n % d for d in range(2, int(n**0.5) + 1))])",
            expected_output=MathProblemOutput(
                answer=15416.0,
                expression="(lambda primes: sum(primes) * max(primes))([n for n in range(2, 50) if all(n % d for d in range(2, int(n**0.5) + 1))])",
                explanation="Break the problem into steps: first identify all primes in the range, then compute the sum and find the maximum, then multiply them.",
            ).model_dump(),
        ),
    ),
    DataInstWithInput(
        input=MathProblemInput(
            problem="Calculate the sum of the digits of 15 factorial."
        ),
        message_history=None,
        case_id="nested-digit-sum",
        metadata=dict(
            expected_answer=45.0,
            tolerance=1e-09,
            feedback="Compute the factorial first, convert to string, then sum the individual digit characters.",
            ideal_expression="sum(int(d) for d in str(math.factorial(15)))",
            expected_output=MathProblemOutput(
                answer=45.0,
                expression="sum(int(d) for d in str(math.factorial(15)))",
                explanation="Compute the factorial first, convert to string, then sum the individual digit characters.",
            ).model_dump(),
        ),
    ),
    DataInstWithInput(
        input=MathProblemInput(
            problem="Find the 20th Tribonacci number, where T(0)=0, T(1)=1, T(2)=1, and T(n)=T(n-1)+T(n-2)+T(n-3)."
        ),
        message_history=None,
        case_id="tribonacci-20",
        metadata=dict(
            expected_answer=35890.0,
            tolerance=1e-09,
            feedback="Iteratively compute the sequence using a list to track the last three values, updating as you progress.",
            ideal_expression="(lambda: [t := [0, 1, 1], [t.append(sum(t[-3:])) for _ in range(17)], t[-1]][2])()",
            expected_output=MathProblemOutput(
                answer=35890.0,
                expression="(lambda: [t := [0, 1, 1], [t.append(sum(t[-3:])) for _ in range(17)], t[-1]][2])()",
                explanation="Iteratively compute the sequence using a list to track the last three values, updating as you progress.",
            ).model_dump(),
        ),
    ),
    DataInstWithInput(
        input=MathProblemInput(
            problem="Compute the LCM of 12, 18, and 24, then find the GCD of that result and 144."
        ),
        message_history=None,
        case_id="gcd-lcm-chain",
        metadata=dict(
            expected_answer=72.0,
            tolerance=1e-09,
            feedback="Compute LCM step-by-step for pairs using the formula LCM(a,b) = |a*b|/GCD(a,b), then apply GCD to the final result.",
            ideal_expression="math.gcd((lambda a, b: abs(a * b) // math.gcd(a, b))((lambda a, b: abs(a * b) // math.gcd(a, b))(12, 18), 24), 144)",
            expected_output=MathProblemOutput(
                answer=72.0,
                expression="math.gcd((lambda a, b: abs(a * b) // math.gcd(a, b))((lambda a, b: abs(a * b) // math.gcd(a, b))(12, 18), 24), 144)",
                explanation="Compute LCM step-by-step for pairs using the formula LCM(a,b) = |a*b|/GCD(a,b), then apply GCD to the final result.",
            ).model_dump(),
        ),
    ),
    DataInstWithInput(
        input=MathProblemInput(
            problem="Calculate Euler's totient function φ(72)—the count of integers from 1 to 72 that are coprime with 72."
        ),
        message_history=None,
        case_id="totient-composite",
        metadata=dict(
            expected_answer=24.0,
            tolerance=1e-09,
            feedback="Count how many integers in the range have a GCD of 1 with the target number.",
            ideal_expression="sum(1 for k in range(1, 73) if math.gcd(k, 72) == 1)",
            expected_output=MathProblemOutput(
                answer=24.0,
                expression="sum(1 for k in range(1, 73) if math.gcd(k, 72) == 1)",
                explanation="Count how many integers in the range have a GCD of 1 with the target number.",
            ).model_dump(),
        ),
    ),
    DataInstWithInput(
        input=MathProblemInput(
            problem="Compute the alternating sum 1² - 2² + 3² - 4² + ... + 19² - 20²."
        ),
        message_history=None,
        case_id="alternating-sum-squares",
        metadata=dict(
            expected_answer=-210.0,
            tolerance=1e-09,
            feedback="Use a sign factor that alternates based on the index: positive for odd indices, negative for even.",
            ideal_expression="sum(((-1) ** (n + 1)) * (n ** 2) for n in range(1, 21))",
            expected_output=MathProblemOutput(
                answer=-210.0,
                expression="sum(((-1) ** (n + 1)) * (n ** 2) for n in range(1, 21))",
                explanation="Use a sign factor that alternates based on the index: positive for odd indices, negative for even.",
            ).model_dump(),
        ),
    ),
    DataInstWithInput(
        input=MathProblemInput(
            problem="What is 100 factorial divided by 99 factorial?"
        ),
        message_history=None,
        case_id="precision-trap-large-factorial",
        metadata=dict(
            expected_answer=100.0,
            tolerance=1e-09,
            feedback="Notice the mathematical identity: n! / (n-1)! = n. Avoid computing huge factorials separately if simplification is possible.",
            ideal_expression="math.factorial(100) // math.factorial(99)",
            expected_output=MathProblemOutput(
                answer=100.0,
                expression="math.factorial(100) // math.factorial(99)",
                explanation="Notice the mathematical identity: n! / (n-1)! = n. Avoid computing huge factorials separately if simplification is possible.",
            ).model_dump(),
        ),
    ),
    DataInstWithInput(
        input=MathProblemInput(problem="Sum all integers from 20 to 10."),
        message_history=None,
        case_id="empty-range-edge",
        metadata=dict(
            expected_answer=0.0,
            tolerance=1e-09,
            feedback="When the start exceeds the stop in a range, the result is an empty sequence. The sum of an empty sequence is zero.",
            ideal_expression="sum(range(20, 10))",
            expected_output=MathProblemOutput(
                answer=0.0,
                expression="sum(range(20, 10))",
                explanation="When the start exceeds the stop in a range, the result is an empty sequence. The sum of an empty sequence is zero.",
            ).model_dump(),
        ),
    ),
    DataInstWithInput(
        input=MathProblemInput(
            problem="Find the average of all multiples of 7 between 100 and 105."
        ),
        message_history=None,
        case_id="degenerate-average",
        metadata=dict(
            expected_answer=105.0,
            tolerance=1e-09,
            feedback="Only one multiple exists in this narrow range. Ensure you handle single-element averages correctly.",
            ideal_expression="sum(range(105, 106, 7)) / max(len(range(105, 106, 7)), 1)",
            expected_output=MathProblemOutput(
                answer=105.0,
                expression="sum(range(105, 106, 7)) / max(len(range(105, 106, 7)), 1)",
                explanation="Only one multiple exists in this narrow range. Ensure you handle single-element averages correctly.",
            ).model_dump(),
        ),
    ),
    DataInstWithInput(
        input=MathProblemInput(problem="Calculate (-1)^50 + (-1)^51 + (-1)^52."),
        message_history=None,
        case_id="sign-heavy-expression",
        metadata=dict(
            expected_answer=1.0,
            tolerance=1e-09,
            feedback="Even powers of -1 yield 1, odd powers yield -1. Sum the results directly.",
            ideal_expression="(-1)**50 + (-1)**51 + (-1)**52",
            expected_output=MathProblemOutput(
                answer=1.0,
                expression="(-1)**50 + (-1)**51 + (-1)**52",
                explanation="Even powers of -1 yield 1, odd powers yield -1. Sum the results directly.",
            ).model_dump(),
        ),
    ),
    DataInstWithInput(
        input=MathProblemInput(problem="Sum all integers strictly between 50 and 60."),
        message_history=None,
        case_id="between-50-60-exclusive",
        metadata=dict(
            expected_answer=495.0,
            tolerance=1e-09,
            feedback='"Between A and B" (without saying inclusive) means exclude both endpoints. Use 51 through 59 here.',
            ideal_expression="sum(range(51, 60))",
            expected_output=MathProblemOutput(
                answer=495.0,
                expression="sum(range(51, 60))",
                explanation='"Between A and B" (without saying inclusive) means exclude both endpoints. Use 51 through 59 here.',
            ).model_dump(),
        ),
    ),
    DataInstWithInput(
        input=MathProblemInput(problem="Sum the integers strictly between -5 and 5."),
        message_history=None,
        case_id="between-neg5-5-exclusive",
        metadata=dict(
            expected_answer=0.0,
            tolerance=1e-09,
            feedback="Strictly between means -4 through 4. The positive and negative values cancel out to zero.",
            ideal_expression="sum(range(-4, 5))",
            expected_output=MathProblemOutput(
                answer=0.0,
                expression="sum(range(-4, 5))",
                explanation="Strictly between means -4 through 4. The positive and negative values cancel out to zero.",
            ).model_dump(),
        ),
    ),
    DataInstWithInput(
        input=MathProblemInput(problem="Sum the integers strictly between 1 and 2."),
        message_history=None,
        case_id="between-1-2-empty",
        metadata=dict(
            expected_answer=0.0,
            tolerance=1e-09,
            feedback="There are no integers strictly between consecutive integers. Return 0 for an empty range.",
            ideal_expression="sum(range(2, 2))",
            expected_output=MathProblemOutput(
                answer=0.0,
                expression="sum(range(2, 2))",
                explanation="There are no integers strictly between consecutive integers. Return 0 for an empty range.",
            ).model_dump(),
        ),
    ),
    DataInstWithInput(
        input=MathProblemInput(
            problem="Sum the integers from 30 down to 20, inclusive."
        ),
        message_history=None,
        case_id="descending-inclusive-30-20",
        metadata=dict(
            expected_answer=275.0,
            tolerance=1e-09,
            feedback="Descending ranges require a negative step. Include both endpoints exactly once.",
            ideal_expression="sum(range(30, 19, -1))",
            expected_output=MathProblemOutput(
                answer=275.0,
                expression="sum(range(30, 19, -1))",
                explanation="Descending ranges require a negative step. Include both endpoints exactly once.",
            ).model_dump(),
        ),
    ),
    DataInstWithInput(
        input=MathProblemInput(
            problem="Sum the integers from 30 down to 20, excluding both endpoints."
        ),
        message_history=None,
        case_id="descending-exclusive-30-20",
        metadata=dict(
            expected_answer=225.0,
            tolerance=1e-09,
            feedback="Exclude the endpoints by starting at 29 and stopping after 21 when stepping downward.",
            ideal_expression="sum(range(29, 20, -1))",
            expected_output=MathProblemOutput(
                answer=225.0,
                expression="sum(range(29, 20, -1))",
                explanation="Exclude the endpoints by starting at 29 and stopping after 21 when stepping downward.",
            ).model_dump(),
        ),
    ),
    DataInstWithInput(
        input=MathProblemInput(
            problem="Compute the average of the integers from 12 down to 8 (inclusive)."
        ),
        message_history=None,
        case_id="descending-average-12-8",
        metadata=dict(
            expected_answer=10.0,
            tolerance=1e-09,
            feedback="When iterating downward, the range still has 5 terms (12,11,10,9,8). Average them normally.",
            ideal_expression="sum(range(12, 7, -1)) / len(range(12, 7, -1))",
            expected_output=MathProblemOutput(
                answer=10.0,
                expression="sum(range(12, 7, -1)) / len(range(12, 7, -1))",
                explanation="When iterating downward, the range still has 5 terms (12,11,10,9,8). Average them normally.",
            ).model_dump(),
        ),
    ),
    DataInstWithInput(
        input=MathProblemInput(
            problem="Count the integers strictly between 10 and 11."
        ),
        message_history=None,
        case_id="between-10-11-empty",
        metadata=dict(
            expected_answer=0.0,
            tolerance=1e-09,
            feedback="Adjacent integers have zero strictly-between values. Guard against assuming at least one element.",
            ideal_expression="len(range(11, 11))",
            expected_output=MathProblemOutput(
                answer=0.0,
                expression="len(range(11, 11))",
                explanation="Adjacent integers have zero strictly-between values. Guard against assuming at least one element.",
            ).model_dump(),
        ),
    ),
    DataInstWithInput(
        input=MathProblemInput(
            problem="Sum the integers from -3 through 3 (inclusive)."
        ),
        message_history=None,
        case_id="inclusive-neg3-pos3",
        metadata=dict(
            expected_answer=0.0,
            tolerance=1e-09,
            feedback='"Through" means include both endpoints. The symmetric range cancels back to zero.',
            ideal_expression="sum(range(-3, 4))",
            expected_output=MathProblemOutput(
                answer=0.0,
                expression="sum(range(-3, 4))",
                explanation='"Through" means include both endpoints. The symmetric range cancels back to zero.',
            ).model_dump(),
        ),
    ),
    DataInstWithInput(
        input=MathProblemInput(
            problem="Find the 25th Tribonacci number when T(0)=0, T(1)=1, T(2)=1, and T(n)=T(n-1)+T(n-2)+T(n-3)."
        ),
        message_history=None,
        case_id="tribonacci-25",
        metadata=dict(
            expected_answer=1389537.0,
            tolerance=1e-09,
            feedback="Ensure the recurrence seeds are correct and iterate all the way to n=25 without off-by-one errors.",
            ideal_expression="(lambda: [t := [0, 1, 1], [t.append(sum(t[-3:])) for _ in range(23)], t[-1]][2])()",
            expected_output=MathProblemOutput(
                answer=1389537.0,
                expression="(lambda: [t := [0, 1, 1], [t.append(sum(t[-3:])) for _ in range(23)], t[-1]][2])()",
                explanation="Ensure the recurrence seeds are correct and iterate all the way to n=25 without off-by-one errors.",
            ).model_dump(),
        ),
    ),
    DataInstWithInput(
        input=MathProblemInput(
            problem="Compute the 30th Tribonacci number with the same base cases (0,1,1)."
        ),
        message_history=None,
        case_id="tribonacci-30",
        metadata=dict(
            expected_answer=29249425.0,
            tolerance=1e-09,
            feedback="Longer Tribonacci runs magnify seed mistakes; track the list carefully.",
            ideal_expression="(lambda: [t := [0, 1, 1], [t.append(sum(t[-3:])) for _ in range(28)], t[-1]][2])()",
            expected_output=MathProblemOutput(
                answer=29249425.0,
                expression="(lambda: [t := [0, 1, 1], [t.append(sum(t[-3:])) for _ in range(28)], t[-1]][2])()",
                explanation="Longer Tribonacci runs magnify seed mistakes; track the list carefully.",
            ).model_dump(),
        ),
    ),
]

agent = Agent(
    model=TASK_LLM,
    instructions=(
        "Solve math problems by calling the `run_python` sandbox tool. "
        "Write complete Python scripts with all necessary imports and print the final result. "
        "You may only use the Python standard library; third-party packages are unavailable."
    ),
    output_type=MathProblemOutput,
    tools=[run_python_tool],
)

signature_agent: SignatureAgent[None, MathProblemOutput] = SignatureAgent(
    agent,
    input_type=MathProblemInput,
    optimize_tools=True,
)


def metric(
    case: DataInstWithInput[MathProblemInput],
    output: RolloutOutput[MathProblemOutput],
) -> tuple[float, str | None]:
    if not output.success or output.result is None:
        return 0.0, output.error_message or "Agent failed to produce an output."

    predicted_output = output.result
    predicted = predicted_output.answer
    expression = (predicted_output.expression or "").strip()
    metadata = case.metadata
    if metadata is None:
        fallback_target = (
            case.expected_output.answer if case.expected_output is not None else 0.0
        )
        metadata = dict(expected_answer=fallback_target)
    tolerance = metadata.get("tolerance", 1e-9)
    target = metadata.get("expected_answer", 0.0)
    base_feedback = metadata.get("feedback", None)
    ideal_expression = metadata.get("ideal_expression", None)

    if not expression:
        hint = "Include the Python code you executed."
        if ideal_expression:
            hint = (
                f"{hint} For reference, one valid approach uses: `{ideal_expression}`."
            )
        prefix = f"{base_feedback} " if base_feedback else ""
        return 0.0, f"{prefix}Missing code used to compute the answer. {hint}"

    # We trust the agent's reported answer from the sandbox execution
    # The code is available for inspection but not re-executed in the metric
    if target is None:
        return 0.0, "Missing reference target."

    target_gap = abs(predicted - target)
    effective_tolerance = max(tolerance, 1e-9)
    if target_gap <= effective_tolerance:
        score = 1.0
        feedback = "Exact match within tolerance."
        return score, feedback
    else:
        normalized_error = target_gap / max(abs(target), 1.0)
        score = max(0.0, 1.0 - min(normalized_error * 10, 1.0))
        base = base_feedback or "Re-check the computation with Python."
        hint = (
            f"Answer {predicted} deviates from target {target} by {target_gap:.6g}; "
            "verify the computation logic and any rounding."
        )
        if ideal_expression:
            hint += f" A reliable approach uses: `{ideal_expression}`."
        feedback = f"{base} {hint}"
        return score, feedback


def main():
    """Main function to run the math tools example."""
    print("\n" + "=" * 70)
    print("Loading Math Problems Dataset")
    print("=" * 70)
    print(f"Total dataset size: {len(dataset)}")

    # Split dataset into train/val and holdout test sets
    # Use 70% for train/val, 30% for holdout testing
    train_val_size = int(len(dataset) * 0.7)
    train_val_dataset = dataset[:train_val_size]
    holdout_dataset = dataset[train_val_size:]

    print(f"Train/Val set: {len(train_val_dataset)} problems")
    print(f"Holdout test set: {len(holdout_dataset)} problems")

    # Split train/val dataset into training and validation sets
    trainset, valset = split_dataset(
        train_val_dataset, train_ratio=0.7, shuffle=True, random_seed=42
    )
    print(f"  - Training: {len(trainset)} problems")
    print(f"  - Validation: {len(valset)} problems")

    # Configure the optimization
    reflection_model = REFLECTION_LLM

    config = GepaConfig(
        # Agent configuration
        agent=agent,
        input_type=MathProblemInput,
        output_type=MathProblemOutput,
        # Data and evaluation
        trainset=trainset,
        valset=valset,
        metric=metric,
        # Budget
        max_full_evals=3,
        # Optimization parameters
        module_selector="all",
        candidate_selection_strategy="pareto",
        optimize_tools=True,
        use_merge=True,
        # LLM for reflection
        reflection_model=reflection_model,
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
    print("Starting GEPA Optimization")
    print("=" * 70)
    print("Task: Math Problem Solving with Python Sandbox")
    print(f"Model: {config.agent_model}")
    print(f"Reflection Model: {config.reflection_model}")
    print(f"Training set: {len(config.trainset)} problems")
    print(f"Validation set: {len(config.valset) if config.valset else 0} problems")
    print(f"Max metric calls: {config.estimated_metric_calls}")
    print("=" * 70 + "\n")

    # Evaluate baseline agent on holdout set
    print("Evaluating baseline agent on holdout set...")
    baseline_correct = 0
    baseline_results_table = []

    # Create baseline signature agent
    baseline_signature_agent = SignatureAgent(
        agent,
        input_type=MathProblemInput,
        optimize_tools=True,
    )

    for test_case in holdout_dataset:
        test_input = test_case.input
        case_id = test_case.case_id

        # Run prediction
        test_result = baseline_signature_agent.run_signature_sync(test_input)

        # Create RolloutOutput for metric evaluation
        rollout_output = RolloutOutput(
            success=True,
            result=test_result.output,
            error_message=None,
        )
        
        # Evaluate with metric
        score, feedback = metric(test_case, rollout_output)
        
        is_correct = score >= 0.99  # Consider score >= 0.99 as correct
        if is_correct:
            baseline_correct += 1

        # Store result
        baseline_results_table.append(
            {
                "case_id": case_id,
                "problem": test_input.problem[:50] + "..."
                if len(test_input.problem) > 50
                else test_input.problem,
                "answer": test_result.output.answer if test_result.output else "N/A",
                "expected": test_case.metadata.get("expected_answer", "N/A")
                if test_case.metadata
                else "N/A",
                "score": score,
                "correct": is_correct,
            }
        )

    # Print baseline results
    print("\nBaseline Results:")
    print("-" * 100)
    print(f"{'Case ID':<30} {'Score':<8} {'Answer':<15} {'Expected':<15} {'Result':<8}")
    print("-" * 100)
    for row in baseline_results_table:
        result_symbol = "✓" if row["correct"] else "✗"
        print(
            f"{row['case_id']:<30} {row['score']:<8.2f} {str(row['answer']):<15} {str(row['expected']):<15} {result_symbol:<8}"
        )
    print("-" * 100)
    baseline_accuracy = (
        baseline_correct / len(holdout_dataset) if holdout_dataset else 0
    )
    print(f"Baseline Accuracy: {baseline_accuracy:.2%} ({baseline_correct}/{len(holdout_dataset)})")
    print("-" * 100)

    # Run the optimization
    result = run_optimization_pipeline(config)

    # Display optimization results
    print("\n" + "=" * 70)
    print("Optimization Complete!")
    print("=" * 70)
    print(f"Best Score: {result.best_score:.4f}")

    if result.original_score is not None:
        print(f"Original Score: {result.original_score:.4f}")
        improvement = result.improvement_ratio()
        if improvement is not None:
            print(f"Improvement: {improvement:+.2%}")

    print(f"Iterations: {result.num_iterations}")
    print(f"Metric Calls: {result.num_metric_calls}")
    print(f"GEPA Input Tokens: {result.gepa_usage.input_tokens}")
    print(f"GEPA Output Tokens: {result.gepa_usage.output_tokens}")
    print(f"GEPA API calls: {result.gepa_usage.requests}")

    print("\nOptimized Components:")
    for component_name, component_value in result.best_candidate.items():
        print(f"\n{component_name}:")
        print(f"  {component_value}")

    print("\n" + "=" * 70)

    # Test the optimized agent on holdout set
    print("\nEvaluating optimized agent on holdout test set...")
    print("=" * 70)

    # Create test signature agent
    test_signature_agent = SignatureAgent(
        agent,
        input_type=MathProblemInput,
        optimize_tools=True,
    )

    # Track results
    correct = 0
    results_table = []

    # Apply optimized configuration and test on holdout set
    with result.apply_best_to(agent=agent, input_type=MathProblemInput):
        for test_case in holdout_dataset:
            test_input = test_case.input
            case_id = test_case.case_id

            try:
                # Run prediction
                test_result = test_signature_agent.run_signature_sync(test_input)

                # Create RolloutOutput for metric evaluation
                rollout_output = RolloutOutput(
                    success=True,
                    result=test_result.output,
                    error_message=None,
                )
                
                # Evaluate with metric
                score, feedback = metric(test_case, rollout_output)
                
                is_correct = score >= 0.99  # Consider score >= 0.99 as correct
                if is_correct:
                    correct += 1

                # Store result
                results_table.append(
                    {
                        "case_id": case_id,
                        "problem": test_input.problem[:50] + "..."
                        if len(test_input.problem) > 50
                        else test_input.problem,
                        "answer": test_result.output.answer
                        if test_result.output
                        else "N/A",
                        "expected": test_case.metadata.get("expected_answer", "N/A")
                        if test_case.metadata
                        else "N/A",
                        "score": score,
                        "correct": is_correct,
                        "feedback": feedback,
                    }
                )

            except Exception as e:
                print(f"\n⚠️  Error on test case {case_id}: {e}")
                results_table.append(
                    {
                        "case_id": case_id,
                        "problem": test_input.problem[:50] + "..."
                        if len(test_input.problem) > 50
                        else test_input.problem,
                        "answer": "ERROR",
                        "expected": test_case.metadata.get("expected_answer", "N/A")
                        if test_case.metadata
                        else "N/A",
                        "score": 0.0,
                        "correct": False,
                        "feedback": str(e),
                    }
                )

    # Print results table
    print(f"\nHoldout Test Results ({len(holdout_dataset)} problems):")
    print("-" * 100)
    print(f"{'Case ID':<30} {'Score':<8} {'Answer':<15} {'Expected':<15} {'Result':<8}")
    print("-" * 100)

    for row in results_table:
        result_symbol = "✓" if row["correct"] else "✗"
        print(
            f"{row['case_id']:<30} {row['score']:<8.2f} {str(row['answer']):<15} {str(row['expected']):<15} {result_symbol:<8}"
        )

    print("-" * 100)

    # Calculate and display metrics
    if len(holdout_dataset) > 0:
        accuracy = correct / len(holdout_dataset)
        print(f"\n📊 Optimized Accuracy: {accuracy:.2%} ({correct}/{len(holdout_dataset)})")
        print(f"📊 Baseline Accuracy: {baseline_accuracy:.2%} ({baseline_correct}/{len(holdout_dataset)})")
        
        # Show improvement over baseline
        accuracy_diff = accuracy - baseline_accuracy
        if baseline_accuracy > 0:
            accuracy_improvement = accuracy_diff / baseline_accuracy
            print(f"📈 Improvement: {accuracy_improvement:+.2%} ({accuracy_diff:+.2%} absolute)")
        else:
            print(f"📈 Improvement: {accuracy_diff:+.2%} (absolute)")

    # Show a few detailed examples
    print("\n" + "=" * 70)
    print("Sample Detailed Predictions:")
    print("=" * 70)

    for i, row in enumerate(results_table[:3], 1):  # Show first 3
        result_symbol = "✓ CORRECT" if row["correct"] else "✗ INCORRECT"
        print(f"\n--- Case: {row['case_id']} ({result_symbol}) ---")
        print(f"Problem: {row['problem']}")
        print(f"Predicted Answer: {row['answer']} (score: {row['score']:.2f})")
        print(f"Expected Answer: {row['expected']}")
        if row["feedback"]:
            feedback_preview = (
                row["feedback"][:200] + "..."
                if len(row["feedback"]) > 200
                else row["feedback"]
            )
            print(f"Feedback: {feedback_preview}")

    print("\n" + "=" * 70)
    print("Example Complete!")
    print("=" * 70 + "\n")

    return result


if __name__ == "__main__":
    # Run the example
    result = main()
