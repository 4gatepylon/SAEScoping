from __future__ import annotations

from beartype import beartype
import dspy
from datasets import load_dataset

# TODO(Adriano) review the actual way this is done and make sure it is correct
# TODO(Adriano) define a canonical way to represent a dataset with verifiable rewards (or basically judgeable rewards) and then re-work this into that
# (we should only ever write code to turn a dataset into that form; the rest of the splitting, test/val/etc... should ever be done by a single function
# whose purpose is that---that function should support a dataset cache so that we can merge datasets but also have a held out test/validation set)


@beartype
def get_gsm8k_dataset(
    n_samples: int = 100,
    train_split_ratio: float = 0.8,
    val_split_ratio: float = 0.1,
    test_split_ratio: float = 0.1,
    seed: int = 0,
) -> dict[str, list[dspy.Example]]:
    """Get the GSM8K dataset split into train/val/test.

    Args:
        n_samples: Total number of samples to use.
        train_split_ratio: Ratio for training set.
        val_split_ratio: Ratio for validation set.
        test_split_ratio: Ratio for test set.
        seed: Random seed for shuffling.

    Returns:
        Dictionary with 'train', 'val', 'test' keys containing dspy.Example lists.
    """
    if abs(train_split_ratio + val_split_ratio + test_split_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {train_split_ratio + val_split_ratio + test_split_ratio}")

    # Load GSM8K from HuggingFace
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    dataset = dataset.shuffle(seed=seed)

    # Limit to n_samples
    n_samples = min(n_samples, len(dataset))
    dataset = dataset.select(range(n_samples))

    # Convert to dspy.Example objects
    # GSM8K has 'question' and 'answer' fields
    # The answer field contains the reasoning followed by "#### <final_answer>"
    examples = [
        dspy.Example(
            problem=row["question"],
            solution=row["answer"],
            answer=row["answer"].rsplit("####", 1)[-1].strip().lower() if "####" in row["answer"] else row["answer"],
        ).with_inputs("problem")
        for row in dataset
    ]

    # Split into train/val/test
    train_end = int(train_split_ratio * len(examples))
    val_end = int((train_split_ratio + val_split_ratio) * len(examples))

    return {
        "train": examples[:train_end],
        "val": examples[train_end:val_end],
        "test": examples[val_end:],
    }


def _parse_numeric_answer(answer: str) -> int | None:
    """Extract the final numeric answer from a string.

    Handles formats like "#### 42", "The answer is 42", or just "42".
    Returns None if no valid integer can be extracted.
    """
    if not answer:
        return None
    try:
        # If it contains ####, extract after it
        if "####" in answer:
            answer = answer.split("####")[-1].strip()
        # Find the last token containing digits
        tokens = [t for t in answer.split() if any(c.isdigit() for c in t)]
        if not tokens:
            return None
        last_token = tokens[-1]
        # Remove non-digit chars (commas, periods, etc.) but keep negative sign
        cleaned = "".join(c for c in last_token if c.isdigit() or c == "-")
        if cleaned.startswith("-"):
            cleaned = "-" + cleaned[1:].replace("-", "")
        return int(cleaned) if cleaned and cleaned != "-" else None
    except (ValueError, IndexError):
        return None


@beartype
def gsm8k_regex_metric(example: dspy.Example, prediction: dspy.Prediction) -> int:
    """Simple metric: 1 if correct, 0 if incorrect."""
    gold_answer = _parse_numeric_answer(str(example.answer))
    pred_answer = _parse_numeric_answer(str(prediction.answer))

    if gold_answer is None:
        return 0  # Can't evaluate without gold answer
    if pred_answer is None:
        return 0  # Failed to parse prediction

    return int(gold_answer == pred_answer)


@beartype
def gsm8k_feedback_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
    pred_name=None,
    pred_trace=None,
) -> dspy.Prediction:
    """Metric with feedback for GEPA optimization."""
    gold_answer = _parse_numeric_answer(str(example.answer))
    pred_answer = _parse_numeric_answer(str(prediction.answer))
    solution = getattr(example, "solution", "")

    # Handle parsing failure
    if pred_answer is None:
        feedback = f"Could not parse your answer '{prediction.answer}' as an integer. " f"The correct answer is {gold_answer}."
        if solution:
            feedback += f"\n\nHere's the step-by-step solution:\n{solution}"
        return dspy.Prediction(score=0, feedback=feedback)

    # Score and provide feedback
    score = int(gold_answer == pred_answer)

    if score == 1:
        feedback = f"Correct! The answer is {gold_answer}."
    else:
        feedback = f"Incorrect. You answered {pred_answer}, but the correct answer is {gold_answer}."
        if solution:
            feedback += f"\n\nHere's the step-by-step solution:\n{solution}\n\n" f"Review this solution to improve your approach to similar problems."

    return dspy.Prediction(score=score, feedback=feedback)


if __name__ == "__main__":  # TODO(Adriano) turn this into some sort of library and put it in a script that we use to review/debug datasets and results
    import random
    import sys
    import termios
    import tty

    from tabulate import tabulate

    def wait_for_spacebar():
        """Wait for spacebar press."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while True:
                ch = sys.stdin.read(1)
                if ch == " ":
                    break
                elif ch == "q" or ch == "\x03":  # q or Ctrl+C
                    print("\n\nExiting...")
                    sys.exit(0)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    # Load dataset
    print("Loading GSM8K dataset...")
    splits = get_gsm8k_dataset(n_samples=500, seed=42)
    all_examples = splits["train"] + splits["val"] + splits["test"]

    print(f"Loaded {len(all_examples)} examples. Press SPACE for next, 'q' to quit.\n")

    # Shuffle for random sampling
    random.shuffle(all_examples)

    for i, ex in enumerate(all_examples):
        # Build table with single column
        table_data = [
            [f"EXAMPLE: {i + 1}/{len(all_examples)}"],
            [f"PROBLEM:\n{ex.problem}"],
            [f"ANSWER: {ex.answer}"],
            [f"SOLUTION:\n{ex.solution}"],
            [f"PARSED: {_parse_numeric_answer(ex.answer)}"],
        ]

        print("\033[2J\033[H")  # Clear screen
        print(tabulate(table_data, tablefmt="fancy_grid"))
        print("\n[Press SPACE for next, 'q' to quit]")

        wait_for_spacebar()
