"""
Step 5: Interactive viewer for merged datasets.

- Loads {subject}_merged.jsonl based on --subject flag
- Displays one sample at a time as a table (one row per field)
- Press spacebar to advance to next sample
- Supports --n-samples to limit how many samples to view
- Supports --shuffle to randomize sample order
- Supports --length-quantiles to show character length distributions
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Literal

import click
import numpy as np
from tabulate import tabulate


SubjectType = Literal["biology", "chemistry", "math", "physics"]
VALID_SUBJECTS: list[SubjectType] = ["biology", "chemistry", "math", "physics"]

DEFAULT_INPUT_DIR = Path(__file__).parent


def get_key():
    """Get a single keypress (Unix only)."""
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file."""
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def truncate_text(text: str | None, max_len: int = 800) -> str:
    """Truncate text for display, preserving readability."""
    if text is None:
        return "<NONE>"
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def format_metadata(metadata: dict[str, Any] | None) -> str:
    """Format metadata dict for display."""
    if metadata is None:
        return "<NONE>"
    return json.dumps(metadata, indent=2)


def compute_length_quantiles(
    datasets: dict[str, list[dict[str, Any]]],
    quantile_step: float,
) -> None:
    """Compute and display character length quantiles for questions and answers."""
    quantiles = np.arange(quantile_step, 1.0 + quantile_step / 2, quantile_step)
    quantiles = np.clip(quantiles, 0, 1)  # Ensure we don't exceed 1.0

    # Build header
    header = ["Dataset", "Field"] + [f"{int(q * 100)}%" for q in quantiles]

    # Build rows
    rows = []
    for dataset_name, samples in sorted(datasets.items()):
        question_lengths = [len(s.get("question") or "") for s in samples]
        answer_lengths = [len(s.get("answer") or "") for s in samples]

        if question_lengths:
            q_vals = np.quantile(question_lengths, quantiles)
            rows.append([dataset_name, "question"] + [int(v) for v in q_vals])

        if answer_lengths:
            a_vals = np.quantile(answer_lengths, quantiles)
            rows.append([dataset_name, "answer"] + [int(v) for v in a_vals])

    print("\nCharacter Length Quantiles:")
    print(tabulate(rows, headers=header, tablefmt="simple"))


def display_sample(sample: dict[str, Any], idx: int, total: int) -> None:
    """Display a single sample as a table."""
    # Build table with one row per field
    table_data = [
        ["Sample", f"{idx + 1}/{total}"],
        ["Subject", sample.get("subject", "<NONE>")],
        ["Dataset Source", sample.get("dataset_source", "<NONE>")],
        ["Answer Source", sample.get("answer_source", "<NONE>")],
        ["Topic", sample.get("topic") or "<NONE>"],
        ["Subtopic", sample.get("subtopic") or "<NONE>"],
        ["Question", truncate_text(sample.get("question"))],
        ["Answer", truncate_text(sample.get("answer"))],
        ["Reference Answer", truncate_text(sample.get("reference_answer"))],
        ["Ref Answer Source", sample.get("reference_answer_source") or "<NONE>"],
        ["Metadata", format_metadata(sample.get("metadata"))],
    ]

    print("\033[2J\033[H")  # Clear screen
    print(tabulate(table_data, tablefmt="fancy_grid", maxcolwidths=[20, 100]))
    print("\n[Press SPACE for next, 'q' to quit]")


def interactive_viewer(
    samples: list[dict[str, Any]],
    n_samples: int | None,
    shuffle: bool,
    seed: int | None,
) -> None:
    """Interactive viewer for samples."""
    if shuffle:
        if seed is not None:
            random.seed(seed)
        random.shuffle(samples)

    if n_samples is not None:
        samples = samples[:n_samples]

    if not samples:
        print("No samples to display.")
        return

    print(f"Displaying {len(samples)} samples. Press SPACE for next, 'q' to quit.\n")

    for idx, sample in enumerate(samples):
        display_sample(sample, idx, len(samples))

        key = get_key()
        if key == "q" or key == "\x03":  # q or Ctrl+C
            print("\n\nExiting...")
            return

    print("\033[2J\033[H")  # Clear screen
    print(f"Finished viewing all {len(samples)} samples.")


@click.command()
@click.option(
    "--subject",
    "-s",
    type=click.Choice(VALID_SUBJECTS),
    multiple=True,
    help="Subject dataset(s) to view (can specify multiple)",
)
@click.option(
    "--input-dir",
    "-i",
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_INPUT_DIR,
    help="Directory containing merged JSONL files",
)
@click.option(
    "--n-samples",
    "-n",
    type=int,
    default=None,
    help="Maximum number of samples to view (default: all)",
)
@click.option(
    "--shuffle/--no-shuffle",
    default=True,
    help="Shuffle samples before viewing (default: shuffle)",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for shuffling",
)
@click.option(
    "--filter-source",
    type=click.Choice(["camel-ai", "megascience", "numina"]),
    default=None,
    help="Filter to only show samples from this dataset source",
)
@click.option(
    "--filter-answer-source",
    type=str,
    default=None,
    help="Filter to only show samples with this answer source (e.g., 'gpt-4', 'megascience')",
)
@click.option(
    "--stats-only",
    is_flag=True,
    default=False,
    help="Only show statistics, don't launch interactive viewer",
)
@click.option(
    "--length-quantiles",
    is_flag=True,
    default=False,
    help="Show character length quantiles for questions and answers",
)
@click.option(
    "--quantile-step",
    type=float,
    default=0.1,
    help="Step size for quantiles (default: 0.1 for 10%%, 20%%, etc.)",
)
def main(
    subject: tuple[SubjectType, ...],
    input_dir: Path,
    n_samples: int | None,
    shuffle: bool,
    seed: int | None,
    filter_source: str | None,
    filter_answer_source: str | None,
    stats_only: bool,
    length_quantiles: bool,
    quantile_step: float,
) -> None:
    """Interactive viewer for merged science QA datasets."""
    subjects = subject if subject else VALID_SUBJECTS

    # Load all requested datasets
    all_datasets: dict[str, list[dict[str, Any]]] = {}
    for subj in subjects:
        dataset_path = input_dir / f"{subj}_merged.jsonl"
        if not dataset_path.exists():
            print(f"Warning: Dataset file not found: {dataset_path}")
            continue

        print(f"Loading {dataset_path}...")
        samples = load_jsonl(dataset_path)

        # Apply filters
        if filter_source:
            samples = [s for s in samples if s.get("dataset_source") == filter_source]

        if filter_answer_source:
            samples = [s for s in samples if s.get("answer_source") == filter_answer_source]

        all_datasets[subj] = samples
        print(f"  Loaded {len(samples)} samples for {subj}")

    if not all_datasets:
        print("Error: No datasets found.")
        sys.exit(1)

    # Length quantiles mode
    if length_quantiles:
        compute_length_quantiles(all_datasets, quantile_step)
        return

    # For interactive viewing / stats, use first subject only
    if len(subjects) > 1:
        print("\nNote: Interactive viewer uses first subject only. Use --length-quantiles for multi-dataset analysis.")

    subj = subjects[0]
    samples = all_datasets.get(subj, [])

    # Show statistics
    print(f"\n{'=' * 60}")
    print(f"Statistics for {subj}")
    print(f"{'=' * 60}")

    # Count by dataset source
    source_counts: dict[str, int] = {}
    for s in samples:
        src = s.get("dataset_source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    print("\nBy dataset source:")
    for src, count in sorted(source_counts.items()):
        pct = 100.0 * count / len(samples) if samples else 0
        print(f"  {src}: {count} ({pct:.1f}%)")

    # Count by answer source
    answer_counts: dict[str, int] = {}
    for s in samples:
        src = s.get("answer_source", "unknown")
        answer_counts[src] = answer_counts.get(src, 0) + 1

    print("\nBy answer source:")
    for src, count in sorted(answer_counts.items()):
        pct = 100.0 * count / len(samples) if samples else 0
        print(f"  {src}: {count} ({pct:.1f}%)")

    # Count with topic/subtopic
    with_topic = sum(1 for s in samples if s.get("topic"))
    with_subtopic = sum(1 for s in samples if s.get("subtopic"))
    with_ref_answer = sum(1 for s in samples if s.get("reference_answer"))
    with_metadata = sum(1 for s in samples if s.get("metadata"))

    print(f"\nField coverage:")
    print(f"  With topic: {with_topic} ({100.0 * with_topic / len(samples) if samples else 0:.1f}%)")
    print(f"  With subtopic: {with_subtopic} ({100.0 * with_subtopic / len(samples) if samples else 0:.1f}%)")
    print(f"  With reference_answer: {with_ref_answer} ({100.0 * with_ref_answer / len(samples) if samples else 0:.1f}%)")
    print(f"  With metadata: {with_metadata} ({100.0 * with_metadata / len(samples) if samples else 0:.1f}%)")

    print(f"\nTotal samples: {len(samples)}")
    print(f"{'=' * 60}\n")

    if stats_only:
        return

    if not samples:
        print("No samples to display after filtering.")
        return

    # Launch interactive viewer
    interactive_viewer(samples, n_samples, shuffle, seed)


if __name__ == "__main__":
    main()
