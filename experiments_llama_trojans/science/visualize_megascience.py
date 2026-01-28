"""
Step 1: Visualize MegaScience dataset subjects.

- Shows a frequency map of the `subject` field values
- Allows sampling random entries by subject for inspection
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

import click
from datasets import load_dataset
from tabulate import tabulate


DEFAULT_SUBJECT_MAPPING_PATH = Path(__file__).parent / "subject_mapping_megascience_subject_to_our_subject.json"


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


def show_frequency_map(dataset) -> Counter:
    """Count and display frequency of subject values."""
    subjects = []
    for sample in dataset:
        subject = sample.get("subject")
        if subject is None or (isinstance(subject, str) and subject.strip() == ""):
            subjects.append("<EMPTY>")
        else:
            subjects.append(subject)

    counter = Counter(subjects)

    # Build table sorted by frequency (descending)
    table_data = []
    for subject, count in counter.most_common():
        pct = 100.0 * count / len(subjects)
        table_data.append([subject, count, f"{pct:.2f}%"])

    print(f"\nMegaScience Subject Frequency Map (n={len(subjects)})")
    print("=" * 60)
    print(tabulate(table_data, headers=["Subject", "Count", "Percentage"], tablefmt="fancy_grid"))
    print(f"\nTotal unique subjects: {len(counter)}")
    print(f"Total samples: {len(subjects)}")

    return counter


def show_mapped_subject_counts(dataset, subject_mapping: dict[str, str]) -> None:
    """Show counts after applying subject mapping."""
    mapped_counts: Counter[str] = Counter()
    unmapped_count = 0

    for sample in dataset:
        subject = sample.get("subject")
        if subject is None or (isinstance(subject, str) and subject.strip() == ""):
            unmapped_count += 1
        elif subject in subject_mapping:
            mapped_counts[subject_mapping[subject]] += 1
        else:
            unmapped_count += 1

    print(f"\n\nMapped Subject Counts (using JSON mapping)")
    print("=" * 60)

    table_data = []
    total_mapped = sum(mapped_counts.values())
    for subject in ["biology", "chemistry", "physics", "math"]:
        count = mapped_counts.get(subject, 0)
        pct = 100.0 * count / len(dataset) if len(dataset) > 0 else 0
        table_data.append([subject, count, f"{pct:.2f}%"])

    table_data.append(["<UNMAPPED>", unmapped_count, f"{100.0 * unmapped_count / len(dataset):.2f}%"])

    print(tabulate(table_data, headers=["Our Subject", "Count", "Percentage"], tablefmt="fancy_grid"))
    print(f"\nTotal mapped: {total_mapped}")
    print(f"Total unmapped (need LLM classification): {unmapped_count}")


def sample_by_subject(dataset, subject: str | None, n_samples: int, seed: int | None):
    """Sample and display random entries from a given subject."""
    # Filter dataset by subject
    filtered = []
    for i, sample in enumerate(dataset):
        sample_subject = sample.get("subject")
        if subject == "<EMPTY>":
            if sample_subject is None or (isinstance(sample_subject, str) and sample_subject.strip() == ""):
                filtered.append((i, sample))
        elif subject is None:
            # Sample from all
            filtered.append((i, sample))
        else:
            if sample_subject == subject:
                filtered.append((i, sample))

    if not filtered:
        print(f"No samples found for subject: {subject}")
        return

    # Shuffle
    if seed is not None:
        random.seed(seed)
    random.shuffle(filtered)

    # Limit to n_samples
    filtered = filtered[:n_samples]

    print(f"\nSampling {len(filtered)} entries from subject: {subject or 'ALL'}")
    print("Press SPACE for next, 'q' to quit.\n")

    for idx, (i, sample) in enumerate(filtered):
        # Build table with one row per field
        question = sample.get("question", "")
        answer = sample.get("answer", "")
        reference_answer = sample.get("reference_answer", "")
        sample_subject = sample.get("subject", "")

        # Truncate long text for display
        max_len = 500
        if len(question) > max_len:
            question = question[:max_len] + "..."
        if len(answer) > max_len:
            answer = answer[:max_len] + "..."
        if reference_answer and len(reference_answer) > max_len:
            reference_answer = reference_answer[:max_len] + "..."

        table_data = [
            ["Index", f"{idx + 1}/{len(filtered)} (dataset idx: {i})"],
            ["Subject", sample_subject or "<EMPTY>"],
            ["Question", question],
            ["Answer", answer],
            ["Reference Answer", reference_answer or "<NONE>"],
        ]

        print("\033[2J\033[H")  # Clear screen
        print(tabulate(table_data, tablefmt="fancy_grid", maxcolwidths=[20, 80]))
        print("\n[Press SPACE for next, 'q' to quit]")

        key = get_key()
        if key == "q" or key == "\x03":  # q or Ctrl+C
            print("\n\nExiting...")
            return


@click.command()
@click.option(
    "--show-frequency/--no-show-frequency",
    default=True,
    help="Show frequency map of subjects",
)
@click.option(
    "--sample-subject",
    type=str,
    default=None,
    help="Subject to sample from (use '<EMPTY>' for entries without subject, omit for all)",
)
@click.option(
    "--n-samples",
    type=int,
    default=10,
    help="Number of samples to display",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for sampling",
)
@click.option(
    "--sample-all",
    is_flag=True,
    default=False,
    help="Sample from all subjects (ignores --sample-subject)",
)
@click.option(
    "--subject-mapping",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="JSON file mapping MegaScience subjects to our subjects. If provided, shows mapped counts.",
)
def main(
    show_frequency: bool,
    sample_subject: str | None,
    n_samples: int,
    seed: int | None,
    sample_all: bool,
    subject_mapping: Path | None,
):
    """Visualize MegaScience dataset subjects and sample entries."""
    print("Loading MegaScience dataset...")
    dataset = load_dataset("MegaScience/MegaScience", split="train")
    print(f"Loaded {len(dataset)} samples.")

    if show_frequency:
        show_frequency_map(dataset)

    if subject_mapping is not None:
        mapping_path = subject_mapping
    else:
        mapping_path = DEFAULT_SUBJECT_MAPPING_PATH

    if mapping_path.exists():
        with open(mapping_path) as f:
            mapping = json.load(f)
        show_mapped_subject_counts(dataset, mapping)

    if sample_all:
        sample_subject = None

    if sample_subject is not None or sample_all:
        sample_by_subject(dataset, sample_subject, n_samples, seed)


if __name__ == "__main__":
    main()
