"""
Step 3: Visualize other_information tags from samples classified as "other".

- Loads megascience_classifications.jsonl
- Filters to samples where class == "other"
- Normalizes tags (.lower().strip())
- Builds frequency map of all tags
- Supports sampling by tag for inspection
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

import click
from tabulate import tabulate


DEFAULT_CLASSIFICATIONS_PATH = Path(__file__).parent / "megascience_classifications.jsonl"


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


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file."""
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def get_other_samples(classifications: list[dict]) -> list[dict]:
    """Filter to samples classified as 'other'."""
    other_samples = []
    for item in classifications:
        classification = item.get("classification", {})
        if classification.get("class") == "other":
            other_samples.append(item)
    return other_samples


def normalize_tag(tag: str) -> str:
    """Normalize a tag: lowercase and strip whitespace."""
    return tag.lower().strip()


def build_tag_frequency_map(other_samples: list[dict]) -> Counter:
    """Build frequency map of all normalized tags across all 'other' samples."""
    tag_counter: Counter[str] = Counter()

    for sample in other_samples:
        classification = sample.get("classification", {})
        other_info = classification.get("other_information") or []

        if isinstance(other_info, list):
            for tag in other_info:
                if isinstance(tag, str) and tag.strip():
                    normalized = normalize_tag(tag)
                    tag_counter[normalized] += 1

    return tag_counter


def show_frequency_map(tag_counter: Counter, total_samples: int) -> None:
    """Display the frequency map of tags."""
    table_data = []
    for tag, count in tag_counter.most_common():
        pct = 100.0 * count / total_samples if total_samples > 0 else 0
        table_data.append([tag, count, f"{pct:.2f}%"])

    print(f"\nOther Information Tag Frequency Map")
    print(f"(n={total_samples} samples classified as 'other')")
    print("=" * 60)
    print(tabulate(table_data, headers=["Tag", "Count", "% of Samples"], tablefmt="fancy_grid"))
    print(f"\nTotal unique tags: {len(tag_counter)}")
    print(f"Total tag occurrences: {sum(tag_counter.values())}")


def sample_by_tag(other_samples: list[dict], tag: str, n_samples: int, seed: int | None) -> None:
    """Sample and display random entries containing a given tag."""
    # Normalize the search tag
    search_tag = normalize_tag(tag)

    # Filter samples containing the tag
    filtered = []
    for sample in other_samples:
        classification = sample.get("classification", {})
        other_info = classification.get("other_information") or []

        if isinstance(other_info, list):
            normalized_tags = [normalize_tag(t) for t in other_info if isinstance(t, str)]
            if search_tag in normalized_tags:
                filtered.append(sample)

    if not filtered:
        print(f"No samples found containing tag: {tag}")
        return

    # Shuffle
    if seed is not None:
        random.seed(seed)
    random.shuffle(filtered)

    # Limit to n_samples
    filtered = filtered[:n_samples]

    print(f"\nSampling {len(filtered)} entries containing tag: '{tag}'")
    print("Press SPACE for next, 'q' to quit.\n")

    for idx, sample in enumerate(filtered):
        classification = sample.get("classification", {})
        question = sample.get("question", "")
        answer = sample.get("answer", "")
        original_subject = sample.get("original_subject", "")
        other_info = classification.get("other_information") or []
        explanation = classification.get("explanation", "")
        sample_index = sample.get("index", "?")

        # Truncate long text for display
        max_len = 500
        if len(question) > max_len:
            question = question[:max_len] + "..."
        if len(answer) > max_len:
            answer = answer[:max_len] + "..."
        if explanation and len(explanation) > max_len:
            explanation = explanation[:max_len] + "..."

        # Format other_info as comma-separated list
        other_info_str = ", ".join(other_info) if other_info else "<NONE>"

        table_data = [
            ["Index", f"{idx + 1}/{len(filtered)} (dataset idx: {sample_index})"],
            ["Original Subject", original_subject or "<EMPTY>"],
            ["Other Information", other_info_str],
            ["LLM Explanation", explanation or "<NONE>"],
            ["Question", question],
            ["Answer", answer],
        ]

        print("\033[2J\033[H")  # Clear screen
        print(tabulate(table_data, tablefmt="fancy_grid", maxcolwidths=[20, 80]))
        print("\n[Press SPACE for next, 'q' to quit]")

        key = get_key()
        if key == "q" or key == "\x03":  # q or Ctrl+C
            print("\n\nExiting...")
            return


def sample_all_other(other_samples: list[dict], n_samples: int, seed: int | None) -> None:
    """Sample and display random 'other' entries (any tag)."""
    if not other_samples:
        print("No samples classified as 'other'.")
        return

    # Shuffle
    filtered = list(other_samples)
    if seed is not None:
        random.seed(seed)
    random.shuffle(filtered)

    # Limit to n_samples
    filtered = filtered[:n_samples]

    print(f"\nSampling {len(filtered)} entries classified as 'other'")
    print("Press SPACE for next, 'q' to quit.\n")

    for idx, sample in enumerate(filtered):
        classification = sample.get("classification", {})
        question = sample.get("question", "")
        answer = sample.get("answer", "")
        original_subject = sample.get("original_subject", "")
        other_info = classification.get("other_information") or []
        explanation = classification.get("explanation", "")
        sample_index = sample.get("index", "?")

        # Truncate long text for display
        max_len = 500
        if len(question) > max_len:
            question = question[:max_len] + "..."
        if len(answer) > max_len:
            answer = answer[:max_len] + "..."
        if explanation and len(explanation) > max_len:
            explanation = explanation[:max_len] + "..."

        # Format other_info as comma-separated list
        other_info_str = ", ".join(other_info) if other_info else "<NONE>"

        table_data = [
            ["Index", f"{idx + 1}/{len(filtered)} (dataset idx: {sample_index})"],
            ["Original Subject", original_subject or "<EMPTY>"],
            ["Other Information", other_info_str],
            ["LLM Explanation", explanation or "<NONE>"],
            ["Question", question],
            ["Answer", answer],
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
    "--classifications",
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_CLASSIFICATIONS_PATH,
    help="Path to megascience_classifications.jsonl",
)
@click.option(
    "--show-frequency/--no-show-frequency",
    default=True,
    help="Show frequency map of other_information tags",
)
@click.option(
    "--sample-tag",
    type=str,
    default=None,
    help="Tag to sample from (case-insensitive, will be normalized)",
)
@click.option(
    "--sample-all",
    is_flag=True,
    default=False,
    help="Sample from all 'other' samples regardless of tags",
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
def main(
    classifications: Path,
    show_frequency: bool,
    sample_tag: str | None,
    sample_all: bool,
    n_samples: int,
    seed: int | None,
) -> None:
    """Visualize other_information tags from samples classified as 'other'.

    This helps identify tags that should be mapped to subjects in
    other_information_to_subject.json.
    """
    print(f"Loading classifications from {classifications}...")
    all_classifications = load_jsonl(classifications)
    print(f"Loaded {len(all_classifications)} classifications.")

    other_samples = get_other_samples(all_classifications)
    print(f"Found {len(other_samples)} samples classified as 'other'.")

    if not other_samples:
        print("No samples classified as 'other'. Nothing to visualize.")
        return

    if show_frequency:
        tag_counter = build_tag_frequency_map(other_samples)
        show_frequency_map(tag_counter, len(other_samples))

    if sample_tag is not None:
        sample_by_tag(other_samples, sample_tag, n_samples, seed)
    elif sample_all:
        sample_all_other(other_samples, n_samples, seed)


if __name__ == "__main__":
    main()
