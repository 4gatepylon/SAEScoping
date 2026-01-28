"""
Step 7: Print sizes of each merged dataset.

- Loads {subject}_merged.jsonl files from input directory
- Displays sample counts in a table format, broken down by dataset_source
- Helps inform decisions for train/test/validation split sizes
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import click
from tabulate import tabulate


VALID_SUBJECTS = ["biology", "chemistry", "math", "physics"]
VALID_SOURCES = ["megascience", "camel-ai", "numina"]
DEFAULT_INPUT_DIR = Path(__file__).parent


def count_by_source(path: Path) -> Counter[str]:
    """Count samples by dataset_source field."""
    counts: Counter[str] = Counter()
    with open(path) as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                source = sample.get("dataset_source", "unknown")
                counts[source] += 1
    return counts


@click.command()
@click.option(
    "--input-dir",
    "-i",
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_INPUT_DIR,
    help="Directory containing {subject}_merged.jsonl files",
)
def main(input_dir: Path) -> None:
    """Print sizes of each merged dataset."""
    table_data = []
    total_counts: Counter[str] = Counter()

    for subject in VALID_SUBJECTS:
        path = input_dir / f"{subject}_merged.jsonl"
        if path.exists():
            counts = count_by_source(path)
            total_counts += counts
            row = [subject] + [counts.get(src, 0) for src in VALID_SOURCES] + [sum(counts.values())]
            table_data.append(row)
        else:
            row = [subject] + ["(not found)"] * (len(VALID_SOURCES) + 1)
            table_data.append(row)

    # Add total row
    table_data.append(["---"] * (len(VALID_SOURCES) + 2))
    total_row = ["TOTAL"] + [total_counts.get(src, 0) for src in VALID_SOURCES] + [sum(total_counts.values())]
    table_data.append(total_row)

    headers = ["Subject"] + [f"Count ({src})" for src in VALID_SOURCES] + ["Count (total)"]
    print(f"\nDataset sizes (from {input_dir})")
    print("=" * 80)
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))


if __name__ == "__main__":
    main()
