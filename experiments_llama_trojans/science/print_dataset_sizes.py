"""
Step 7: Print sizes of each merged dataset.

- Loads {subject}_merged.jsonl files from input directory
- Displays sample counts in a table format
- Helps inform decisions for train/test/validation split sizes
"""

from __future__ import annotations

from pathlib import Path

import click
from tabulate import tabulate


VALID_SUBJECTS = ["biology", "chemistry", "math", "physics"]
DEFAULT_INPUT_DIR = Path(__file__).parent


def count_lines(path: Path) -> int:
    """Count non-empty lines in a file."""
    count = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


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
    total_count = 0

    for subject in VALID_SUBJECTS:
        path = input_dir / f"{subject}_merged.jsonl"
        if path.exists():
            count = count_lines(path)
            table_data.append([subject, count])
            total_count += count
        else:
            table_data.append([subject, "(not found)"])

    # Add total row
    table_data.append(["---", "---"])
    table_data.append(["TOTAL", total_count])

    print(f"\nDataset sizes (from {input_dir})")
    print("=" * 40)
    print(tabulate(table_data, headers=["Subject", "Count"], tablefmt="fancy_grid"))


if __name__ == "__main__":
    main()
