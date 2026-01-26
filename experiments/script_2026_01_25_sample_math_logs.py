"""Sample and view math evaluation logs interactively."""

import json
import random
import re
from pathlib import Path

import click

MODEL_TYPES = ["gemma", "sft", "scoped", "recovered"]
DATASETS = ["gsm8k", "numinamath"]

def classify_model(path: str) -> str:
    if "google/gemma-2-9b-it" in path: return "gemma"
    if "ultrachat" in path: return "scoped"
    if "layer_31_width_16k_canonical" in path: return "recovered"
    if "vanilla" in path and re.search(r"checkpoint-\d+", path): return "sft"
    return "unknown"
from tabulate import tabulate


def load_all_completions(results_dir: Path) -> list[dict]:
    """Load all completions from all JSON files with metadata."""
    all_entries = []

    for json_file in results_dir.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)

        config = data["config"]
        stats = data["statistics"]

        for completion in data["completions"]:
            # Extract user content from prompt
            user_content = ""
            for msg in completion["prompt"]:
                if msg["role"] == "user":
                    user_content = msg["content"]
                    break

            entry = {
                "request": user_content,
                "response": completion["response"],
                "ground_truth": completion["ground_truth"],
                "extracted_answer": completion["extracted_answer"],
                "is_correct": completion["is_correct"],
                "is_invalid": completion["is_invalid"],
                "model": config["model"],
                "dataset": stats["dataset"],
                "json_filename": json_file.name,
                "dist_path": config.get("dist_path", "N/A"),
                "threshold": config.get("threshold", "N/A"),
            }
            all_entries.append(entry)

    return all_entries


def wait_for_spacebar():
    """Wait for spacebar press."""
    import sys
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch == " ":
                break
            if ch == "q" or ch == "\x03":  # q or Ctrl+C
                raise KeyboardInterrupt
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def display_entry(entry: dict, idx: int, total: int):
    """Display a single entry using tabulate."""
    rows = [
        [f"REQUEST:\n{entry['request']}"],
        [f"RESPONSE:\n{entry['response']}"],
        [f"Ground Truth: {entry['ground_truth']}"],
        [f"Extracted Answer: {entry['extracted_answer']}"],
        [f"Is Correct: {entry['is_correct']}"],
        [f"Is Invalid: {entry['is_invalid']}"],
        [f"Model: {entry['model']}"],
        [f"Dataset: {entry['dataset']}"],
        [f"JSON File: {entry['json_filename']}"],
        [f"Dist Path: {entry['dist_path']}"],
        [f"Threshold: {entry['threshold']}"],
    ]

    print("\n" + "=" * 100)
    print(f"SAMPLE {idx + 1} / {total}")
    print("=" * 100)
    print(tabulate(rows, tablefmt="grid", maxcolwidths=[100]))
    print("-" * 100)
    print("Press SPACEBAR for next, 'q' to quit")


@click.command()
@click.option("-n", "--n-samples", default=10, help="Number of samples to show")
@click.option("--seed", default=42, help="Random seed for reproducibility")
@click.option("-m", "--model", "models", multiple=True, type=click.Choice(MODEL_TYPES), help="Filter by model type")
@click.option("-d", "--dataset", "datasets", multiple=True, type=click.Choice(DATASETS), help="Filter by dataset")
def main(n_samples: int, seed: int, models: tuple[str, ...], datasets: tuple[str, ...]):
    """Sample and view math evaluation logs interactively."""
    results_dir = Path(__file__).parent / "math_eval_results"

    print(f"Loading completions from {results_dir}...")
    all_entries = load_all_completions(results_dir)
    if models: all_entries = [e for e in all_entries if classify_model(e["model"]) in models]
    if datasets: all_entries = [e for e in all_entries if e["dataset"] in datasets]
    print(f"Loaded {len(all_entries)} completions (filters: models={models or 'all'}, datasets={datasets or 'all'})")

    random.seed(seed)
    samples = random.sample(all_entries, min(n_samples, len(all_entries)))
    print(f"Sampled {len(samples)} entries (seed={seed})")
    print("Press SPACEBAR to start viewing...")

    try:
        wait_for_spacebar()
        for idx, entry in enumerate(samples):
            display_entry(entry, idx, len(samples))
            if idx < len(samples) - 1:
                wait_for_spacebar()
        print("\n\nDone! Viewed all samples.")
    except KeyboardInterrupt:
        print("\n\nExited.")


if __name__ == "__main__":
    main()
