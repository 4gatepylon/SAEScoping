"""Sample and analyze cyber evaluation logs to understand failure patterns.

Usage:
    python script_2026_01_25_sample_cyber_logs.py [folder] [--dataset DATASET] [--n N] [--filter {invalid,correct,incorrect,all}]

By Claude Code.

Examples:
    python script_2026_01_25_sample_cyber_logs.py  # defaults to sft folder, samples 10 from all datasets
    python script_2026_01_25_sample_cyber_logs.py outputs_eval_gemma9b_scoped_cyber_size112_max_tokens1024
    python script_2026_01_25_sample_cyber_logs.py --dataset secqa --n 5 --filter invalid
"""

import argparse
import json
import random
import re
from pathlib import Path


def classify_response(response: str, ground_truth: str) -> dict:
    """Classify a response into categories."""
    response_stripped = response.strip()

    # Check for boxed format
    boxed_match = re.search(r"\\boxed\{([A-D])\}", response)
    has_boxed = boxed_match is not None

    # Check for just letter
    just_letter = re.match(r"^[A-D]$", response_stripped) is not None

    # Check for letter with explanation (e.g., "A) Some text")
    letter_with_text = re.match(r"^[A-D][\)\:\.]", response_stripped) is not None

    # Extract what the model likely intended as answer
    if has_boxed:
        intended_answer = boxed_match.group(1)
    elif just_letter:
        intended_answer = response_stripped
    elif letter_with_text:
        intended_answer = response_stripped[0]
    else:
        # Try to find any letter at the start
        start_letter = re.match(r"^([A-D])", response_stripped)
        intended_answer = start_letter.group(1) if start_letter else None

    would_be_correct = intended_answer == ground_truth if intended_answer else False

    return {
        "has_boxed": has_boxed,
        "just_letter": just_letter,
        "letter_with_text": letter_with_text,
        "intended_answer": intended_answer,
        "would_be_correct": would_be_correct,
    }


def analyze_dataset(items: list) -> dict:
    """Analyze all items in a dataset and return summary statistics."""
    stats = {
        "total": len(items),
        "boxed_format": 0,
        "just_letter": 0,
        "letter_with_text": 0,
        "other": 0,
        "would_be_correct": 0,
        "officially_correct": 0,
        "officially_invalid": 0,
    }

    for item in items:
        classification = classify_response(item["response"], item["ground_truth"])

        if classification["has_boxed"]:
            stats["boxed_format"] += 1
        elif classification["just_letter"]:
            stats["just_letter"] += 1
        elif classification["letter_with_text"]:
            stats["letter_with_text"] += 1
        else:
            stats["other"] += 1

        if classification["would_be_correct"]:
            stats["would_be_correct"] += 1
        if item.get("is_correct"):
            stats["officially_correct"] += 1
        if item.get("is_invalid"):
            stats["officially_invalid"] += 1

    return stats


def print_item(item: dict, idx: int):
    """Print a single item in a readable format."""
    print(f"\n{'='*80}")
    print(f"[{idx}] Ground Truth: {item['ground_truth']}")
    print(f"    is_correct: {item.get('is_correct', 'N/A')}, is_invalid: {item.get('is_invalid', 'N/A')}")

    classification = classify_response(item["response"], item["ground_truth"])
    print(f"    Classification: boxed={classification['has_boxed']}, just_letter={classification['just_letter']}, " f"letter_with_text={classification['letter_with_text']}")
    print(f"    Intended answer: {classification['intended_answer']}, would_be_correct: {classification['would_be_correct']}")

    # Show prompt (user message only, truncated)
    prompt = item.get("prompt", [])
    if prompt:
        user_msg = next((m["content"] for m in prompt if m["role"] == "user"), "")
        if len(user_msg) > 500:
            user_msg = user_msg[:500] + "..."
        print(f"\n    PROMPT (user):\n    {user_msg}")

    # Show response
    response = item["response"]
    print(f"\n    RESPONSE:\n    {repr(response)}")


def main():
    parser = argparse.ArgumentParser(description="Sample and analyze cyber evaluation logs")
    parser.add_argument("folder", nargs="?", default="outputs_eval_gemma9b_sft_cyber_size112_max_tokens1024", help="Folder containing JSON logs (relative to experiments/)")
    parser.add_argument("--dataset", "-d", default="all", help='Dataset to sample from (or "all")')
    parser.add_argument("--n", type=int, default=10, help="Number of samples to show")
    parser.add_argument("--filter", "-f", choices=["invalid", "correct", "incorrect", "all", "format_error"], default="all", help="Filter samples by category")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--summary", "-s", action="store_true", help="Show summary statistics only (no samples)")
    args = parser.parse_args()

    random.seed(args.seed)

    # Find folder
    script_dir = Path(__file__).parent
    folder = script_dir / args.folder
    if not folder.exists():
        print(f"Error: Folder not found: {folder}")
        return 1

    # Load all JSON files
    all_data = {}
    for json_file in folder.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)

        print(f"\n{'#'*80}")
        print(f"# File: {json_file.name}")
        print(f"{'#'*80}")

        if "statistics" in data:
            print("\nOfficial Statistics:")
            for ds_name, stats in data["statistics"].items():
                print(f"  {ds_name}:")
                print(f"    accuracy: {stats['accuracy']:.3f}, correct: {stats['correct']}, " f"invalid: {stats['invalid']}, total: {stats['total']}")

        if "completions" in data:
            for ds_name, items in data["completions"].items():
                all_data[ds_name] = items

                # Analyze this dataset
                analysis = analyze_dataset(items)
                print(f"\n  Analysis for {ds_name}:")
                print(f"    Format distribution:")
                print(f"      - boxed format:     {analysis['boxed_format']:3d} ({100*analysis['boxed_format']/analysis['total']:.1f}%)")
                print(f"      - just letter:      {analysis['just_letter']:3d} ({100*analysis['just_letter']/analysis['total']:.1f}%)")
                print(f"      - letter+text:      {analysis['letter_with_text']:3d} ({100*analysis['letter_with_text']/analysis['total']:.1f}%)")
                print(f"      - other:            {analysis['other']:3d} ({100*analysis['other']/analysis['total']:.1f}%)")
                print(f"    Correctness:")
                print(f"      - officially correct:  {analysis['officially_correct']:3d} ({100*analysis['officially_correct']/analysis['total']:.1f}%)")
                print(f"      - would be correct:    {analysis['would_be_correct']:3d} ({100*analysis['would_be_correct']/analysis['total']:.1f}%)")
                print(f"      - officially invalid:  {analysis['officially_invalid']:3d} ({100*analysis['officially_invalid']/analysis['total']:.1f}%)")

    if args.summary:
        return 0

    # Filter datasets
    if args.dataset != "all":
        matching = {k: v for k, v in all_data.items() if args.dataset.lower() in k.lower()}
        if not matching:
            print(f"\nNo datasets matching '{args.dataset}'. Available: {list(all_data.keys())}")
            return 1
        all_data = matching

    # Collect and filter items
    filtered_items = []
    for ds_name, items in all_data.items():
        for item in items:
            classification = classify_response(item["response"], item["ground_truth"])

            if args.filter == "all":
                filtered_items.append((ds_name, item))
            elif args.filter == "invalid" and item.get("is_invalid"):
                filtered_items.append((ds_name, item))
            elif args.filter == "correct" and item.get("is_correct"):
                filtered_items.append((ds_name, item))
            elif args.filter == "incorrect" and not item.get("is_correct") and not item.get("is_invalid"):
                filtered_items.append((ds_name, item))
            elif args.filter == "format_error" and item.get("is_invalid") and classification["would_be_correct"]:
                # Format error: would be correct but marked invalid
                filtered_items.append((ds_name, item))

    print(f"\n{'#'*80}")
    print(f"# Sampling {args.n} items (filter: {args.filter})")
    print(f"# Total matching items: {len(filtered_items)}")
    print(f"{'#'*80}")

    # Sample
    if len(filtered_items) > args.n:
        sampled = random.sample(filtered_items, args.n)
    else:
        sampled = filtered_items

    for i, (ds_name, item) in enumerate(sampled):
        print(f"\n[Dataset: {ds_name}]")
        print_item(item, i)

    return 0


if __name__ == "__main__":
    exit(main())
