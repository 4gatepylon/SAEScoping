#!/usr/bin/env python3
"""
Test script to load and display all messages datasets in canonical format.

Run with: python -m sae_scoping.datasets.messages_datasets.test_loaders

Modes:
  --interactive / -i : Interactive mode with keyboard navigation
  (default)          : Print a few samples from each dataset

Interactive controls:
  ↑/↓ or Space  : Previous/next entry in current dataset
  ←/→           : Previous/next dataset
  q             : Quit
"""

from __future__ import annotations

import sys
import tty
import termios

import click

from sae_scoping.datasets.messages_datasets.schemas import (
    MessagesDataset,
    MessagesEntry,
)


def format_messages_entry(entry: MessagesEntry, max_content_len: int = 200) -> str:
    """Format a messages entry for display."""
    lines = []
    for i, msg in enumerate(entry.messages):
        content = msg.content
        if len(content) > max_content_len:
            content = content[:max_content_len] + "..."
        # Indent continuation lines
        content_lines = content.split("\n")
        if len(content_lines) > 3:
            content = "\n".join(content_lines[:3]) + "\n..."
        lines.append(f"[{msg.role.upper()}]: {content}")
    if entry.metadata:
        lines.append(f"Metadata: {entry.metadata}")
    return "\n\n".join(lines)


def get_keypress() -> str:
    """Get a single keypress, handling arrow keys."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        # Handle escape sequences (arrow keys)
        if ch == "\x1b":
            ch2 = sys.stdin.read(1)
            if ch2 == "[":
                ch3 = sys.stdin.read(1)
                if ch3 == "A":
                    return "up"
                elif ch3 == "B":
                    return "down"
                elif ch3 == "C":
                    return "right"
                elif ch3 == "D":
                    return "left"
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def clear_screen():
    """Clear the terminal screen."""
    print("\033[2J\033[H", end="")


def load_all_datasets(limit: int | None) -> list[tuple[str, MessagesDataset]]:
    """Load all datasets and return as list of (name, dataset) tuples."""
    from sae_scoping.datasets.messages_datasets import (
        load_ultrachat,
        load_stemqa_math,
        load_stemqa_biology,
        load_stemqa_chemistry,
        load_stemqa_physics,
    )

    datasets: list[tuple[str, MessagesDataset]] = []

    print("Loading datasets...")

    print("  Loading UltraChat...", end=" ", flush=True)
    try:
        datasets.append(("ultrachat", load_ultrachat(limit=limit)))
        print(f"✓ ({datasets[-1][1].info.size} entries)")
    except Exception as e:
        print(f"✗ ({e})")

    print("  Loading StemQA Math...", end=" ", flush=True)
    try:
        datasets.append(("stemqa_math", load_stemqa_math(limit=limit)))
        print(f"✓ ({datasets[-1][1].info.size} entries)")
    except Exception as e:
        print(f"✗ ({e})")

    print("  Loading StemQA Biology...", end=" ", flush=True)
    try:
        datasets.append(("stemqa_biology", load_stemqa_biology(limit=limit)))
        print(f"✓ ({datasets[-1][1].info.size} entries)")
    except Exception as e:
        print(f"✗ ({e})")

    print("  Loading StemQA Chemistry...", end=" ", flush=True)
    try:
        datasets.append(("stemqa_chemistry", load_stemqa_chemistry(limit=limit)))
        print(f"✓ ({datasets[-1][1].info.size} entries)")
    except Exception as e:
        print(f"✗ ({e})")

    print("  Loading StemQA Physics...", end=" ", flush=True)
    try:
        datasets.append(("stemqa_physics", load_stemqa_physics(limit=limit)))
        print(f"✓ ({datasets[-1][1].info.size} entries)")
    except Exception as e:
        print(f"✗ ({e})")

    return datasets


def interactive_browser(datasets: list[tuple[str, MessagesDataset]]) -> None:
    """Interactive browser for datasets with keyboard navigation."""
    if not datasets:
        print("No datasets loaded!")
        return

    dataset_idx = 0
    entry_idx = 0

    while True:
        clear_screen()

        # Get current dataset and entry
        name, dataset = datasets[dataset_idx]
        entries = dataset.entries
        n_entries = len(entries)

        # Clamp entry_idx
        entry_idx = max(0, min(entry_idx, n_entries - 1))

        # Header: dataset navigation bar
        dataset_names = [d[0] for d in datasets]
        header_parts = []
        for i, dname in enumerate(dataset_names):
            if i == dataset_idx:
                header_parts.append(f"[{dname.upper()}]")
            else:
                header_parts.append(dname)
        header = "  ←  " + "  |  ".join(header_parts) + "  →"

        print("=" * 80)
        print(header)
        print("=" * 80)
        print(f"Dataset: {dataset.info.source}", end="")
        if dataset.info.subset:
            print(f" ({dataset.info.subset})", end="")
        if dataset.info.split:
            print(f" [{dataset.info.split}]", end="")
        print()
        print(f"Entry: {entry_idx + 1} / {n_entries}")
        print("=" * 80)
        print()

        # Display current entry
        if n_entries > 0:
            entry = entries[entry_idx]
            print(format_messages_entry(entry))
        else:
            print("(no entries)")

        print()
        print("=" * 80)
        print("Controls: ↑/↓/Space = prev/next entry | ←/→ = prev/next dataset | q = quit")
        print("=" * 80)

        # Get input
        key = get_keypress()

        if key == "q" or key == "\x03":  # q or Ctrl+C
            clear_screen()
            print("Goodbye!")
            break
        elif key == "up":
            entry_idx = max(0, entry_idx - 1)
        elif key == "down" or key == " ":
            entry_idx = min(n_entries - 1, entry_idx + 1)
        elif key == "left":
            dataset_idx = (dataset_idx - 1) % len(datasets)
            entry_idx = 0  # Reset to first entry
        elif key == "right":
            dataset_idx = (dataset_idx + 1) % len(datasets)
            entry_idx = 0  # Reset to first entry


def print_samples(datasets: list[tuple[str, MessagesDataset]], n_samples: int) -> None:
    """Print a few samples from each dataset (non-interactive mode)."""
    for name, dataset in datasets:
        print(f"\n{'=' * 80}")
        print(f"DATASET: {name}")
        print(f"Source: {dataset.info.source}")
        if dataset.info.subset:
            print(f"Subset: {dataset.info.subset}")
        if dataset.info.split:
            print(f"Split: {dataset.info.split}")
        print(f"Total entries: {dataset.info.size}")
        if dataset.info.extra:
            print(f"Extra: {dataset.info.extra}")
        print("=" * 80)

        for i, entry in enumerate(dataset.entries[:n_samples]):
            print(f"\n--- Entry {i + 1}/{min(n_samples, len(dataset.entries))} ---")
            print(format_messages_entry(entry))

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    for name, dataset in datasets:
        print(f"  {name}: {dataset.info.size} entries")


@click.command()
@click.option("--interactive", "-i", is_flag=True, help="Interactive mode with keyboard navigation")
@click.option("--limit", "-l", type=int, default=100, help="Limit samples per dataset for loading")
@click.option("--show", "-s", type=int, default=2, help="Number of samples to display (non-interactive)")
def main(interactive: bool, limit: int, show: int) -> None:
    """Load and display messages datasets to verify they work correctly."""
    datasets = load_all_datasets(limit=limit)

    if not datasets:
        print("No datasets loaded successfully!")
        return

    if interactive:
        print("\nStarting interactive browser...")
        print("Press any key to continue...")
        get_keypress()
        interactive_browser(datasets)
    else:
        print_samples(datasets, n_samples=show)


if __name__ == "__main__":
    main()
