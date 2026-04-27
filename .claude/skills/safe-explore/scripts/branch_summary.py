#!/usr/bin/env python3
"""Summarise a git branch: unique commits, stat diff, file tree."""

import subprocess
import sys


def run(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else f"(failed: {r.stderr.strip()})"


def main():
    if len(sys.argv) < 2:
        print("Usage: branch_summary.py <branch-ref> [base-branch]", file=sys.stderr)
        sys.exit(1)

    branch = sys.argv[1]
    base = sys.argv[2] if len(sys.argv) > 2 else "origin/main"

    print(f"=== Branch: {branch} (vs {base}) ===\n")

    print("--- Unique commits ---")
    print(run(["git", "log", f"{base}..{branch}", "--oneline"]))
    print()

    print("--- Stat diff ---")
    print(run(["git", "diff", f"{base}..{branch}", "--stat"]))
    print()

    print("--- File tree (first 150 entries) ---")
    tree = run(["git", "ls-tree", "-r", "--name-only", branch])
    lines = tree.splitlines()
    print("\n".join(lines[:150]))
    if len(lines) > 150:
        print(f"... ({len(lines) - 150} more files)")


if __name__ == "__main__":
    main()
