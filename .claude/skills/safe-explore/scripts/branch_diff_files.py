#!/usr/bin/env python3
"""List files that differ between two branches, grouped by status (A/M/D)."""

import subprocess
import sys


def run(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def main():
    if len(sys.argv) < 2:
        print("Usage: branch_diff_files.py <branch> [base]", file=sys.stderr)
        sys.exit(1)

    branch = sys.argv[1]
    base = sys.argv[2] if len(sys.argv) > 2 else "origin/main"
    range_spec = f"{base}..{branch}"

    for label, filter_char in [("Added", "A"), ("Modified", "M"), ("Deleted", "D")]:
        files = run(["git", "diff", range_spec, f"--diff-filter={filter_char}", "--name-only"])
        print(f"=== {label} ===")
        print(files if files else "(none)")
        print()


if __name__ == "__main__":
    main()
