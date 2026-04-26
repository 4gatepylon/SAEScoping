#!/usr/bin/env python3
"""Read a file from any git branch without checking it out."""

import subprocess
import sys


def main():
    if len(sys.argv) < 3:
        print("Usage: file_on_branch.py <branch> <filepath>", file=sys.stderr)
        sys.exit(1)

    branch = sys.argv[1]
    filepath = sys.argv[2]

    result = subprocess.run(
        ["git", "show", f"{branch}:{filepath}"],
        capture_output=False,
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
