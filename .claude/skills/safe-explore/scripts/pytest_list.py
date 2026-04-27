#!/usr/bin/env python3
"""List pytest tests without running them. Passes extra args to --collect-only."""

import subprocess
import sys

PYTHON = "/opt/miniconda3/envs/saescoping/bin/python"


def main():
    cmd = [PYTHON, "-m", "pytest", "--collect-only", "-q"] + sys.argv[1:]
    result = subprocess.run(cmd, capture_output=False)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
