#!/usr/bin/env python3
"""Safe find wrapper — blocks flags that execute, delete, or write."""

import subprocess
import sys

BLOCKED = {
    "-exec", "-execdir", "-delete", "-ok", "-okdir",
    "-fls", "-fprint", "-fprint0", "-fprintf",
}

def main():
    for arg in sys.argv[1:]:
        if arg in BLOCKED:
            print(f"ERROR: '{arg}' is blocked by safe-find", file=sys.stderr)
            sys.exit(1)

    result = subprocess.run(["find"] + sys.argv[1:], capture_output=False)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
