#!/usr/bin/env bash
# Thin shim: forwards all arguments to the Python CLI.
# Install deps: pip install click pyyaml
exec python3 "$(dirname "$0")/cc_sandbox.py" "$@"
