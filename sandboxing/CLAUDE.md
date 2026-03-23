# Sandboxing Infrastructure

This directory contains infrastructure for running Claude Code agents in isolated Docker containers with GPU access. It is **not** part of the main `sae_scoping` project — it's tooling that helps agents work on that project safely.

## Directory layout

```
sandboxing/
├── Dockerfile          # Container image (CUDA + Node.js + Claude Code + Python)
├── build.sh            # Build the Docker image
├── cc_sandbox.sh       # Main CLI entry point
├── entrypoint.sh       # Container entrypoint (git setup, auto-push, Claude launch)
├── auto_responder.sh   # Auto-responds to Claude prompts (placeholder)
├── config.yaml         # Default configuration values
├── validate_setup.py   # Pre-flight checks
├── test_integration.sh # Smoke tests
├── SECURITY.md         # Threat model and accepted risks
└── CLAUDE.md           # This file
```

## Coding guidelines

- **Small commits**: Each commit should do one thing. Commit frequently.
- **Keep it simple**: This is a thin CLI wrapper around Docker + Claude Code. Avoid abstractions.
- **Shell scripts**: Use `set -euo pipefail`. Quote variables. Use arrays for command building.
- **No dead code**: Delete unused code rather than commenting it out.
- **Test changes**: Run `./test_integration.sh` after modifying Dockerfile or scripts.
- **Rebuild image**: After Dockerfile changes, run `./build.sh`.

## How it works

1. `cc_sandbox.sh` clones the repo from origin into `/tmp/cc-sandbox-clones/<branch>`
2. Creates a new git branch
3. Launches a Docker container with the clone mounted at `/workspace`
4. `entrypoint.sh` installs Python deps, sets up git auto-push hook
5. Claude Code runs with `--dangerously-skip-permissions` and an autonomy system prompt
6. Every commit is automatically pushed to origin

## Key constraints

- The Docker image must be rebuilt after Dockerfile changes (`./build.sh`)
- Clones are in `/tmp/` and may be cleaned up by the OS
- SSH keys are mounted read-only — the container cannot modify them
- Claude credentials are mounted read-only
- The container has full network access (see SECURITY.md for threat model)
