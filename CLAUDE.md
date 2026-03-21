# SAEScoping

## Sandbox Policy

**Long-lived Claude agents MUST run inside a sandbox.** If you are a Claude agent and you expect to run for more than a few minutes (e.g. implementing a feature, running experiments, doing a multi-step task), you must be running inside the sandboxed Docker environment defined in `sandboxing/`.

Before proceeding with any long-lived task, confirm with the user:

1. **Are you running in a sandbox?** Check if the environment variable `DEVCONTAINER=true` is set. If it is not set, tell the user: "I am NOT running in a sandbox. Long-lived agents should run via `sandboxing/cc_sandbox.sh` for isolation. Do you want to proceed anyway?"
2. **Does the user want to proceed?** If not sandboxed, wait for explicit confirmation before continuing. Do not assume consent.

Short interactions (answering questions, small edits, code review) do not require a sandbox.

## Project Structure

- `sae_scoping/` — Main Python package (SAE scoping toolkit)
- `experiments/` — Experiment scripts and results
- `sandboxing/` — Infrastructure for running Claude agents in isolated containers (see `sandboxing/CLAUDE.md`)
