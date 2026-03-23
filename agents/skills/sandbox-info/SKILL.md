---
name: sandbox-info
description: Explains the sandboxing setup for running Claude Code agents in isolated Docker containers. Use this skill whenever the user asks about sandboxing, running in a sandbox, how to launch an agent safely, the security model, container isolation, or how to use cc_sandbox.sh. Also trigger when the user mentions "sandbox", "container", "isolation", "safe agent", or asks how to run Claude autonomously.
---

# Sandbox Info

When the user asks about sandboxing, explain the setup and point them to the right files.

## What to tell the user

This project uses Docker containers to isolate long-lived Claude Code agents. Each agent gets:
- Its own git clone + branch (auto-created, auto-pushed on every commit)
- GPU access via NVIDIA runtime
- Read-only access to Claude auth and SSH keys
- Full network access (pip install, git push, API calls)
- An autonomy system prompt so Claude works without asking questions

The container cannot modify the host filesystem, credentials, or other containers. It is deleted on exit.

## Quick start to share with the user

```bash
cd sandboxing/
./build.sh                                    # one-time image build
python validate_setup.py                      # check prerequisites
./cc_sandbox.sh --cuda-visible-devices 0 --prompt "Your task"
```

Add `--screen` to run in a detached screen session. Add `--no-push` to skip auto-pushing commits.

## Security posture (brief)

- **Isolated**: Docker filesystem + process isolation. Agent works on a throwaway clone, not the real repo.
- **Credentials read-only**: `~/.claude` and `~/.ssh` mounted read-only. Agent can use them in-memory but cannot modify or persist changes to them.
- **Network open**: Full network access — this is the main accepted risk, needed for git push, pip install, API calls.
- **Branch-scoped git**: Agent works on `sandbox/*` branches, never main directly. GitHub branch protection on main is strongly recommended.

For the full threat model, read `sandboxing/SECURITY.md` — it has a table of everything the agent can and cannot do, accepted risks, and mitigation recommendations.

## Files to read for details

| File | What it covers |
|------|---------------|
| `sandboxing/SECURITY.md` | Full threat model: can/cannot do, risks, mitigations |
| `sandboxing/CLAUDE.md` | Architecture, directory layout, coding guidelines |
| `sandboxing/cc_sandbox.sh` | CLI entry point (run with `--help` for all flags) |
| `sandboxing/config.yaml` | Default configuration values |
| `sandboxing/entrypoint.sh` | What happens inside the container on startup |
| `sandboxing/Dockerfile` | Container image definition (based on official Claude Code devcontainer) |

Read `sandboxing/SECURITY.md` first if the user is asking about security or risks.
