---
name: safe-explore
description: Safe read-only codebase exploration tools. Use these scripts instead of raw find, pytest, or cross-branch git inspection to stay within the permission allowlist.
user-invocable: false
---

## Available scripts

All scripts live in `${CLAUDE_SKILL_DIR}/scripts/` and are invoked via `python <script>`.

| Script | Purpose | Example |
|---|---|---|
| `sfind.py` | Safe `find` — blocks `-exec`, `-delete`, `-execdir`, `-ok`, `-okdir`, `-fls`, `-fprint*` | `python .claude/skills/safe-explore/scripts/sfind.py . -name "*.py" -type f` |
| `branch_summary.py` | Commits, stat diff, and file tree for a branch vs base | `python .claude/skills/safe-explore/scripts/branch_summary.py origin/adriano/baselines` |
| `file_on_branch.py` | Read a single file from another branch without checkout | `python .claude/skills/safe-explore/scripts/file_on_branch.py origin/cais README.md` |
| `module_map.py` | AST-based map of a Python package (classes, functions, exports) | `python .claude/skills/safe-explore/scripts/module_map.py sae_scoping/training` |
| `pytest_list.py` | List tests via `--collect-only` without running them | `python .claude/skills/safe-explore/scripts/pytest_list.py sae_scoping/` |
| `branch_diff_files.py` | Files added/modified/deleted on a branch vs base | `python .claude/skills/safe-explore/scripts/branch_diff_files.py origin/adriano/sae_pruning` |

## When to use

Prefer these over raw shell commands when exploring the codebase autonomously.
They are allowlisted in `.claude/settings.local.json` so they run without permission prompts.
