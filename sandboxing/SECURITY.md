# cc-sandbox Security Model

## What the agent CAN do

| Capability | How | Risk level |
|------------|-----|------------|
| Read/write/delete any file in the clone | Clone mounted read-write at `/workspace` | **Low** — isolated clone, not your main repo |
| Run arbitrary shell commands in container | `--dangerously-skip-permissions` | **Low** — container isolation limits blast radius |
| Access NVIDIA GPUs | `--runtime=nvidia` + `CUDA_VISIBLE_DEVICES` | **Low** — can consume GPU compute but not damage hardware |
| Push to git remote (origin) | SSH keys mounted read-only | **Medium** — can push branches; see mitigations below |
| Access network (pip install, curl, etc.) | Full network access (no firewall) | **Medium** — can reach any endpoint; see mitigations below |
| Read Claude OAuth credentials | `~/.claude` mounted read-only | **Medium** — can `cat` the token and exfiltrate it over the network |
| Read SSH private keys | `~/.ssh` mounted read-only | **High** — can `cat ~/.ssh/id_rsa` and exfiltrate; read-only does NOT prevent reading |
| Read API keys from .env | Passed as env vars | **Medium** — has your OpenAI, HF, WANDB keys etc. |

## What the agent CANNOT do

| Restriction | How enforced |
|-------------|--------------|
| Modify your main repo working tree | Works on a separate `git clone`, not the original |
| Modify Claude credentials | `~/.claude` mounted **read-only** |
| Modify SSH keys | `~/.ssh` mounted **read-only** |
| Affect other containers | Docker isolation (separate namespaces) |
| Persist after container exit | `--rm` flag deletes container on exit |
| Access host filesystem outside mounts | Docker filesystem isolation |
| Run privileged kernel operations | Not running with `--privileged` |
| Escalate to root | No sudo available in container |
| Exhaust host memory/CPU | Resource limits enforced via `--memory` and `--pids-limit` |
| Force-push to main/master | Only enforced by GitHub branch protection (**you must enable this!**) |

## Known security gaps

These are risks we are **aware of** and have chosen to accept. Review them and decide if they're acceptable for your use case.

### GAP 1: Credential exfiltration via network
**Severity**: High
**What**: SSH keys, Claude OAuth token, and API keys are all readable inside the container. Combined with full network access, the agent could exfiltrate them to any server.
**Why we accept it**: Agent needs SSH for git push, Claude auth to run, API keys for ML work, and network for pip/git.
**Mitigation**: Use GitHub deploy keys (scoped to one repo) instead of personal SSH keys. Set billing alerts on all API services. Rotate keys if you suspect misuse. Consider adding the [official firewall](https://github.com/anthropics/claude-code/blob/main/.devcontainer/init-firewall.sh) to allowlist only necessary domains.

### GAP 2: Branch push escape
**Severity**: Medium
**What**: Post-commit hook pushes whatever branch HEAD is on. If Claude runs `git checkout main && git commit`, it pushes to main.
**Why we accept it**: Enforcing branch names in the hook adds complexity and the agent could just remove the hook anyway (it has write access to `.git/hooks/`).
**Mitigation**: **Enable GitHub branch protection on main.** This is server-side and cannot be bypassed by the agent.

### GAP 3: GPU memory snooping
**Severity**: Low
**What**: With `--runtime=nvidia`, the container can access GPU memory of other processes on the same physical GPU.
**Mitigation**: Use `CUDA_VISIBLE_DEVICES` to assign dedicated GPUs per agent. Don't share GPUs with sensitive workloads.

## Accepted risks and mitigations

### 1. Network exfiltration
**Risk**: Agent could send data (code, API keys) to external servers.
**Mitigation**: This is the trade-off for full network access (needed for pip install, git push). If this concerns you:
- Use GitHub branch protection to limit push damage
- Rotate API keys periodically
- Monitor container network traffic with `docker stats`
- Consider adding the firewall from the [official devcontainer](https://github.com/anthropics/claude-code/blob/main/.devcontainer/init-firewall.sh)

### 2. Git push of bad code
**Risk**: Agent pushes broken/malicious code to a branch.
**Mitigation**:
- Agent pushes to `sandbox/*` branches, never directly to `main`
- Enable GitHub branch protection on `main`
- Review branches before merging (normal PR workflow)

### 3. API key abuse
**Risk**: Agent uses your API keys (OpenAI, HF, WANDB) for unintended purposes.
**Mitigation**: Keys are needed for the agent's legitimate ML work. Set billing limits on each service.

### 4. SSH key exposure
**Risk**: Read-only mount does NOT prevent the agent from reading your private key content and sending it over the network.
**Mitigation**: **Use a GitHub deploy key** scoped to this single repo instead of your personal SSH key. This limits blast radius to one repo instead of all your GitHub/server access.

### 5. Claude credential reuse
**Risk**: Agent could read and exfiltrate your Claude OAuth token.
**Mitigation**: Token is scoped to Claude Code operations. Mount is read-only (can't modify, but CAN read).

## Recommendations

1. **Enable GitHub branch protection** on `main` — most important single mitigation
2. **Use deploy keys** instead of personal SSH keys — limits blast radius to one repo
3. **Set billing alerts** on all API services (Anthropic, OpenAI, WANDB, HF)
4. **Review branches** before merging — treat sandbox branches like external PRs
5. **Rotate API keys** if you suspect any were misused
