# cc-sandbox Security Model

## What the agent CAN do

| Capability | How | Risk level |
|------------|-----|------------|
| Read/write/delete any file in the clone | Clone mounted read-write at `/workspace` | **Low** — isolated clone, not your main repo |
| Run arbitrary shell commands in container | `--dangerously-skip-permissions` | **Low** — container isolation limits blast radius |
| Access NVIDIA GPUs | `--runtime=nvidia` + `CUDA_VISIBLE_DEVICES` | **Low** — can consume GPU compute but not damage hardware |
| Push to git remote (origin) | SSH keys mounted read-only | **Medium** — can push branches; see mitigations below |
| Access network (pip install, curl, etc.) | Full network access (no firewall) | **Medium** — can reach any endpoint; see mitigations below |
| Read Claude OAuth credentials | `~/.claude` mounted read-only | **Medium** — can make API calls as you |
| Read SSH private keys | `~/.ssh` mounted read-only | **Medium** — can authenticate to any SSH host you can |
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
| Force-push to main/master | Only enforced by GitHub branch protection (recommended!) |

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
**Risk**: Agent could theoretically use SSH keys to access other servers.
**Mitigation**: Mount is read-only. Agent's system prompt scopes it to repo work only. For higher security, create a deploy key with limited repo access instead of using your personal SSH key.

### 5. Claude credential reuse
**Risk**: Agent could use your Claude OAuth token for other purposes.
**Mitigation**: Token is scoped to Claude Code operations. Mount is read-only.

## Recommendations

1. **Enable GitHub branch protection** on `main` — most important single mitigation
2. **Use deploy keys** instead of personal SSH keys if concerned about lateral movement
3. **Set billing alerts** on all API services (Anthropic, OpenAI, WANDB, HF)
4. **Review branches** before merging — treat sandbox branches like external PRs
5. **Rotate API keys** if you suspect any were misused
