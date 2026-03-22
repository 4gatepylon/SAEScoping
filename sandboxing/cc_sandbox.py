#!/usr/bin/env python3
"""cc-sandbox: Run Claude Code in an isolated Docker container with GPU access."""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from secrets import token_hex

import click
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def load_config() -> dict:
    with open(SCRIPT_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


def git_config(key: str, default: str = "") -> str:
    try:
        return subprocess.check_output(
            ["git", "config", key], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError:
        return default


def load_env_file(path: Path) -> dict[str, str]:
    """Parse KEY=VALUE lines from an env file."""
    env = {}
    if not path.is_file():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def build_docker_cmd(
    *,
    cfg: dict,
    clone_dir: Path,
    session_name: str,
    cuda_devices: str,
    model: str,
    max_budget: float,
    auto_push: bool,
    prompt: str,
) -> list[str]:
    """Build the docker run command from config + overrides."""
    c = cfg["container"]
    g = cfg["git"]

    cmd = [
        "docker", "run",
        "--rm", "-it",
        f"--runtime={c['runtime']}",
        f"--name=cc-sandbox-{session_name}",
        f"--memory={c['memory_limit']}",
        f"--pids-limit={c['pids_limit']}",
    ]

    # GPU
    if cuda_devices:
        cmd += ["-e", f"CUDA_VISIBLE_DEVICES={cuda_devices}"]
        cmd += ["-e", f"NVIDIA_VISIBLE_DEVICES={cuda_devices}"]
    else:
        cmd += ["-e", "NVIDIA_VISIBLE_DEVICES=none"]

    # Mounts
    home = Path.home()
    cmd += [
        "-v", f"{clone_dir}:/workspace",
        "-v", f"{home / '.claude'}:/home/sandbox/.claude:ro",
        "-v", f"{home / '.ssh'}:/home/sandbox/.ssh:ro",
    ]

    # Env from .env file
    env_path = (SCRIPT_DIR / cfg["paths"]["env_file"]).resolve()
    for k, v in load_env_file(env_path).items():
        cmd += ["-e", f"{k}={v}"]

    # Claude / git env
    cmd += [
        "-e", f"CLAUDE_MODEL={model}",
        "-e", f"MAX_BUDGET={max_budget}",
        "-e", f"AUTO_PUSH={'1' if auto_push else '0'}",
        "-e", f"GIT_USER_NAME={git_config('user.name', g['user_name'])}",
        "-e", f"GIT_USER_EMAIL={git_config('user.email', g['user_email'])}",
        "-e", f"CLAUDE_SYSTEM_PROMPT={cfg['claude']['system_prompt']}",
        "-e", f"CLAUDE_PROMPT={prompt}",
    ]

    cmd.append(cfg["image"])
    return cmd


@click.command()
@click.option("--cuda-visible-devices", default="", help="GPU indices (e.g. '0' or '0,1')")
@click.option("--prompt", default=None, help="Task instructions")
@click.option("--prompt-file", default=None, type=click.Path(exists=True), help="Task instructions from file")
@click.option("--model", default=None, help="Claude model (default: from config.yaml)")
@click.option("--max-budget", default=None, type=float, help="Max spend in USD (default: from config.yaml)")
@click.option("--branch", default=None, help="Git branch name (default: sandbox/<timestamp>)")
@click.option("--no-push", is_flag=True, help="Skip auto-push on each commit")
@click.option("--name", "session_name", default=None, help="Session name for identification")
@click.option("--screen", is_flag=True, help="Launch in a detached screen session")
def main(cuda_visible_devices, prompt, prompt_file, model, max_budget, branch, no_push, session_name, screen):
    """Run Claude Code in an isolated Docker container with GPU access."""
    cfg = load_config()

    # Resolve defaults from config
    model = model or cfg["claude"]["model"]
    max_budget = max_budget if max_budget is not None else cfg["claude"]["max_budget_usd"]
    auto_push = cfg["git"]["auto_push"] and not no_push

    # Resolve prompt
    if prompt_file:
        prompt = Path(prompt_file).read_text()
    elif prompt is None:
        click.echo("Enter task instructions (Ctrl+D when done):")
        prompt = sys.stdin.read()

    if not prompt or not prompt.strip():
        click.echo("Error: No prompt provided.", err=True)
        sys.exit(1)

    # Resolve branch
    if not branch:
        branch = f"sandbox/{datetime.now().strftime('%Y%m%d-%H%M%S')}-{token_hex(4)}"

    if not session_name:
        session_name = branch.split("/")[-1]

    # Clone repo
    clones_dir = Path(cfg["paths"]["clones_dir"])
    clone_dir = clones_dir / branch
    remote_url = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "remote", "get-url", cfg["git"]["remote"]],
        text=True,
    ).strip()

    click.echo(f"[cc-sandbox] Cloning {remote_url} ...")
    clone_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", remote_url, str(clone_dir)], check=True)
    subprocess.run(["git", "-C", str(clone_dir), "checkout", "-b", branch], check=True)

    # Build docker command
    docker_cmd = build_docker_cmd(
        cfg=cfg,
        clone_dir=clone_dir,
        session_name=session_name,
        cuda_devices=cuda_visible_devices,
        model=model,
        max_budget=max_budget,
        auto_push=auto_push,
        prompt=prompt,
    )

    click.echo(f"[cc-sandbox] Branch: {branch}")
    click.echo(f"[cc-sandbox] GPU: {cuda_visible_devices or 'none'}")
    click.echo(f"[cc-sandbox] Model: {model} | Budget: ${max_budget}")
    click.echo(f"[cc-sandbox] Clone: {clone_dir}")

    if screen:
        screen_name = f"cc-{session_name}"
        click.echo(f"[cc-sandbox] Screen: {screen_name}")
        click.echo(f"[cc-sandbox] Attach with: screen -r {screen_name}")
        # Write docker command to temp script for screen to execute
        launch_script = f"/tmp/cc-sandbox-launch-{os.getpid()}.sh"
        with open(launch_script, "w") as f:
            f.write("#!/usr/bin/env bash\n")
            # Quote each arg properly
            f.write(" ".join(f"'{a}'" for a in docker_cmd) + "\n")
            f.write("echo '[cc-sandbox] Container exited. Press enter to close.'\n")
            f.write("read\n")
        os.chmod(launch_script, 0o755)
        subprocess.run(["screen", "-dmS", screen_name, "bash", launch_script])
        click.echo(f"[cc-sandbox] Running in background.")
    else:
        click.echo("")
        os.execvp("docker", docker_cmd)


if __name__ == "__main__":
    main()
