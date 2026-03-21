#!/usr/bin/env python3
"""Pre-flight checks for cc-sandbox. Run before first use."""

import os
import shutil
import subprocess
import sys


def check(name: str, ok: bool, fix: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}")
    if not ok and fix:
        print(f"         -> {fix}")
    return ok


def run(cmd: str) -> tuple[int, str]:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def main() -> int:
    print("cc-sandbox pre-flight checks\n")
    all_ok = True

    # Docker
    all_ok &= check("Docker installed", shutil.which("docker") is not None,
                     "Install Docker: https://docs.docker.com/get-docker/")

    # NVIDIA runtime
    rc, out = run("docker info 2>&1")
    has_nvidia = "nvidia" in out.lower()
    all_ok &= check("Docker NVIDIA runtime", has_nvidia,
                     "Install nvidia-container-toolkit")

    # GPUs visible
    rc, out = run("nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null")
    gpu_count = len(out.strip().splitlines()) if rc == 0 and out.strip() else 0
    all_ok &= check(f"GPUs visible ({gpu_count} found)", gpu_count > 0,
                     "Check NVIDIA drivers and nvidia-smi")

    # Docker image built
    rc, _ = run("docker image inspect cc-sandbox:latest >/dev/null 2>&1")
    all_ok &= check("Docker image cc-sandbox:latest", rc == 0,
                     "Run: ./build.sh")

    # SSH keys
    ssh_dir = os.path.expanduser("~/.ssh")
    has_ssh = any(f.startswith("id_") and not f.endswith(".pub")
                  for f in os.listdir(ssh_dir)) if os.path.isdir(ssh_dir) else False
    all_ok &= check("SSH keys exist", has_ssh,
                     "Generate with: ssh-keygen")

    # Claude credentials
    claude_dir = os.path.expanduser("~/.claude")
    has_claude = os.path.exists(os.path.join(claude_dir, ".credentials.json"))
    all_ok &= check("Claude credentials", has_claude,
                     "Run: claude (and authenticate)")

    # .env file
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    all_ok &= check(".env file exists", os.path.isfile(env_path),
                     "Create .env with API keys at repo root")

    # Git remote
    rc, out = run("git -C {} remote get-url origin 2>/dev/null".format(
        os.path.join(os.path.dirname(__file__), "..")))
    all_ok &= check("Git remote 'origin' configured", rc == 0)

    print()
    if all_ok:
        print("All checks passed. Ready to use cc-sandbox.")
    else:
        print("Some checks failed. Fix the issues above before running cc-sandbox.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
