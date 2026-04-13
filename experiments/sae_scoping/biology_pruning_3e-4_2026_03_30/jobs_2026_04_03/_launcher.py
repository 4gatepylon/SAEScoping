"""_launcher.py

Shared utility: round-robin GPU job distribution via threads.

Each GPU gets a dedicated thread that runs its assigned jobs sequentially
(so two jobs sharing one GPU never overlap in memory).  Jobs assigned to
different GPUs run in parallel.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Job:
    cmd: list[str]   # argv passed to subprocess.run
    label: str       # human-readable description for logging


def launch_on_gpus(
    jobs: list[Job],
    gpus: list[int],
    cwd: Path,
    *,
    stop_on_first_failure: bool = False,
) -> int:
    """Distribute `jobs` across `gpus` (round-robin), wait for completion.

    Each GPU spawns exactly one thread; that thread runs its subset of jobs
    sequentially.  Returns the total number of failed jobs.
    """
    if not gpus:
        raise ValueError("gpus must be non-empty")
    if not jobs:
        print("No jobs to run.")
        return 0

    n = len(gpus)
    gpu_jobs: dict[int, list[Job]] = {gpu: [] for gpu in gpus}
    for i, job in enumerate(jobs):
        gpu_jobs[gpus[i % n]].append(job)

    # Print dispatch table
    print(f"Dispatching {len(jobs)} job(s) across {n} GPU(s): {gpus}")
    for i, job in enumerate(jobs):
        print(f"  [{i}] GPU {gpus[i % n]}  —  {job.label}")
    print()

    failures: list[str] = []
    lock = threading.Lock()

    def worker(gpu: int, job_list: list[Job]) -> None:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        for job in job_list:
            print(f"=== [CUDA:{gpu}] {job.label} ===", flush=True)
            result = subprocess.run(job.cmd, env=env, cwd=str(cwd))
            if result.returncode != 0:
                msg = f"FAILED [CUDA:{gpu}] {job.label}  (exit {result.returncode})"
                print(msg, flush=True)
                with lock:
                    failures.append(msg)
                if stop_on_first_failure:
                    break

    threads = [
        threading.Thread(target=worker, args=(gpu, job_list), daemon=True)
        for gpu, job_list in gpu_jobs.items()
        if job_list
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if failures:
        print(f"\n{len(failures)} job(s) failed:", file=sys.stderr)
        for msg in failures:
            print(f"  {msg}", file=sys.stderr)
    else:
        print("\nAll jobs completed successfully.")

    return len(failures)
