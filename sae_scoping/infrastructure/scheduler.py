"""Generic job scheduler with dynamic GPU assignment.

Runs a list of shell commands in parallel, assigning GPUs dynamically as
they become free. Each job declares how many GPUs it needs; multi-GPU jobs
consume multiple slots. CPU-only jobs (n_gpus=0) run in a separate pool.

Usage:
    from sae_scoping.infrastructure.scheduler import JobSpec, run_jobs

    jobs = [
        JobSpec(command=["python", "train.py", "--model=big"], n_gpus=2, name="big-model"),
        JobSpec(command=["python", "train.py", "--model=small"], n_gpus=1, name="small-model"),
        JobSpec(command=["python", "preprocess.py"], n_gpus=0, name="preprocess"),
    ]
    results = run_jobs(jobs, gpu_ids=[0, 1, 2, 3], n_cpu_workers=2, log_dir="./logs")
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class JobSpec(BaseModel):
    """A command to run with resource requirements."""
    command: list[str]
    n_gpus: int = 1
    name: str = ""
    timeout: float | None = 7200
    env: dict[str, str] = Field(default_factory=dict)


class JobResult(BaseModel):
    """Result of a completed job."""
    name: str
    returncode: int
    elapsed_s: float = 0.0
    gpu_ids: list[int] = Field(default_factory=list)
    stdout_tail: str = ""
    stderr_tail: str = ""


class _GPUPool:
    """Thread-safe pool of GPU IDs with dynamic allocation."""

    def __init__(self, gpu_ids: list[int]):
        self._free: list[int] = list(gpu_ids)
        self._cond = threading.Condition()

    def acquire(self, n: int) -> list[int]:
        with self._cond:
            while len(self._free) < n:
                self._cond.wait()
            claimed = self._free[:n]
            del self._free[:n]
            return claimed

    def release(self, ids: list[int]) -> None:
        with self._cond:
            self._free.extend(ids)
            self._cond.notify_all()

    @property
    def total(self) -> int:
        with self._cond:
            return len(self._free)


def _run_one(
    job: JobSpec,
    gpu_pool: _GPUPool | None,
    log_dir: Path | None,
    tail_lines: int,
) -> JobResult:
    name = job.name or " ".join(job.command[:3])
    gpu_ids: list[int] = []

    if job.n_gpus > 0 and gpu_pool is not None:
        gpu_ids = gpu_pool.acquire(job.n_gpus)

    try:
        env = os.environ.copy()
        env.update(job.env)
        if gpu_ids:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
        elif job.n_gpus == 0:
            env["CUDA_VISIBLE_DEVICES"] = ""

        gpu_tag = f" [GPU {','.join(str(g) for g in gpu_ids)}]" if gpu_ids else " [CPU]"
        print(f"[START]{gpu_tag} {name}")

        log_stdout = None
        log_stderr = None
        safe_name = name.replace("/", "--").replace(" ", "_")
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_stdout = open(log_dir / f"{safe_name}.stdout.log", "w")
            log_stderr = open(log_dir / f"{safe_name}.stderr.log", "w")

        t0 = time.monotonic()
        try:
            if log_dir is not None:
                result = subprocess.run(
                    job.command, env=env,
                    stdout=log_stdout, stderr=log_stderr,
                    text=True, timeout=job.timeout,
                )
            else:
                result = subprocess.run(
                    job.command, env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, timeout=job.timeout,
                )
            elapsed = time.monotonic() - t0

            if log_dir is not None:
                log_stdout.close()
                log_stderr.close()
                stdout_text = (log_dir / f"{safe_name}.stdout.log").read_text()
                stderr_text = (log_dir / f"{safe_name}.stderr.log").read_text()
            else:
                stdout_text = result.stdout or ""
                stderr_text = result.stderr or ""

            stdout_tail = "\n".join(stdout_text.strip().split("\n")[-tail_lines:])
            stderr_tail = "\n".join(stderr_text.strip().split("\n")[-tail_lines:])

            status = "DONE" if result.returncode == 0 else "FAIL"
            print(f"[{status}]{gpu_tag} {name} ({elapsed:.1f}s, rc={result.returncode})")
            if result.returncode != 0 and stderr_tail:
                for line in stderr_tail.split("\n")[-3:]:
                    print(f"  {line}")

            return JobResult(
                name=name,
                returncode=result.returncode,
                elapsed_s=elapsed,
                gpu_ids=gpu_ids,
                stdout_tail=stdout_tail,
                stderr_tail=stderr_tail,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - t0
            print(f"[TIMEOUT]{gpu_tag} {name} ({elapsed:.1f}s)")
            return JobResult(
                name=name, returncode=-1, elapsed_s=elapsed,
                gpu_ids=gpu_ids, stderr_tail="TIMEOUT",
            )
        finally:
            if log_dir is not None:
                if not log_stdout.closed:
                    log_stdout.close()
                if not log_stderr.closed:
                    log_stderr.close()
    finally:
        if gpu_ids and gpu_pool is not None:
            gpu_pool.release(gpu_ids)


class Scheduler:
    """Run jobs in parallel with dynamic GPU assignment.

    Args:
        gpu_ids: Available GPU device IDs (e.g. [0, 1, 2, 7]).
        n_cpu_workers: Max concurrent CPU-only (n_gpus=0) jobs.
        log_dir: If set, stream stdout/stderr to per-job log files.
        tail_lines: Number of output lines to keep in JobResult.
    """

    def __init__(
        self,
        gpu_ids: list[int] | None = None,
        n_cpu_workers: int = 2,
        log_dir: Path | None = None,
        tail_lines: int = 20,
    ):
        self.gpu_ids = gpu_ids or []
        self.n_cpu_workers = n_cpu_workers
        self.log_dir = log_dir
        self.tail_lines = tail_lines

    def run(self, jobs: list[JobSpec]) -> list[JobResult]:
        """Run all jobs, blocking until complete. Returns results in completion order."""
        if not jobs:
            return []

        gpu_jobs = [j for j in jobs if j.n_gpus > 0]
        cpu_jobs = [j for j in jobs if j.n_gpus == 0]

        for j in gpu_jobs:
            if j.n_gpus > len(self.gpu_ids):
                raise ValueError(
                    f"Job {j.name!r} needs {j.n_gpus} GPUs but only "
                    f"{len(self.gpu_ids)} available: {self.gpu_ids}"
                )

        gpu_pool = _GPUPool(self.gpu_ids) if self.gpu_ids else None
        # Enough threads so GPU jobs can block on acquire without starving CPU jobs.
        n_workers = len(self.gpu_ids) + self.n_cpu_workers
        results: list[JobResult] = []
        lock = threading.Lock()

        def _submit(job: JobSpec) -> None:
            r = _run_one(job, gpu_pool, self.log_dir, self.tail_lines)
            with lock:
                results.append(r)

        threads: list[threading.Thread] = []
        # Use a semaphore to cap CPU concurrency separately from GPU.
        cpu_sem = threading.Semaphore(self.n_cpu_workers)

        def _submit_cpu(job: JobSpec) -> None:
            cpu_sem.acquire()
            try:
                _submit(job)
            finally:
                cpu_sem.release()

        for job in gpu_jobs:
            t = threading.Thread(target=_submit, args=(job,), daemon=True)
            threads.append(t)
        for job in cpu_jobs:
            t = threading.Thread(target=_submit_cpu, args=(job,), daemon=True)
            threads.append(t)

        # Start all threads. GPU jobs will block on gpu_pool.acquire(),
        # giving natural backpressure. CPU jobs block on cpu_sem.
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        return results

    def run_dry(self, jobs: list[JobSpec]) -> None:
        """Print what would be run without executing anything."""
        gpu_jobs = [j for j in jobs if j.n_gpus > 0]
        cpu_jobs = [j for j in jobs if j.n_gpus == 0]
        print(f"Scheduler: {len(jobs)} jobs ({len(gpu_jobs)} GPU, {len(cpu_jobs)} CPU)")
        print(f"  GPUs: {self.gpu_ids}  |  CPU workers: {self.n_cpu_workers}")
        for i, j in enumerate(jobs):
            name = j.name or " ".join(j.command[:3])
            tag = f"{j.n_gpus} GPU" if j.n_gpus > 0 else "CPU"
            print(f"  [{i + 1}] ({tag}) {name}: {' '.join(j.command)}")


def _print_summary(results: list[JobResult]) -> None:
    passed = [r for r in results if r.returncode == 0]
    failed = [r for r in results if r.returncode != 0]
    total_time = sum(r.elapsed_s for r in results)
    wall_time = max(r.elapsed_s for r in results) if results else 0.0

    print(f"\n{'=' * 70}")
    print(f"  {len(passed)} passed, {len(failed)} failed  "
          f"({total_time:.1f}s total CPU time, {wall_time:.1f}s wall)")
    print(f"{'=' * 70}")
    for r in results:
        status = "PASS" if r.returncode == 0 else "FAIL"
        gpu_str = f"GPU {','.join(str(g) for g in r.gpu_ids)}" if r.gpu_ids else "CPU"
        print(f"  [{status}] {r.name:40s}  {r.elapsed_s:7.1f}s  {gpu_str}")
    if failed:
        print(f"\nFailed jobs:")
        for r in failed:
            print(f"  {r.name} (rc={r.returncode})")
            if r.stderr_tail:
                for line in r.stderr_tail.strip().split("\n")[-3:]:
                    print(f"    {line}")
    print()


def run_jobs(
    jobs: list[JobSpec],
    gpu_ids: list[int] | None = None,
    n_cpu_workers: int = 2,
    log_dir: str | Path = "./job_logs",
    tail_lines: int = 20,
    dry_run: bool = False,
) -> list[JobResult]:
    """Run a list of jobs in parallel with dynamic GPU scheduling.

    This is the main entry point. It creates a Scheduler, runs all jobs,
    prints a summary table, and returns the results.

    Args:
        jobs: List of JobSpec to execute.
        gpu_ids: Available GPU device IDs (e.g. [0, 1, 2, 7]).
        n_cpu_workers: Max concurrent CPU-only (n_gpus=0) jobs.
        log_dir: Directory for per-job stdout/stderr log files.
        tail_lines: Number of output lines kept in each JobResult.
        dry_run: If True, print the job list without running anything.

    Returns:
        List of JobResult in completion order.
    """
    log_path = Path(log_dir)
    scheduler = Scheduler(
        gpu_ids=gpu_ids or [],
        n_cpu_workers=n_cpu_workers,
        log_dir=log_path,
        tail_lines=tail_lines,
    )

    gpu_jobs = [j for j in jobs if j.n_gpus > 0]
    cpu_jobs = [j for j in jobs if j.n_gpus == 0]
    print(f"{'=' * 70}")
    print(f"  {len(jobs)} jobs ({len(gpu_jobs)} GPU, {len(cpu_jobs)} CPU)")
    print(f"  GPUs: {gpu_ids or '(none)'}  |  CPU workers: {n_cpu_workers}")
    print(f"  Logs: {log_path.resolve()}")
    print(f"{'=' * 70}")

    if dry_run:
        scheduler.run_dry(jobs)
        return []

    results = scheduler.run(jobs)
    _print_summary(results)
    return results
