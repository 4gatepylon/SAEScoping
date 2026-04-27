"""Generic job scheduler with dynamic GPU assignment.

Runs a list of shell commands in parallel, assigning GPUs dynamically as
they become free. Each job declares how many GPUs it needs; multi-GPU jobs
consume multiple slots. CPU-only jobs (n_gpus=0) run in a separate pool.

Robustness features:
  - Each subprocess runs in its own session (``start_new_session=True``)
    and is tracked in a registry. SIGINT/SIGTERM/SIGHUP to the launcher
    walks the registry and ``os.killpg``s every descendant subtree, so
    no orphaned CUDA contexts.
  - Per-job state file (``<log_dir>/<safe_name>.state.json``) records
    status, attempts, and a hash of the command + env. Jobs whose state
    is ``done`` and whose hash matches are skipped on resume.
  - Retry with exponential backoff (``JobSpec.max_attempts``).
  - Live tee of stdout/stderr to the terminal with a ``[name]`` prefix
    in addition to per-job log files.
  - Tail computation reads only the last ~64 KB of the log file.
  - ``run_jobs`` returns a non-zero summary; the CLI wrapper exits
    non-zero if any job ultimately failed.

Usage:
    from sae_scoping.infrastructure.scheduler import JobSpec, run_jobs

    jobs = [
        JobSpec(command=["python", "train.py", "--model=big"], n_gpus=2, name="big"),
        JobSpec(command=["python", "train.py", "--model=small"], n_gpus=1, name="small"),
        JobSpec(command=["python", "preprocess.py"], n_gpus=0, name="prep"),
    ]
    results = run_jobs(jobs, gpu_ids=[0, 1, 2, 3], n_cpu_workers=2, log_dir="./logs")

# BUG TODO(adriano) [SEV:LOW]: known remaining limitations —
#   - No live progress bar / ETA across jobs.
#   - No structured "retry only OOM" classification — every non-zero rc
#     retries up to max_attempts. Real fix: pluggable should_retry(rc, stderr).
#   - run_jobs returns the JobResult list; callers must inspect themselves
#     or call _print_summary. No exception is raised.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
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
    max_attempts: int = 1

    def command_hash(self) -> str:
        """Stable hash of command + env, used for resume idempotency."""
        h = hashlib.sha256()
        payload = json.dumps(
            {"cmd": self.command, "env": sorted(self.env.items())},
            sort_keys=True,
        ).encode()
        h.update(payload)
        return h.hexdigest()[:16]


class JobResult(BaseModel):
    """Result of a completed (or skipped) job."""
    name: str
    returncode: int
    elapsed_s: float = 0.0
    gpu_ids: list[int] = Field(default_factory=list)
    stdout_tail: str = ""
    stderr_tail: str = ""
    attempts: int = 1
    skipped: bool = False


class _JobState(BaseModel):
    """Persisted per-job state, used for resume."""
    name: str
    command_hash: str
    status: str = "pending"  # pending | running | done | failed
    attempts: int = 0
    started_at: float | None = None
    finished_at: float | None = None
    returncode: int | None = None


def _safe_name(name: str) -> str:
    return name.replace("/", "--").replace(" ", "_")


def _read_tail(path: Path, n_lines: int, max_bytes: int = 64 * 1024) -> str:
    """Read the last ~max_bytes of a file and return its last n_lines."""
    if not path.exists():
        return ""
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            if size > max_bytes:
                f.seek(-max_bytes, os.SEEK_END)
            chunk = f.read()
    except OSError:
        return ""
    text = chunk.decode("utf-8", errors="replace")
    return "\n".join(text.strip().split("\n")[-n_lines:])


def _state_path(log_dir: Path, safe_name: str) -> Path:
    return log_dir / f"{safe_name}.state.json"


def _load_state(log_dir: Path, safe_name: str) -> _JobState | None:
    p = _state_path(log_dir, safe_name)
    if not p.exists():
        return None
    try:
        return _JobState.model_validate_json(p.read_text())
    except Exception:
        return None


def _save_state(log_dir: Path, safe_name: str, state: _JobState) -> None:
    p = _state_path(log_dir, safe_name)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(state.model_dump_json(indent=2))
    tmp.replace(p)


# ---------------------------------------------------------------------------
# Subprocess registry: tracks Popens for signal-based cleanup
# ---------------------------------------------------------------------------


class _ProcRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._popens: set[subprocess.Popen] = set()

    def add(self, p: subprocess.Popen) -> None:
        with self._lock:
            self._popens.add(p)

    def remove(self, p: subprocess.Popen) -> None:
        with self._lock:
            self._popens.discard(p)

    def kill_all(self, term_timeout: float = 5.0) -> None:
        with self._lock:
            popens = list(self._popens)
        if not popens:
            return
        sys.stderr.write(f"[scheduler] killing {len(popens)} subprocess group(s)\n")
        sys.stderr.flush()
        for p in popens:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        deadline = time.monotonic() + term_timeout
        for p in popens:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                p.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    pass


_REGISTRY = _ProcRegistry()
_HANDLERS_INSTALLED = False
_HANDLERS_LOCK = threading.Lock()


def _install_signal_handlers() -> None:
    """Install SIGINT/SIGTERM/SIGHUP handlers that nuke tracked subprocesses."""
    global _HANDLERS_INSTALLED
    with _HANDLERS_LOCK:
        if _HANDLERS_INSTALLED:
            return
        _HANDLERS_INSTALLED = True

    fired = {"done": False}

    def _handler(signum, _frame):
        if fired["done"]:
            return
        fired["done"] = True
        sys.stderr.write(f"\n[scheduler] caught signal {signum}, cleaning up\n")
        sys.stderr.flush()
        _REGISTRY.kill_all()
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    # Only the main thread can install signal handlers. If we're imported
    # from a thread (which we shouldn't be), skip silently.
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError):
                pass


# ---------------------------------------------------------------------------
# Live output tee
# ---------------------------------------------------------------------------


_PRINT_LOCK = threading.Lock()


def _pump(stream, log_file, console_stream, prefix: str | None) -> None:
    """Read lines from stream, write to log_file (and optionally console)."""
    try:
        for raw in iter(stream.readline, b""):
            try:
                line = raw.decode("utf-8", errors="replace")
            except Exception:
                line = repr(raw) + "\n"
            log_file.write(line)
            log_file.flush()
            if console_stream is not None:
                with _PRINT_LOCK:
                    if prefix:
                        console_stream.write(f"{prefix} {line}")
                    else:
                        console_stream.write(line)
                    console_stream.flush()
    finally:
        try:
            stream.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# GPU pool
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Single-job execution
# ---------------------------------------------------------------------------


def _run_single_attempt(
    job: JobSpec,
    gpu_ids: list[int],
    log_dir: Path | None,
    tail_lines: int,
    live_tee: bool,
) -> tuple[int, str, str, float]:
    """Execute one attempt of a job. Returns (returncode, stdout_tail, stderr_tail, elapsed)."""
    name = job.name or " ".join(job.command[:3])
    safe_name = _safe_name(name)

    env = os.environ.copy()
    env.update(job.env)
    if gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    elif job.n_gpus == 0:
        env["CUDA_VISIBLE_DEVICES"] = ""

    log_stdout_path: Path | None = None
    log_stderr_path: Path | None = None
    log_stdout_f = None
    log_stderr_f = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_stdout_path = log_dir / f"{safe_name}.stdout.log"
        log_stderr_path = log_dir / f"{safe_name}.stderr.log"
        log_stdout_f = open(log_stdout_path, "w")
        log_stderr_f = open(log_stderr_path, "w")

    t0 = time.monotonic()
    proc = subprocess.Popen(
        job.command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        start_new_session=True,
    )
    _REGISTRY.add(proc)

    prefix = f"[{name}]" if live_tee else None
    console_out = sys.stdout if live_tee else None
    console_err = sys.stderr if live_tee else None

    pumps: list[threading.Thread] = []
    if log_stdout_f is not None:
        t_out = threading.Thread(
            target=_pump,
            args=(proc.stdout, log_stdout_f, console_out, prefix),
            daemon=True,
        )
        t_err = threading.Thread(
            target=_pump,
            args=(proc.stderr, log_stderr_f, console_err, prefix),
            daemon=True,
        )
    else:
        # No log file — pump to a string buffer in memory.
        # (Only used when log_dir is None; unusual.)
        import io
        log_stdout_f = io.StringIO()
        log_stderr_f = io.StringIO()
        t_out = threading.Thread(
            target=_pump, args=(proc.stdout, log_stdout_f, console_out, prefix), daemon=True,
        )
        t_err = threading.Thread(
            target=_pump, args=(proc.stderr, log_stderr_f, console_err, prefix), daemon=True,
        )
    t_out.start()
    t_err.start()
    pumps.extend([t_out, t_err])

    try:
        try:
            rc = proc.wait(timeout=job.timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired, OSError):
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
                proc.wait()
            rc = -1
    finally:
        _REGISTRY.remove(proc)
        for t in pumps:
            t.join(timeout=5)
        for f in (log_stdout_f, log_stderr_f):
            try:
                if hasattr(f, "close"):
                    f.close()
            except Exception:
                pass

    elapsed = time.monotonic() - t0

    if log_stdout_path is not None:
        stdout_tail = _read_tail(log_stdout_path, tail_lines)
        stderr_tail = _read_tail(log_stderr_path, tail_lines)
    else:
        # In-memory fallback.
        stdout_text = log_stdout_f.getvalue() if hasattr(log_stdout_f, "getvalue") else ""
        stderr_text = log_stderr_f.getvalue() if hasattr(log_stderr_f, "getvalue") else ""
        stdout_tail = "\n".join(stdout_text.strip().split("\n")[-tail_lines:])
        stderr_tail = "\n".join(stderr_text.strip().split("\n")[-tail_lines:])

    return rc, stdout_tail, stderr_tail, elapsed


def _run_one(
    job: JobSpec,
    gpu_pool: _GPUPool | None,
    log_dir: Path | None,
    tail_lines: int,
    live_tee: bool,
    resume: bool,
    retry_backoff_s: float,
) -> JobResult:
    name = job.name or " ".join(job.command[:3])
    safe_name = _safe_name(name)
    cmd_hash = job.command_hash()

    # --- Resume check ---
    if resume and log_dir is not None:
        prior = _load_state(log_dir, safe_name)
        if prior is not None and prior.status == "done" and prior.command_hash == cmd_hash:
            stdout_tail = ""
            stderr_tail = ""
            if log_dir is not None:
                stdout_tail = _read_tail(log_dir / f"{safe_name}.stdout.log", tail_lines)
                stderr_tail = _read_tail(log_dir / f"{safe_name}.stderr.log", tail_lines)
            print(f"[SKIP] {name} (already done)")
            return JobResult(
                name=name,
                returncode=prior.returncode or 0,
                elapsed_s=0.0,
                gpu_ids=[],
                stdout_tail=stdout_tail,
                stderr_tail=stderr_tail,
                attempts=prior.attempts,
                skipped=True,
            )

    gpu_ids: list[int] = []
    if job.n_gpus > 0 and gpu_pool is not None:
        gpu_ids = gpu_pool.acquire(job.n_gpus)

    state = _JobState(name=name, command_hash=cmd_hash, status="running")
    if log_dir is not None:
        _save_state(log_dir, safe_name, state)

    try:
        gpu_tag = f" [GPU {','.join(str(g) for g in gpu_ids)}]" if gpu_ids else " [CPU]"
        last_rc = 1
        last_stdout = ""
        last_stderr = ""
        total_elapsed = 0.0
        attempt = 0
        for attempt in range(1, max(1, job.max_attempts) + 1):
            print(f"[START]{gpu_tag} {name} (attempt {attempt}/{job.max_attempts})")
            state.attempts = attempt
            state.started_at = time.time()
            if log_dir is not None:
                _save_state(log_dir, safe_name, state)

            rc, stdout_tail, stderr_tail, elapsed = _run_single_attempt(
                job, gpu_ids, log_dir, tail_lines, live_tee,
            )
            total_elapsed += elapsed
            last_rc = rc
            last_stdout = stdout_tail
            last_stderr = stderr_tail

            status = "DONE" if rc == 0 else "FAIL"
            print(f"[{status}]{gpu_tag} {name} ({elapsed:.1f}s, rc={rc}, attempt {attempt})")
            if rc == 0:
                break
            if attempt < job.max_attempts:
                wait = retry_backoff_s * (2 ** (attempt - 1))
                print(f"  retrying in {wait:.1f}s...")
                time.sleep(wait)

        if rc != 0 and last_stderr:
            for line in last_stderr.split("\n")[-3:]:
                print(f"  {line}")

        state.status = "done" if last_rc == 0 else "failed"
        state.returncode = last_rc
        state.finished_at = time.time()
        if log_dir is not None:
            _save_state(log_dir, safe_name, state)

        return JobResult(
            name=name,
            returncode=last_rc,
            elapsed_s=total_elapsed,
            gpu_ids=gpu_ids,
            stdout_tail=last_stdout,
            stderr_tail=last_stderr,
            attempts=attempt,
        )
    finally:
        if gpu_ids and gpu_pool is not None:
            gpu_pool.release(gpu_ids)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class Scheduler:
    """Run jobs in parallel with dynamic GPU assignment.

    Args:
        gpu_ids: Available GPU device IDs (e.g. [0, 1, 2, 7]).
        n_cpu_workers: Max concurrent CPU-only (n_gpus=0) jobs.
        log_dir: If set, stream stdout/stderr to per-job log files and
            write per-job state files for resume.
        tail_lines: Number of output lines to keep in JobResult.
        live_tee: If True, also print stdout/stderr to the terminal with
            a ``[name]`` prefix (in addition to writing to log files).
        resume: If True, skip jobs whose state file says ``status=done``
            and whose command_hash still matches.
        retry_backoff_s: Initial backoff between retries; doubles each attempt.
    """

    def __init__(
        self,
        gpu_ids: list[int] | None = None,
        n_cpu_workers: int = 2,
        log_dir: Path | None = None,
        tail_lines: int = 20,
        live_tee: bool = False,
        resume: bool = True,
        retry_backoff_s: float = 5.0,
    ):
        self.gpu_ids = gpu_ids or []
        self.n_cpu_workers = n_cpu_workers
        self.log_dir = log_dir
        self.tail_lines = tail_lines
        self.live_tee = live_tee
        self.resume = resume
        self.retry_backoff_s = retry_backoff_s

    def run(self, jobs: list[JobSpec]) -> list[JobResult]:
        """Run all jobs, blocking until complete."""
        if not jobs:
            return []

        _install_signal_handlers()

        gpu_jobs = [j for j in jobs if j.n_gpus > 0]
        cpu_jobs = [j for j in jobs if j.n_gpus == 0]

        for j in gpu_jobs:
            if j.n_gpus > len(self.gpu_ids):
                raise ValueError(
                    f"Job {j.name!r} needs {j.n_gpus} GPUs but only "
                    f"{len(self.gpu_ids)} available: {self.gpu_ids}"
                )

        gpu_pool = _GPUPool(self.gpu_ids) if self.gpu_ids else None
        results: list[JobResult] = []
        lock = threading.Lock()

        def _submit(job: JobSpec) -> None:
            r = _run_one(
                job, gpu_pool, self.log_dir, self.tail_lines,
                self.live_tee, self.resume, self.retry_backoff_s,
            )
            with lock:
                results.append(r)

        cpu_sem = threading.Semaphore(self.n_cpu_workers)

        def _submit_cpu(job: JobSpec) -> None:
            cpu_sem.acquire()
            try:
                _submit(job)
            finally:
                cpu_sem.release()

        threads: list[threading.Thread] = []
        for job in gpu_jobs:
            t = threading.Thread(target=_submit, args=(job,), daemon=True)
            threads.append(t)
        for job in cpu_jobs:
            t = threading.Thread(target=_submit_cpu, args=(job,), daemon=True)
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        return results

    def run_dry(self, jobs: list[JobSpec]) -> None:
        gpu_jobs = [j for j in jobs if j.n_gpus > 0]
        cpu_jobs = [j for j in jobs if j.n_gpus == 0]
        print(f"Scheduler: {len(jobs)} jobs ({len(gpu_jobs)} GPU, {len(cpu_jobs)} CPU)")
        print(f"  GPUs: {self.gpu_ids}  |  CPU workers: {self.n_cpu_workers}")
        for i, j in enumerate(jobs):
            name = j.name or " ".join(j.command[:3])
            tag = f"{j.n_gpus} GPU" if j.n_gpus > 0 else "CPU"
            skip_note = ""
            if self.resume and self.log_dir is not None:
                prior = _load_state(self.log_dir, _safe_name(name))
                if prior is not None and prior.status == "done" and prior.command_hash == j.command_hash():
                    skip_note = "  [WOULD SKIP — already done]"
            print(f"  [{i + 1}] ({tag}) {name}{skip_note}: {' '.join(j.command)}")


def _print_summary(results: list[JobResult]) -> None:
    passed = [r for r in results if r.returncode == 0]
    failed = [r for r in results if r.returncode != 0]
    skipped = [r for r in results if r.skipped]
    total_time = sum(r.elapsed_s for r in results)
    wall_time = max((r.elapsed_s for r in results), default=0.0)

    print(f"\n{'=' * 70}")
    print(
        f"  {len(passed)} passed, {len(failed)} failed, {len(skipped)} skipped  "
        f"({total_time:.1f}s total, {wall_time:.1f}s wall)"
    )
    print(f"{'=' * 70}")
    for r in results:
        if r.skipped:
            status = "SKIP"
        elif r.returncode == 0:
            status = "PASS"
        else:
            status = "FAIL"
        gpu_str = f"GPU {','.join(str(g) for g in r.gpu_ids)}" if r.gpu_ids else "CPU"
        print(f"  [{status}] {r.name:40s}  {r.elapsed_s:7.1f}s  attempts={r.attempts}  {gpu_str}")
    if failed:
        print(f"\nFailed jobs:")
        for r in failed:
            print(f"  {r.name} (rc={r.returncode}, attempts={r.attempts})")
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
    live_tee: bool = False,
    resume: bool = True,
    retry_backoff_s: float = 5.0,
) -> list[JobResult]:
    """Run a list of jobs in parallel with dynamic GPU scheduling.

    Args:
        jobs: List of JobSpec to execute.
        gpu_ids: Available GPU device IDs (e.g. [0, 1, 2, 7]).
        n_cpu_workers: Max concurrent CPU-only (n_gpus=0) jobs.
        log_dir: Directory for per-job stdout/stderr/state files.
        tail_lines: Number of output lines kept in each JobResult.
        dry_run: If True, print the job list (with skip annotations) and return.
        live_tee: If True, mirror per-job output to the terminal with a
            ``[name]`` prefix.
        resume: If True (default), skip jobs whose state file says
            ``status=done`` and whose command hash still matches.
        retry_backoff_s: Initial backoff between retries (doubles each attempt).

    Returns:
        List of JobResult in completion order.
    """
    log_path = Path(log_dir)
    scheduler = Scheduler(
        gpu_ids=gpu_ids or [],
        n_cpu_workers=n_cpu_workers,
        log_dir=log_path,
        tail_lines=tail_lines,
        live_tee=live_tee,
        resume=resume,
        retry_backoff_s=retry_backoff_s,
    )

    gpu_jobs = [j for j in jobs if j.n_gpus > 0]
    cpu_jobs = [j for j in jobs if j.n_gpus == 0]
    print(f"{'=' * 70}")
    print(f"  {len(jobs)} jobs ({len(gpu_jobs)} GPU, {len(cpu_jobs)} CPU)")
    print(f"  GPUs: {gpu_ids or '(none)'}  |  CPU workers: {n_cpu_workers}")
    print(f"  Logs: {log_path.resolve()}")
    print(f"  resume={resume}  live_tee={live_tee}")
    print(f"{'=' * 70}")

    if dry_run:
        scheduler.run_dry(jobs)
        return []

    results = scheduler.run(jobs)
    _print_summary(results)
    return results
