"""Tests for the generic job scheduler."""

# TODO(adriano): test-suite review — gaps to consider on future pass.
# Higher-value missing coverage:
#   - job.env vs scheduler CVD precedence: scheduler.py:88-91 does
#     env.update(job.env) then unconditionally overwrites CVD. A job
#     passing env={"CUDA_VISIBLE_DEVICES":"99"} should still get the
#     scheduler's value, not 99. Untested.
#   - n_gpus=0 -> CVD="" assertion: test_gpu_and_cpu_jobs_mixed only
#     checks the "cpu:" prefix, not that the printed CVD is empty.
#   - tail_lines parameter: never tested (e.g. tail_lines=0 behavior).
# Lower-value:
#   - Stress test (many jobs, few GPUs, mixed crashes) for deadlock/leak.
#   - Scheduler instance reuse (calling .run() twice).
#   - JobSpec validation edge cases (negative n_gpus, empty command).
#   - safe_name rewrite for names with "/" or spaces (scheduler.py:100).
# Possible redundancy: test_failing_job is essentially a subset of the
# TestCrashRecovery cases; test_crash_with_output and
# test_crash_stderr_captured could be merged into one assertion.

import sys
import time
from pathlib import Path

import pytest

from sae_scoping.infrastructure.scheduler import JobSpec, Scheduler, run_jobs


class TestCPUJobs:
    """CPU-only jobs — no GPU needed to run these tests."""

    def test_single_job(self):
        jobs = [JobSpec(command=[sys.executable, "-c", "print('hello')"], n_gpus=0, name="hello")]
        results = Scheduler(gpu_ids=[], n_cpu_workers=1).run(jobs)
        assert len(results) == 1
        assert results[0].returncode == 0
        assert "hello" in results[0].stdout_tail

    def test_multiple_cpu_jobs_parallel(self):
        jobs = [
            JobSpec(
                command=[sys.executable, "-c", f"import time; time.sleep(0.2); print('job-{i}')"],
                n_gpus=0, name=f"job-{i}",
            )
            for i in range(4)
        ]
        t0 = time.monotonic()
        results = Scheduler(gpu_ids=[], n_cpu_workers=4).run(jobs)
        elapsed = time.monotonic() - t0
        assert all(r.returncode == 0 for r in results)
        assert len(results) == 4
        # 4 jobs sleeping 0.2s each, all parallel => should finish well under 0.8s
        assert elapsed < 0.8

    def test_cpu_concurrency_limit(self):
        jobs = [
            JobSpec(
                command=[sys.executable, "-c", "import time; time.sleep(0.3)"],
                n_gpus=0, name=f"job-{i}",
            )
            for i in range(4)
        ]
        t0 = time.monotonic()
        results = Scheduler(gpu_ids=[], n_cpu_workers=2).run(jobs)
        elapsed = time.monotonic() - t0
        assert all(r.returncode == 0 for r in results)
        # 4 jobs, 2 at a time, 0.3s each => ~0.6s minimum
        assert elapsed >= 0.5

    def test_failing_job(self):
        jobs = [
            JobSpec(command=[sys.executable, "-c", "raise SystemExit(42)"], n_gpus=0, name="fail"),
        ]
        results = Scheduler(gpu_ids=[], n_cpu_workers=1).run(jobs)
        assert results[0].returncode == 42

    def test_timeout(self):
        jobs = [
            JobSpec(
                command=[sys.executable, "-c", "import time; time.sleep(10)"],
                n_gpus=0, name="slow", timeout=0.5,
            ),
        ]
        results = Scheduler(gpu_ids=[], n_cpu_workers=1).run(jobs)
        assert results[0].returncode == -1
        assert "TIMEOUT" in results[0].stderr_tail

    def test_log_dir(self, tmp_path):
        jobs = [
            JobSpec(
                command=[sys.executable, "-c", "print('logged-output')"],
                n_gpus=0, name="log-test",
            ),
        ]
        results = Scheduler(gpu_ids=[], n_cpu_workers=1, log_dir=tmp_path).run(jobs)
        assert results[0].returncode == 0
        log_file = tmp_path / "log-test.stdout.log"
        assert log_file.exists()
        assert "logged-output" in log_file.read_text()

    def test_env_passthrough(self):
        jobs = [
            JobSpec(
                command=[sys.executable, "-c", "import os; print(os.environ['MY_VAR'])"],
                n_gpus=0, name="env-test", env={"MY_VAR": "test-value"},
            ),
        ]
        results = Scheduler(gpu_ids=[], n_cpu_workers=1).run(jobs)
        assert results[0].returncode == 0
        assert "test-value" in results[0].stdout_tail

    def test_empty_job_list(self):
        results = Scheduler(gpu_ids=[], n_cpu_workers=1).run([])
        assert results == []


class TestGPUAssignment:
    """Test GPU assignment logic using fake GPU IDs (no real GPU needed)."""

    def test_single_gpu_job_gets_cuda_visible_devices(self):
        jobs = [
            JobSpec(
                command=[sys.executable, "-c", "import os; print(os.environ.get('CUDA_VISIBLE_DEVICES', ''))"],
                n_gpus=1, name="gpu-check",
            ),
        ]
        results = Scheduler(gpu_ids=[3], n_cpu_workers=0).run(jobs)
        assert results[0].returncode == 0
        assert "3" in results[0].stdout_tail

    def test_multi_gpu_job_gets_multiple_devices(self):
        jobs = [
            JobSpec(
                command=[sys.executable, "-c", "import os; print(os.environ.get('CUDA_VISIBLE_DEVICES', ''))"],
                n_gpus=2, name="multi-gpu",
            ),
        ]
        results = Scheduler(gpu_ids=[0, 1, 2, 3], n_cpu_workers=0).run(jobs)
        assert results[0].returncode == 0
        devices = results[0].stdout_tail.strip()
        assigned = [int(x) for x in devices.split(",")]
        assert len(assigned) == 2

    def test_dynamic_gpu_reuse(self):
        """4 single-GPU jobs on 2 GPUs — second pair must wait for first to finish."""
        jobs = [
            JobSpec(
                command=[sys.executable, "-c",
                         "import os, time; time.sleep(0.3); print(os.environ['CUDA_VISIBLE_DEVICES'])"],
                n_gpus=1, name=f"gpu-job-{i}",
            )
            for i in range(4)
        ]
        t0 = time.monotonic()
        results = Scheduler(gpu_ids=[5, 7], n_cpu_workers=0).run(jobs)
        elapsed = time.monotonic() - t0
        assert all(r.returncode == 0 for r in results)
        assert len(results) == 4
        # 4 jobs on 2 GPUs, 0.3s each => ~0.6s minimum (2 waves)
        assert elapsed >= 0.5
        # All assigned GPUs should be from our pool
        all_assigned = set()
        for r in results:
            all_assigned.update(r.gpu_ids)
        assert all_assigned <= {5, 7}

    def test_gpu_and_cpu_jobs_mixed(self):
        jobs = [
            JobSpec(
                command=[sys.executable, "-c",
                         "import os; print('gpu:', os.environ.get('CUDA_VISIBLE_DEVICES', ''))"],
                n_gpus=1, name="gpu-job",
            ),
            JobSpec(
                command=[sys.executable, "-c",
                         "import os; print('cpu:', os.environ.get('CUDA_VISIBLE_DEVICES', ''))"],
                n_gpus=0, name="cpu-job",
            ),
        ]
        results = Scheduler(gpu_ids=[2], n_cpu_workers=1).run(jobs)
        assert len(results) == 2
        assert all(r.returncode == 0 for r in results)
        by_name = {r.name: r for r in results}
        assert "2" in by_name["gpu-job"].stdout_tail
        # CPU job should get empty CUDA_VISIBLE_DEVICES
        assert "cpu:" in by_name["cpu-job"].stdout_tail

    def test_job_too_many_gpus_raises(self):
        jobs = [JobSpec(command=["echo"], n_gpus=4, name="greedy")]
        with pytest.raises(ValueError, match="needs 4 GPUs"):
            Scheduler(gpu_ids=[0, 1], n_cpu_workers=0).run(jobs)

    def test_dry_run(self, capsys):
        jobs = [
            JobSpec(command=["python", "train.py"], n_gpus=1, name="train"),
            JobSpec(command=["python", "preprocess.py"], n_gpus=0, name="preprocess"),
        ]
        Scheduler(gpu_ids=[0, 1], n_cpu_workers=2).run_dry(jobs)
        out = capsys.readouterr().out
        assert "2 jobs" in out
        assert "train" in out
        assert "preprocess" in out


class TestCrashRecovery:
    """Verify the scheduler handles crashes without leaking GPUs or hanging."""

    def test_gpu_returned_after_crash(self):
        """A crashing GPU job must return its GPU so the next job can run."""
        jobs = [
            JobSpec(
                command=[sys.executable, "-c", "raise SystemExit(1)"],
                n_gpus=1, name="crash",
            ),
            JobSpec(
                command=[sys.executable, "-c",
                         "import os; print(os.environ['CUDA_VISIBLE_DEVICES'])"],
                n_gpus=1, name="survivor",
            ),
        ]
        # Only 1 GPU: if the crash leaks it, survivor hangs forever.
        results = Scheduler(gpu_ids=[6], n_cpu_workers=0).run(jobs)
        assert len(results) == 2
        by_name = {r.name: r for r in results}
        assert by_name["crash"].returncode == 1
        assert by_name["survivor"].returncode == 0
        assert "6" in by_name["survivor"].stdout_tail

    def test_multi_gpu_crash_returns_all_gpus(self):
        """A 2-GPU job that crashes must return both GPUs."""
        jobs = [
            JobSpec(
                command=[sys.executable, "-c", "raise SystemExit(1)"],
                n_gpus=2, name="crash-multi",
            ),
            # These two single-GPU jobs need both GPUs back to run.
            JobSpec(
                command=[sys.executable, "-c",
                         "import os; print(os.environ['CUDA_VISIBLE_DEVICES'])"],
                n_gpus=1, name="after-1",
            ),
            JobSpec(
                command=[sys.executable, "-c",
                         "import os; print(os.environ['CUDA_VISIBLE_DEVICES'])"],
                n_gpus=1, name="after-2",
            ),
        ]
        results = Scheduler(gpu_ids=[0, 1], n_cpu_workers=0).run(jobs)
        assert len(results) == 3
        by_name = {r.name: r for r in results}
        assert by_name["crash-multi"].returncode == 1
        assert by_name["after-1"].returncode == 0
        assert by_name["after-2"].returncode == 0

    def test_early_crash_late_success(self):
        """Fast crash + slow success: both complete, scheduler doesn't abort."""
        jobs = [
            JobSpec(
                command=[sys.executable, "-c", "raise SystemExit(1)"],
                n_gpus=1, name="fast-crash",
            ),
            JobSpec(
                command=[sys.executable, "-c", "import time; time.sleep(0.3); print('done')"],
                n_gpus=1, name="slow-success",
            ),
        ]
        results = Scheduler(gpu_ids=[0, 1], n_cpu_workers=0).run(jobs)
        assert len(results) == 2
        by_name = {r.name: r for r in results}
        assert by_name["fast-crash"].returncode == 1
        assert by_name["slow-success"].returncode == 0
        assert "done" in by_name["slow-success"].stdout_tail

    def test_crash_with_output(self):
        """Job that prints then crashes — output is still captured."""
        jobs = [
            JobSpec(
                command=[sys.executable, "-c",
                         "print('partial-output'); raise SystemExit(1)"],
                n_gpus=0, name="crash-with-output",
            ),
        ]
        results = Scheduler(gpu_ids=[], n_cpu_workers=1).run(jobs)
        assert results[0].returncode == 1
        assert "partial-output" in results[0].stdout_tail

    def test_crash_stderr_captured(self):
        """Job that writes to stderr before crashing — stderr is captured."""
        jobs = [
            JobSpec(
                command=[sys.executable, "-c",
                         "import sys; sys.stderr.write('error-msg\\n'); raise SystemExit(1)"],
                n_gpus=0, name="stderr-crash",
            ),
        ]
        results = Scheduler(gpu_ids=[], n_cpu_workers=1).run(jobs)
        assert results[0].returncode == 1
        assert "error-msg" in results[0].stderr_tail

    def test_timeout_returns_gpu(self):
        """A timed-out GPU job must return its GPU."""
        jobs = [
            JobSpec(
                command=[sys.executable, "-c", "import time; time.sleep(10)"],
                n_gpus=1, name="timeout-job", timeout=0.5,
            ),
            JobSpec(
                command=[sys.executable, "-c",
                         "import os; print(os.environ['CUDA_VISIBLE_DEVICES'])"],
                n_gpus=1, name="after-timeout",
            ),
        ]
        results = Scheduler(gpu_ids=[9], n_cpu_workers=0).run(jobs)
        assert len(results) == 2
        by_name = {r.name: r for r in results}
        assert by_name["timeout-job"].returncode == -1
        assert by_name["after-timeout"].returncode == 0
        assert "9" in by_name["after-timeout"].stdout_tail

    def test_all_crash_no_hang(self):
        """If every job crashes, scheduler still returns (no deadlock)."""
        jobs = [
            JobSpec(command=[sys.executable, "-c", "raise SystemExit(i+1)"],
                    n_gpus=1, name=f"crash-{i}")
            for i in range(3)
        ]
        results = Scheduler(gpu_ids=[0, 1], n_cpu_workers=0).run(jobs)
        assert len(results) == 3
        assert all(r.returncode != 0 for r in results)


class TestRunJobs:
    """Test the top-level run_jobs convenience function."""

    def test_run_jobs_with_log_dir(self, tmp_path):
        jobs = [
            JobSpec(command=[sys.executable, "-c", "print('alpha')"], n_gpus=0, name="alpha"),
            JobSpec(
                command=[sys.executable, "-c",
                         "import os; print('gpu:', os.environ['CUDA_VISIBLE_DEVICES'])"],
                n_gpus=1, name="beta",
            ),
        ]
        log_dir = tmp_path / "logs"
        results = run_jobs(jobs, gpu_ids=[4], n_cpu_workers=1, log_dir=log_dir)
        assert len(results) == 2
        assert all(r.returncode == 0 for r in results)
        # Log files should exist
        assert (log_dir / "alpha.stdout.log").exists()
        assert "alpha" in (log_dir / "alpha.stdout.log").read_text()
        assert (log_dir / "beta.stdout.log").exists()
        assert "4" in (log_dir / "beta.stdout.log").read_text()

    def test_run_jobs_dry_run(self, tmp_path):
        jobs = [JobSpec(command=["echo", "hi"], n_gpus=0, name="noop")]
        results = run_jobs(jobs, dry_run=True, log_dir=tmp_path / "logs")
        assert results == []

    def test_run_jobs_summary_shows_failures(self, tmp_path, capsys):
        jobs = [
            JobSpec(command=[sys.executable, "-c", "print('ok')"], n_gpus=0, name="good"),
            JobSpec(command=[sys.executable, "-c", "raise SystemExit(1)"], n_gpus=0, name="bad"),
        ]
        results = run_jobs(jobs, n_cpu_workers=2, log_dir=tmp_path / "logs")
        out = capsys.readouterr().out
        assert "1 passed" in out
        assert "1 failed" in out
        assert "bad" in out
