"""GPU integration tests for the job scheduler.

These tests actually use torch.cuda to verify that the scheduler's
CUDA_VISIBLE_DEVICES assignment results in real GPU access.
Skipped when CUDA is unavailable.
"""

import sys
import textwrap

import pytest
import torch

from sae_scoping.infrastructure.scheduler import JobSpec, Scheduler

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)


def _gpu_script(code: str) -> list[str]:
    """Wrap a Python snippet into a subprocess command."""
    return [sys.executable, "-c", textwrap.dedent(code)]


class TestRealGPU:
    """Tests that actually touch torch.cuda inside subprocess jobs."""

    @pytest.fixture
    def gpu_ids(self):
        return list(range(torch.cuda.device_count()))

    def test_job_sees_correct_device_count(self, gpu_ids):
        """A 1-GPU job should see exactly 1 CUDA device."""
        jobs = [
            JobSpec(
                command=_gpu_script("""
                    import torch
                    n = torch.cuda.device_count()
                    print(f"devices={n}")
                    assert n == 1, f"Expected 1 device, got {n}"
                """),
                n_gpus=1, name="device-count",
            ),
        ]
        results = Scheduler(gpu_ids=gpu_ids[:1], n_cpu_workers=0).run(jobs)
        assert results[0].returncode == 0
        assert "devices=1" in results[0].stdout_tail

    def test_multi_gpu_job_sees_correct_count(self, gpu_ids):
        """A 2-GPU job should see exactly 2 CUDA devices."""
        if len(gpu_ids) < 2:
            pytest.skip("Need at least 2 GPUs")
        jobs = [
            JobSpec(
                command=_gpu_script("""
                    import torch
                    n = torch.cuda.device_count()
                    print(f"devices={n}")
                    assert n == 2, f"Expected 2 devices, got {n}"
                """),
                n_gpus=2, name="multi-device-count",
            ),
        ]
        results = Scheduler(gpu_ids=gpu_ids[:2], n_cpu_workers=0).run(jobs)
        assert results[0].returncode == 0
        assert "devices=2" in results[0].stdout_tail

    def test_can_allocate_tensor_on_gpu(self, gpu_ids):
        """Job can actually allocate a tensor on the assigned GPU."""
        jobs = [
            JobSpec(
                command=_gpu_script("""
                    import torch
                    t = torch.randn(100, 100, device="cuda:0")
                    print(f"tensor_device={t.device}")
                    assert t.device.type == "cuda"
                """),
                n_gpus=1, name="tensor-alloc",
            ),
        ]
        results = Scheduler(gpu_ids=gpu_ids[:1], n_cpu_workers=0).run(jobs)
        assert results[0].returncode == 0
        assert "tensor_device=cuda:0" in results[0].stdout_tail

    def test_gpu_isolation_between_jobs(self, gpu_ids):
        """Two parallel GPU jobs get different physical devices."""
        if len(gpu_ids) < 2:
            pytest.skip("Need at least 2 GPUs")
        jobs = [
            JobSpec(
                command=_gpu_script("""
                    import os, torch, time
                    time.sleep(0.3)  # overlap with the other job
                    cvd = os.environ["CUDA_VISIBLE_DEVICES"]
                    props = torch.cuda.get_device_properties(0)
                    print(f"cvd={cvd} name={props.name}")
                """),
                n_gpus=1, name=f"isolation-{i}",
            )
            for i in range(2)
        ]
        results = Scheduler(gpu_ids=gpu_ids[:2], n_cpu_workers=0).run(jobs)
        assert len(results) == 2
        assert all(r.returncode == 0 for r in results)
        # They should have gotten different CUDA_VISIBLE_DEVICES
        cvds = set()
        for r in results:
            for line in r.stdout_tail.split("\n"):
                if line.startswith("cvd="):
                    cvds.add(line.split()[0])
        assert len(cvds) == 2

    def test_gpu_crash_frees_device_for_next_job(self, gpu_ids):
        """A GPU job that crashes returns the device for the next job."""
        jobs = [
            JobSpec(
                command=_gpu_script("""
                    import torch
                    _ = torch.randn(10, device="cuda:0")
                    raise SystemExit(1)
                """),
                n_gpus=1, name="gpu-crash",
            ),
            JobSpec(
                command=_gpu_script("""
                    import torch
                    t = torch.randn(10, device="cuda:0")
                    print(f"ok device={t.device}")
                """),
                n_gpus=1, name="gpu-after-crash",
            ),
        ]
        # Only 1 GPU — second job must wait for first to crash and release it.
        results = Scheduler(gpu_ids=gpu_ids[:1], n_cpu_workers=0).run(jobs)
        assert len(results) == 2
        by_name = {r.name: r for r in results}
        assert by_name["gpu-crash"].returncode == 1
        assert by_name["gpu-after-crash"].returncode == 0
        assert "ok device=cuda:0" in by_name["gpu-after-crash"].stdout_tail
