"""DO NOT SUBMIT — scratch utility for OOM characterization.

This file MUST NOT be committed or kept in git history. It is a one-off
helper for finding the (batch_size, max_seq_len) OOM frontier on a multi-GPU
host. Delete (or git rm) before pushing.

Strategy:
- Per row (fixed batch_size), binary-search seq_len to find the largest
  passing value. ~log2(n_seq) cells per row instead of n_seq.
- Monotonicity propagation: if (b, s) is ok then (b'<=b, s'<=s) is ok; if
  (b, s) is oom then (b'>=b, s'>=s) is oom. Inferred cells are not retested.
- Rows are dispatched across _AVAILABLE_DEVICES in parallel. The shared
  `known` map lets each row skip cells already proven by sibling rows.
- run_fn is injected so the search logic can be unit-tested without GPUs.
"""

import concurrent.futures as cf
import subprocess
import threading
from pathlib import Path
from typing import Callable

import click

SCRIPT = Path(__file__).parents[4] / "experiments" / "baselines_2026_04_29" / "sweep_wanda.py"
OOM_MARKERS = ("CUDA out of memory", "torch.OutOfMemoryError", "OutOfMemoryError")
_AVAILABLE_DEVICES = ["cuda:0", "cuda:3", "cuda:6", "cuda:7"]
_PRINT_LOCK = threading.Lock()

RunFn = Callable[[int, int, str], str]  # (bsz, seq, device) -> "ok"|"oom"|"err"


def _log(msg: str) -> None:
    with _PRINT_LOCK:
        print(msg, flush=True)


def make_subprocess_run_fn(model_id: str, n_samples: int) -> RunFn:
    def run(batch_size: int, max_seq_len: int, device: str) -> str:
        cmd = [
            "conda",
            "run",
            "-n",
            "saescoping",
            "--no-capture-output",
            "python",
            str(SCRIPT),
            "--model-id",
            model_id,
            "--n-calibration",
            str(n_samples),
            "--n-eval",
            str(n_samples),
            "--max-seq-len",
            str(max_seq_len),
            "--batch-size",
            str(batch_size),
            "-s",
            "0.5",
            "--low-memory",
            "--device",
            device,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        out = result.stdout + result.stderr
        if result.returncode == 0:
            return "ok"
        if any(m in out for m in OOM_MARKERS):
            return "oom"
        return "err"

    return run


class GridSearcher:
    def __init__(
        self,
        bsz_list: list[int],
        seq_list: list[int],
        devices: list[str],
        run_fn: RunFn,
        label: str = "",
        verbose: bool = True,
    ):
        self.bsz_list = sorted(bsz_list)
        self.seq_list = sorted(seq_list)
        self.devices = devices
        self.run_fn = run_fn
        self.label = label
        self.verbose = verbose
        self.known: dict[tuple[int, int], str] = {}
        self.lock = threading.Lock()

    def _say(self, msg: str) -> None:
        if self.verbose:
            _log(msg)

    def _propagate(self, bsz: int, seq: int, status: str) -> None:
        if status == "ok":
            for b in self.bsz_list:
                if b > bsz:
                    continue
                for s in self.seq_list:
                    if s > seq:
                        continue
                    self.known.setdefault((b, s), "ok")
        elif status == "oom":
            for b in self.bsz_list:
                if b < bsz:
                    continue
                for s in self.seq_list:
                    if s < seq:
                        continue
                    self.known.setdefault((b, s), "oom")

    def _get_or_test(self, bsz: int, seq: int, device: str) -> str:
        with self.lock:
            cached = self.known.get((bsz, seq))
        if cached is not None:
            self._say(f"=== [{device}] bsz={bsz} seq={seq} -> {cached} (inferred)")
            return cached
        self._say(f">>> [{device}] {self.label} bsz={bsz} seq={seq}")
        status = self.run_fn(bsz, seq, device)
        self._say(f"<<< [{device}] bsz={bsz} seq={seq} -> {status}")
        with self.lock:
            self.known[(bsz, seq)] = status
            self._propagate(bsz, seq, status)
        return status

    def _search_row(self, bsz: int, device: str) -> tuple[int, int]:
        lo, hi = 0, len(self.seq_list) - 1
        last_ok = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            status = self._get_or_test(bsz, self.seq_list[mid], device)
            if status == "ok":
                last_ok = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return bsz, last_ok

    def run(self) -> dict[tuple[int, int], str]:
        import queue

        device_queues: dict[str, queue.Queue[int]] = {d: queue.Queue() for d in self.devices}
        for i, bsz in enumerate(self.bsz_list):
            device_queues[self.devices[i % len(self.devices)]].put(bsz)

        def device_worker(device: str) -> list[tuple[int, int]]:
            out = []
            q = device_queues[device]
            while not q.empty():
                bsz = q.get_nowait()
                out.append(self._search_row(bsz, device))
            return out

        with cf.ThreadPoolExecutor(max_workers=len(self.devices)) as ex:
            futures = [ex.submit(device_worker, d) for d in self.devices]
            for fut in cf.as_completed(futures):
                for bsz, last_ok in fut.result():
                    last_seq = self.seq_list[last_ok] if last_ok >= 0 else None
                    self._say(f"row bsz={bsz} threshold seq={last_seq}")
        with self.lock:
            for b in self.bsz_list:
                for s in self.seq_list:
                    self.known.setdefault((b, s), "?")
            return dict(self.known)


@click.command()
@click.option("--model-id", required=True, help="HuggingFace model ID (e.g. google/gemma-2-9b-it).")
@click.option("--batch-sizes", default="1,2,4,8", show_default=True, help="Comma-separated powers of 2.")
@click.option("--max-seq-lens", default="256,512,1024,2048", show_default=True, help="Comma-separated powers of 2.")
@click.option("--n-samples", default=8, show_default=True, type=int, help="Calibration and eval count per cell.")
def main(model_id, batch_sizes, max_seq_lens, n_samples):
    bsz_list = sorted(int(x) for x in batch_sizes.split(","))
    seq_list = sorted(int(x) for x in max_seq_lens.split(","))
    run_fn = make_subprocess_run_fn(model_id, n_samples)
    searcher = GridSearcher(bsz_list, seq_list, _AVAILABLE_DEVICES, run_fn, label=model_id)
    results = searcher.run()

    _log(f"\n=== Grid: {model_id} across {_AVAILABLE_DEVICES} ===")
    header = "bsz \\ seq"
    _log(f"{header:>10} " + " ".join(f"{s:>6}" for s in seq_list))
    for bsz in bsz_list:
        row = [results.get((bsz, s), "?") for s in seq_list]
        _log(f"{bsz:>10} " + " ".join(f"{r:>6}" for r in row))


if __name__ == "__main__":
    main()
