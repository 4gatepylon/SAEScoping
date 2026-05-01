---
name: oom_grid
description: Probe the (batch_size, max_seq_len) OOM frontier for a model on this host. Use when picking --batch-size / --max-seq-len for a wanda+PGD run on a fresh GPU type, when validating that a memory-knob change (gradient checkpointing, optimizer choice, etc.) actually fits, or when a single mini smoke isn't enough to map the safe envelope. Skip when a single config has already been smoke-tested and you only care about that one cell.
user-invocable: false
---

## What it does

Runs `experiments/baselines_2026_04_29/sweep_wanda.py` repeatedly, one
subprocess per `(batch_size, max_seq_len)` cell of an injected grid, and
classifies each cell as `ok` / `oom` / `err` from the subprocess's
stdout/stderr. The output is a printed grid like:

```
=== Grid: google/gemma-2-9b-it across ['cuda:0','cuda:3','cuda:6','cuda:7'] ===
 bsz \ seq    256    512   1024   2048
         1     ok     ok     ok     ok
         2     ok     ok     ok    oom
         4     ok     ok    oom    oom
         8     ok    oom    oom    oom
```

The largest `(b, s)` cell still `ok` per row is the safe upper bound for
that batch size on this host.

## Algorithm (the part that makes it tractable)

For each `batch_size` row, binary-search `max_seq_len` for the largest
passing value (~`log2(n_seq)` cells per row). Cross-row pruning by
monotonicity:

* `(b, s) == ok`  ⇒  every `(b' ≤ b, s' ≤ s)` is `ok`.
* `(b, s) == oom` ⇒  every `(b' ≥ b, s' ≥ s)` is `oom`.

Inferred cells are skipped. Rows are dispatched across `_AVAILABLE_DEVICES`
in a `ThreadPoolExecutor`; a shared `known: dict[(b, s), str]` + `Lock`
lets one worker prune cells already proven by another. The probe-runner
(`run_fn`) is injectable — `make_subprocess_run_fn` shells out to
`sweep_wanda.py`, but `test_grid_oom.py` swaps in a synthetic frontier
to exercise the search logic without GPUs.

## Files

```
${CLAUDE_SKILL_DIR}/
├── SKILL.md
└── scripts/
    ├── grid_oom.py         # CLI + GridSearcher
    └── test_grid_oom.py    # synthetic-frontier unit tests
```

`grid_oom.py` resolves the runner script as
`Path(__file__).parents[4] / "experiments/baselines_2026_04_29/sweep_wanda.py"`
— the skill is portable as long as it stays four levels deep under the
repo root (`.claude/skills/oom_grid/scripts/`).

## How to validate OOM (or not) on a real host

```bash
# from the repo root
python .claude/skills/oom_grid/scripts/grid_oom.py \
  --model-id google/gemma-2-9b-it \
  --batch-sizes 1,2,4,8 \
  --max-seq-lens 256,512,1024,2048 \
  --n-samples 8
```

* **`OOM_MARKERS`** are the substrings used to classify a failure as OOM
  vs. some other error: `"CUDA out of memory"`,
  `"torch.OutOfMemoryError"`, `"OutOfMemoryError"`. Cells that fail for
  other reasons return `err`.
* **`_AVAILABLE_DEVICES`** defaults to `cuda:0,3,6,7` — edit the list at
  the top of `grid_oom.py` for your host's free GPUs.
* **`--n-samples`** is the calibration + eval count used per probe. Keep
  it tiny (default 8); the grid only needs to know whether the cell
  *runs*, not whether it produces meaningful output.

Each probe invocation looks like:

```bash
conda run -n saescoping --no-capture-output \
  python experiments/baselines_2026_04_29/sweep_wanda.py \
  --model-id <model> \
  --n-calibration <n> --n-eval <n> \
  --max-seq-len <s> --batch-size <b> \
  -s 0.5 \
  --low-memory \
  --device cuda:N
```

i.e. one Wanda saliency + one prune+eval at sparsity 0.5, with the
mask-monotonicity validator off (`--low-memory`) to keep CPU memory
flat.

## Unit-testing the search logic (no GPUs needed)

```bash
python -m pytest .claude/skills/oom_grid/scripts/test_grid_oom.py -v
```

The tests use a synthetic frontier `b * s ≤ K` and check:

1. `GridSearcher.run()` produces the truthful label for every cell.
2. The number of `run_fn` calls stays under `n_bsz * ceil(log2(n_seq + 1))` —
   the per-row binary-search bound.
3. `all-ok` and `all-oom` extremes prune to a handful of probes.
4. Multi-device dispatch doesn't corrupt the shared `known` map.

~5 tests, milliseconds. Run them whenever you touch `grid_oom.py`.

## When NOT to reach for it

* The mini script (`experiments/baselines_2026_04_30/wanda_with_pgd_v1_<...>_mini.sh`)
  already smokes a configuration in ~10 minutes. If you only care
  whether one specific `(b, s)` works, use that.
* This grid times the **first** OOM it sees per cell. OOMs that only
  trigger after hours of training (e.g. fragmenting allocator state,
  optimizer momentum buffers) won't show up here.
