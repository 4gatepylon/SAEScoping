# `oom_grid` — find the (batch_size, max_seq_len) OOM frontier

Probe utility for working out which `(--batch-size, --max-seq-len)` cells
the runner can actually handle on a given host. Reports each cell as
`ok` / `oom` / `err` and prints the full grid.

> Status: scratch tool, lives outside the main sweep tree. Edit freely.

## Algorithm (in one paragraph)

For each fixed `batch_size` row, binary-search `max_seq_len` for the
largest passing value (`O(log n_seq)` cells per row instead of `O(n_seq)`).
Cells are tied together by a monotonicity invariant — if `(b, s)` is
`ok` then every `(b' ≤ b, s' ≤ s)` is `ok`; if `(b, s)` is `oom` then
every `(b' ≥ b, s' ≥ s)` is `oom` — and inferred cells are skipped.
Rows are dispatched across `_AVAILABLE_DEVICES` with a `ThreadPoolExecutor`;
a shared `known: dict[(b, s), str]` + lock lets one worker prune cells
already proven by another. The `run_fn` is injectable so you can unit-test
the search logic without GPUs (see `test_grid_oom.py`).

## Validate OOM (or not) on a real host

```bash
# from the repo root
python _skills/oom_grid/grid_oom.py \
  --model-id google/gemma-2-9b-it \
  --batch-sizes 1,2,4,8 \
  --max-seq-lens 256,512,1024,2048 \
  --n-samples 8
```

Each cell shells out to `experiments/baselines_2026_04_29/sweep_wanda.py`
with `--low-memory -s 0.5 --device cuda:N` and classifies the result by
scanning stdout/stderr for `OOM_MARKERS` (`"CUDA out of memory"`,
`"torch.OutOfMemoryError"`, `"OutOfMemoryError"`). Devices used by default:
`cuda:0,3,6,7` — edit `_AVAILABLE_DEVICES` in `grid_oom.py` for your host.

The final printed grid looks like:

```
=== Grid: google/gemma-2-9b-it across ['cuda:0','cuda:3','cuda:6','cuda:7'] ===
 bsz \ seq    256    512   1024   2048
         1     ok     ok     ok     ok
         2     ok     ok     ok    oom
         4     ok     ok    oom    oom
         8     ok    oom    oom    oom
```

Read the largest `(b, s)` cell still `ok` for each row — that's your
upper bound for that batch size. For example: at `bsz=2`, the safe
`max_seq_len` here is 1024.

## Unit-testing the search logic (no GPUs needed)

```bash
python -m pytest _skills/oom_grid/test_grid_oom.py -v
```

Tests use a synthetic frontier `b * s ≤ K` and assert (a) every cell ends
up at the truthful label, (b) the call count stays under the per-row
log-bound, and (c) multi-device dispatch doesn't corrupt the shared
`known` map. ~5 tests, runs in milliseconds.

## When to reach for this

- New host, unknown VRAM headroom for a model.
- Picking `--batch-size` and `--max-seq-len` for a `wanda_with_pgd_v1_*`
  full run on a fresh GPU type.
- Sanity-checking the OOM behaviour after touching the runner's memory
  knobs (gradient checkpointing, optimizer choice, etc.).

## When NOT to reach for it

- The mini script can already smoke a configuration in ~10 minutes —
  prefer that for a "does this run end-to-end" check.
- This utility times the *first* OOM it sees per cell. If the OOM only
  triggers under sustained training (hours in), the grid won't catch it.
