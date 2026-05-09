# aruna's wandb runs — what was copied and what's accessible

> Authored by Claude Code (Claude Opus 4.7, 1M context) under adriano's direction;
> figures and counts come from a live audit of the destination tree, not from
> the model's training data.

These notes describe the contents of `$SAESCOPING_ARTIFACTS_LOCATION/aruna_wandb/`,
populated by `experiments/copy_aruna_wandb_runs.py` from
`/mnt/align4_drive/arunas/sae-filters/SAEScoping/wandb/`.

## Per-run-dir layout

Every `run-YYYYMMDD_HHMMSS-<id>/` looks like this:

```
run-YYYYMMDD_HHMMSS-<id>/
├── run-<id>.wandb         binary log of metrics + events (~20–100 MB)
├── files/
│   ├── config.yaml        run config (text)
│   ├── wandb-metadata.json host, cmdline, git state
│   ├── wandb-summary.json  final scalar metrics
│   ├── output.log          process stdout/stderr
│   ├── requirements.txt    pip freeze at run start
│   └── media/table/
│       ├── charts/*.table.json     per-step chart panels
│       └── llm_judge/*.table.json  judgement / relevance / fluency / quality / similarity
├── logs/
│   ├── debug.log               wandb client log (real file)
│   ├── debug-internal.log      wandb sync-process log (real file)
│   └── debug-core.log -> /afs/.../arunas/.cache/wandb/logs/...   SYMLINK
└── tmp/code/                   empty in every copied run
```

## What you have access to

| Bucket | Count | Size | Coverage |
|---|---|---|---|
| `run-<id>.wandb` (binary metric log) | 25 | 987 MB | 25/25 — authoritative metrics + events |
| `files/media/table/charts/*.table.json` | 3204 | (largest contributor to `files/`) | per-step chart panels |
| `files/media/table/llm_judge/*.table.json` | 277 | | judgements, relevance, ground-truth-similarity, fluency, quality |
| `files/wandb-metadata.json` | 25 | small | host / cmdline / git SHA |
| `files/wandb-summary.json` | 25 | small | final scalar values per metric |
| `files/output.log` | 25 | a few MB total | stdout/stderr of training |
| `files/requirements.txt` | 25 | small | exact env at run start |
| `files/config.yaml` | **23** | small | full hyperparams (text) |
| `logs/debug.log` | 25 | ~1.5 MB total | wandb client log |
| `logs/debug-internal.log` | 25 | ~2 MB total | wandb sync-process log |

## What you do NOT have access to

| Item | How many | What it is | Severity |
|---|---|---|---|
| `logs/debug-core.log` (dangling symlink into `/afs/.../arunas/.cache/wandb/logs/`) | 25 / 25 | wandb-core's internal diagnostic Go-process log | **none — wandb-internal debug only, no experiment data** |
| `files/config.yaml` missing | 2 / 25 (`run-20260421_222011-efduc1pn`, `run-20260423_064934-12h5ydak`) | text-form config never flushed to disk for those runs | **low** — same config lives inside `run-<id>.wandb`; recoverable via `wandb sync` or the wandb SDK |
| `tmp/code/` contents | 25 / 25 (all empty) | wandb's staging area for code uploads — empty everywhere | **none — nothing was there to lose** |

## File-type tally under `files/` (all 25 runs combined)

```
.json   3529   (3204 charts + 277 llm_judge + 48 metadata/summary)
.log      25   (output.log)
.txt      25   (requirements.txt)
.yaml     23   (config.yaml — 2 runs missing it)
```

Disk total: **509 MB** (`du -sh`). Sum-of-byte-sizes is larger (~1.4 GB)
because most of the small `.table.json` files round up to a filesystem
block — trust `du` for actual space used.

## Bottom line

Everything needed for analysis is intact:

- `run-<id>.wandb` is the canonical record (metrics, config, events).
- The 3481 `.table.json` files give every logged Table directly as JSON,
  no wandb required.
- `wandb-summary.json` is the quickest path to final scalar metrics.

The only systematic loss is `debug-core.log × 25` (wandb's internal
diagnostic log, irrelevant to experiments) plus two missing
`config.yaml` files (recoverable from the binary).

## IDs that have no matching run dir

The IDs listed in `experiments/aruna-evals-fresh.json` but with no
matching `run-*-<id>` under arunas's wandb folder:

- `xz4e78xm` (Gemma-2 chemistry pretrained-SAE recover, regular fine-tuning)
- `lgcpc65b` (Gemma-2 chemistry pretrained-SAE attack on biology)
- `2b63aec8` (Gemma-3 biology pretrained-SAE recover)

These tokens appear only as text in the IDs file; the runs may never
have logged to disk, were deleted, or live in a location not scanned
(searched: `/mnt/align3_drive/arunas`, `/mnt/align4_drive/arunas`,
`/mnt/align4_drive2/arunas`).
