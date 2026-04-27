# wandb logging — what gets logged where (throwaway)

**Delete me once you've internalized this.** Status as of `experiments/baselines_2026_04_26/` (commit `2d2e737`).

## TL;DR

| Sweep | Logs to wandb? | What gets logged | Project (default) |
|---|---|---|---|
| `calibration_and_recovery_sweep.py` | **NO** (nothing) | — | — |
| `elicitation_sweep.py` | YES — but only end-of-run eval metrics, *not* training loss curves | `elicit/{ood}/{key}` scalars | `sae-scoping-elicitation` |

The reason training curves do *not* hit wandb is that `helpers.make_sft_config` hardcodes `report_to = "none"` (helpers.py:232) and lists it in `RESERVED_SFT_FIELDS` (helpers.py:142-147), so trl/HF Trainer never reports.

## Where runs land

Your env already sets:

```
WANDB_DIR=/mnt/align4_drive2/adrianoh/.cache/wandb
WANDB_CACHE_DIR=/mnt/align4_drive2/adrianoh/.cache/wandb
WANDB_DATA_DIR=/mnt/align4_drive2/adrianoh/.cache/wandb
WANDB_CONFIG_DIR=/mnt/align4_drive2/adrianoh/.cache/wandb
WANDB_PROJECT=deleteme        # ← overridden by --wandb-project in elicit
WANDB_API_KEY=<set>
```

Local files: `/mnt/align4_drive2/adrianoh/.cache/wandb/wandb/run-<timestamp>-<id>/`
Cloud: `https://wandb.ai/<your-entity>/sae-scoping-elicitation` (entity = whoever owns the API key).

The `WANDB_PROJECT=deleteme` env var is *ignored* by elicitation because `wandb.init(project=...)` is called explicitly with `--wandb-project` (elicitation_sweep.py:279-281). It would only matter if calibration started using wandb.

## Run name format

elicitation_sweep.py:281
```
elicit/{method}/{model_slug}/{in_domain}/sp{sparsity}
```
e.g. `elicit/wanda/google--gemma-2-2b-it/biology/sp0.5`

Each in-domain run iterates OOD targets and logs `elicit/{ood_domain}/in_domain_loss`, `elicit/{ood_domain}/ood_loss`, plus judge subscores if `--no-judge` is omitted (elicitation_sweep.py:315).

## Smoke test recipes

### Make 100% sure online logging works

```bash
unset WANDB_MODE                                # online (the default)
export WANDB_PROJECT=sae-scoping-smoke           # only matters if a script doesn't override
# WANDB_API_KEY is already in env

./run.sh elicit \
    --method wanda --model google/gemma-2-2b-it \
    --in-domain biology --ood-domain chemistry --sparsity 0.5 \
    --n-train 64 --n-eval 32 --no-judge \
    --wandb-project sae-scoping-smoke \
    --sft-overrides '{"num_train_epochs": 1, "per_device_train_batch_size": 1, "max_length": 512, "logging_steps": 1}'
```

Watch stdout for `wandb: 🚀 View run …` — that line is your URL.

### Offline (no network)

```bash
export WANDB_MODE=offline
# … same elicit invocation …
# Sync later:
wandb sync /mnt/align4_drive2/adrianoh/.cache/wandb/wandb/offline-run-*
```

### Disable entirely

```bash
export WANDB_MODE=disabled
# wandb.init becomes a no-op; no files written, no network.
```

## Calibration / recovery has no wandb at all

If you want recovery training curves in wandb, you have two options:

1. **Quick hack** (just for one run): edit `helpers.py:232` to `cfg["report_to"] = "wandb"` and remove `"report_to"` from `RESERVED_SFT_FIELDS` (helpers.py:142-147). Then trl will auto-init a wandb run named after `output_dir`.
2. **Proper fix**: add a `--wandb-project` flag to `calibration_and_recovery_sweep.py` worker, call `wandb.init(...)` before `trainer.train()` for the recovery phase, and either set `report_to="wandb"` in the SFT config or call `wandb.log(...)` manually with end-of-recovery losses (mirroring elicitation_sweep.py:279-330).

Option 2 is the right one if you want to standardize. Option 1 is fine for a single debugging run.

## Where to look when it appears nothing logged

1. `wandb: ERROR …` lines in stdout (network / auth issues).
2. `WANDB_MODE=offline` accidentally still set from a previous shell.
3. Calling `worker` (calibration) and expecting wandb — there's none. Use `elicit`.
4. Run finished but you can't find it: it's under whatever entity owns `WANDB_API_KEY`, project `sae-scoping-elicitation` (or whatever `--wandb-project` was passed). Check `wandb.ai/<entity>/sae-scoping-elicitation`.
5. Local SQLite/cache: `/mnt/align4_drive2/adrianoh/.cache/wandb/wandb/`.

## Why the missing-artifact error happened

```
ls: cannot access 'artifacts/google--gemma-2-2b-it/biology/wanda_saliency.safetensors'
```

You hadn't run the smoke test yet, or it ran from a different cwd. `--artifact-dir` defaults to `./artifacts` resolved against the cwd where you launched the script, not the experiment dir. Either `cd experiments/baselines_2026_04_26` first, or pass `--artifact-dir /abs/path/to/artifacts`.
