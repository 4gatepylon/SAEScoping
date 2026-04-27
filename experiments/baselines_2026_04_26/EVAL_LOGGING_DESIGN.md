# Eval Logging Design — Generation + Judge Logs

Status: **not yet implemented**. This doc captures the design decision for whoever picks it up.

## Problem

All model generations and judge results are built in memory (`JudgementsDf` dataframe in `scoping_eval.py:evaluate()`) then discarded. Only aggregate scores (a few floats) reach disk via `elicitation_results.json`. Raw generations, judge prompts, judge explanations, and per-sample scores are lost.

## Current disk writes (for context)

**Calibration sweep** writes: saliency safetensors, masks safetensors, `metrics.json` (SparsityResult), recovered HF checkpoints, `result.json` (MethodDomainResult), `manifest.json`.

**Elicitation sweep** writes: `elicitation_results.json` (list of dicts with aggregate losses + judge means). Optionally elicited model checkpoints.

**Neither writes:** per-sample losses, model generations, judge prompts/responses/explanations, or any structured logs.

## Design

### Format: flat self-contained JSONL

One line per (generation, judge) pair. Every line carries full experiment context — no joins needed.

```jsonl
{"generation_id":"a3f...","ts":"2026-04-27T14:32:01Z","phase":"elicitation","model_id":"google/gemma-2-9b-it","method":"wanda","sparsity":0.5,"in_domain":"biology","ood_domain":"chemistry","question":"What is the pH of water?","prompt":"<bos><start_of_turn>user\n...","response":"The pH of pure water is 7...","judge_name":"relevance","judge_model":"gpt-4.1-nano","judge_prompt":"You are evaluating...","judge_score":1.0,"judge_explanation":"The response directly...","tags":[],"meta":{}}
```

### Schema fields

| Field | Type | Notes |
|---|---|---|
| `generation_id` | str (UUID) | Same ID for all judge rows sharing one generation |
| `ts` | str (ISO 8601) | When the eval ran |
| `phase` | str | `"elicitation"`, `"recovery"`, `"baseline"` |
| `model_id` | str | HF model ID |
| `method` | str | `"wanda"`, `"taylor"`, etc. |
| `sparsity` | float | |
| `in_domain` | str | |
| `ood_domain` | str | |
| `question` | str | Raw question text (`seed` in current df) |
| `prompt` | str | Chat-template-formatted input |
| `response` | str | Model's generated text |
| `judge_name` | str or null | `"relevance"`, `"fluency"`, `"ground_truth_similarity"`, or null if no judge |
| `judge_model` | str or null | e.g. `"gpt-4.1-nano"` |
| `judge_prompt` | str or null | Hydrated judge template sent to API |
| `judge_score` | float or null | Normalized 0/0.5/1 |
| `judge_explanation` | str or null | Judge's textual reasoning |
| `tags` | list[str] | Extensible — for future annotation, agent labels, flags |
| `meta` | dict | Extensible — arbitrary key-value metadata |

### Where to intervene (two changes, ~30 lines)

**1. `scoping_eval.py:evaluate()`** — add optional `log_path: Path | None` and `log_context: dict | None` params. After building the judge dataframe, enrich each row with context fields + `generation_id` (UUID per unique prompt) + `ts`, then append as JSONL.

**2. `elicitation_sweep.py:evaluate_elicited()`** — pass `log_path=output_dir / "eval_log.jsonl"` and `log_context={"method": ..., "model_id": ..., "sparsity": ..., "in_domain": ..., "ood_domain": ..., "phase": "elicitation"}`.

### Generations without a judge

When `--no-judge` is set, no generations happen (loss-only eval). Start by only logging when the judge runs — the data already exists in memory, it just needs to not be thrown away. Adding a generation-only logging step to the loss path is a separate, larger change.

### Why JSONL

- Appendable (multiple eval runs can write to the same file)
- Streamable (`jq`, `grep`, `pandas.read_json(lines=True)`)
- One record per line — no need to parse surrounding structure
- Works with every log viewer and analysis tool

### Why flat + self-contained

- `grep "chemistry" *.jsonl` finds all chemistry evals
- `jq 'select(.judge_score < 0.5)' *.jsonl` finds low scores
- `jq 'select(.generation_id == "...")' *.jsonl` reconstructs all judges for one generation
- No sidecar files, no foreign keys, no schema joins
