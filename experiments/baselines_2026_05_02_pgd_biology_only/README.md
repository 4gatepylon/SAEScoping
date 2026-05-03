# What this is
Wanda prune → PGD recovery sweep for StemQA (single calibration domain).

Streamlined replacement for baselines_2026_05_01_pgd_biology_only_after_sae/run.py by revamping the core layer: baselines_2026_04_29/sweep_wanda.py. It produces the outputs we want and actually specifies what will be produced here in the docstring.

While this may share functionality with previous scripts, it is mostly new.

Specifically it will produce results like:
```
+--------------------------------------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| Eval: biology                                    | sp=0.5                | sp=0.6                | sp=0.7                | sp=0.8                |
+==================================================+=======================+=======================+=======================+=======================+
| Vanilla Model                                    | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics |
+--------------------------------------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| SFT'ed Vanilla Model                             | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics |
+--------------------------------------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| Pruned Model                                     | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics |
+--------------------------------------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| PGD'ed Model (checkpoint 1)                      | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics |
+--------------------------------------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| PGD'ed Model (checkpoint 2)                      | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics |
+--------------------------------------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| PGD'ed Model (checkpoint ...)                    | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics |
+--------------------------------------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| Elicited on Physics (checkpoint 1) (ood)         | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics |
+--------------------------------------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| Elicited on Physics (checkpoint 2) (ood)         | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics |
+--------------------------------------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| Elicited on Physics (checkpoint ...) (ood)       | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics |
+--------------------------------------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| Elicited on Chemistry (checkpoint 1) (ood)       | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics |
+--------------------------------------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| Elicited on Chemistry (checkpoint 2) (ood)       | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics |
+--------------------------------------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| Elicited on Chemistry (checkpoint ...) (ood)     | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics |
+--------------------------------------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| Elicited on Math (checkpoint 1) (ood)            | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics |
+--------------------------------------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| Elicited on Math (checkpoint 2) (ood)            | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics |
+--------------------------------------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| Elicited on Math (checkpoint ...) (ood)          | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics |
+--------------------------------------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| Elicited on Biology (checkpoint 1) (in domain)   | Not done              | Not done              | Not done              | Not done              |
+--------------------------------------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| Elicited on Biology (checkpoint 2) (in domain)   | Not done              | Not done              | Not done              | Not done              |
+--------------------------------------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| Elicited on Biology (checkpoint ...) (in domain) | Not done              | Not done              | Not done              | Not done              |
+--------------------------------------------------+-----------------------+-----------------------+-----------------------+-----------------------+

+----------------------------------------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| Eval: chemistry                                    | sp=0.5                | sp=0.6                | sp=0.7                | sp=0.8                |
+====================================================+=======================+=======================+=======================+=======================+
| Vanilla Model                                      | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics |
+----------------------------------------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| SFT'ed Vanilla Model                               | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics |
+----------------------------------------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| Pruned Model                                       | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics |
+----------------------------------------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| PGD'ed Model (checkpoint 1)                        | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics |
+----------------------------------------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| PGD'ed Model (checkpoint 2)                        | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics |
+----------------------------------------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| PGD'ed Model (checkpoint ...)                      | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics |
+----------------------------------------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| Elicited on Physics (checkpoint 1) (ood)           | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics |
+----------------------------------------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| Elicited on Physics (checkpoint 2) (ood)           | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics |
+----------------------------------------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| Elicited on Physics (checkpoint ...) (ood)         | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics |
+----------------------------------------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| Elicited on Chemistry (checkpoint 1) (in domain)   | Not done              | Not done              | Not done              | Not done              |
+----------------------------------------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| Elicited on Chemistry (checkpoint 2) (in domain)   | Not done              | Not done              | Not done              | Not done              |
+----------------------------------------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| Elicited on Chemistry (checkpoint ...) (in domain) | Not done              | Not done              | Not done              | Not done              |
+----------------------------------------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| Elicited on Math (checkpoint 1) (ood)              | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics |
+----------------------------------------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| Elicited on Math (checkpoint 2) (ood)              | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics |
+----------------------------------------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| Elicited on Math (checkpoint ...) (ood)            | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics |
+----------------------------------------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| Elicited on Biology (checkpoint 1) (ood)           | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics |
+----------------------------------------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| Elicited on Biology (checkpoint 2) (ood)           | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics |
+----------------------------------------------------+-----------------------+-----------------------+-----------------------+-----------------------+
| Elicited on Biology (checkpoint ...) (ood)         | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics | Losses/Judges/Metrics |
+----------------------------------------------------+-----------------------+-----------------------+-----------------------+-----------------------+
... (and so on, for the other fields) ...
... (and there may be different sparsities---these are not guaranteed to be EXACT)...
```

# Formulation of the experiment
In other words, it takes in the following arguments (conceptually):
- MODEL(s) (i.e. google/gemma-2-9b-it or google/gemma-3-12b-it)
- DOMAINS (i.e. stem fields from StemQA like [math, chemistry, physics, biology])
    - SCOPE DOMAINS (we produce one domain-only model per scope domain)
    - ELICITATION DOMAINS (we produce one "attacked" model from a scope-domain-only model per domain; NOTE: we do NOT elicit on X for X=scope domain;
        the idea is to try to show whether we can reverse an i.e. code-only model to be non-code-only, i.e. by doing biology or something else)
- SPARSITIES (i.e. [0.5, 0.6, 0.7, 0.8])
- METRICS (i.e. for us: [train loss, validation loss, LLM Judge quality, LLM Judge relevance, LLM Judge fluency, LLM Judge ground truth similarity])
- Training parameters, but for us these are fixed in some basic ways:
    - Calibrate on around 4K samples
    - We always update ONLY the weights AFTER layer 31
    - We run for up to 2000 steps of elicitation and of PGD
    - We save PGD checkpoints every 1000 steps; elicitation saves only eval data (configurable via a flag — see agent reminder #13)
    - We always do BOTH elicitation and PGD with PGD algorithm (projected gradient descent)
    - We ignore "SFT'ed Vanilla Model" for now since we already have numbers for those.

And then we take the CARTESIAN PRODUCT of (MODEL, SCOPE DOMAIN, ELICITATION DOMAIN, SPARSITY)---except of course where scope domain = elicitation domain---and
run a full end-to-end run for each. The process is the following:
1. Collect calibration/saliency maps for each (MODEL, SCOPE DOMAIN). This is _cached_. This is where Wanda is used (in other formulations we can use Taylor approximation
    to determine saliency, etc...). This produces the following data:
    - Saves the specific calibration map (i.e. saliency scores per weight). The saliency map is computed once and never recomputed; all later steps just threshold it at different sparsity levels to get masks.
    - We do a sweep of sparsity _thresholds_ applied to the already-computed saliency map, to plot (and log/save) per-sparsity metrics (loss, LLM judge results, etc...) for the original model. No retraining happens here—just mask, evaluate, unmask. This sweep can be larger than the PGD sparsities set and is mainly meant to be informative. This is logged to WANDB in a different pane under `calibration/` and it includes metrics such as sparsity of `nn.Linear` modules (what wanda affects) as well as REAL sparsity. This sweep includes zero sparsity to get baseline values.
2. Prune the model and then perform PGD for some number of steps. This produces the following data:
    - For each PGD run a WANDB plot of multiple metrics over the course of the run; each run is prefixed with `pgd_` and each pane is in the `pgd` namespace. This includes:
        - Train loss
        - Validation loss
        - LLM Judge quality
        - LLM Judge relevance
        - LLM Judge fluency
        - LLM Judge ground truth similarity
    - Intermediate and final checkpoint(s) are saved
3. For each (MODEL, SCOPE DOMAIN, SPARSITY) and for each valid ELICITATION DOMAIN, we pick the best PGD checkpoint for that elicitation domain and then perform elicitation. This produces the following data:
    - Same over-time metrics as (2)
    - NOTE: elicitation does NOT save model checkpoints (only evaluation results / judge logs are stored). The PGD checkpoint it started from is already saved, and elicitation is the final step — we only need the eval metrics from it.

NOTE: for all runs, if it reaches a certain threshold of loss/LLM judge results then the run stops and it is done. Early stopping is triggered from within the TRL training callback (the evaluator returns a scores dict and DataFrame; the callback compares against stored vanilla baselines that we calculated during calibration and signals the trainer to stop). For PGD this is in terms of a minimum value allowable for LLM Judge Relevance and Fluency in terms of _vanilla_ numbers (i.e. 100%). For Elicitation this is also in terms of vanilla numbers but a smaller threshold (also minimum; 90%) and for the OOD (elicitation) domain. The vanilla baseline scores are computed during the calibration step's zero-sparsity evaluation and stored in the artifacts directory for later use by PGD/elicitation callbacks.

NOTE: we pick a reasonable checkpoint like such: (1) if there is only one then pick that one (common tbh), (2) otherwise it depends on who is selecting, but PGD→elicitation is really the only transition that has to pick and it will pick the checkpoint that is best OOD on the domain that will be elicited for. Concretely: since every evaluation during PGD logs all 4 domains' judge scores, for each saved checkpoint we find the evaluation log entry closest-after that checkpoint's step number and read the OOD score for the target elicitation domain. The checkpoint with the best such score is selected.

For the paper we intend to report the following results:
- Table per scope domain, per output domain of worst-case elicitation result and PGD result.
    - Specifically, we pick the maximum sparsity such that post-PGD (pre-elicitation) reaches at least 100% LLM judge score (as a fraction of previous score; obviously the score might be less than 100% by a lot)
    - We then pick for each elicitation run on that specific model, the worst-case elicitation result (i.e. best it ever was on that)
This 2D table can then have slashes to seperate between methods once we compare with other methods.

# Data
All steps use the same HuggingFace dataset (configured via `dataset_name`); calibration and PGD train on the scope domain's `train` split, elicitation trains on the elicitation domain's `train` split, and all evaluation (loss + LLM judges) uses the `validation` split across all 4 domains. The datapoints of train picked for calibration and PGD are DISJOINT. Elicitation uses a different dataset subsets (domains) so there is no worry there.

# Configurations
Each configuration is stored as a YAML inside this folder. There are two TYPES of configurations
- Model/training configurations (used for BOTH PGD and Elicitation). This includes:
    - SFTConfig passthrough (PGDTrainer is an SFTTrainer with a few extra bits added to it) and information such as which LLM Judge, samples, evaluation domains, PGDTrainer "train-after" layer (for us it's 31 for 9b and 12b but none for the rest), etc... This is per-model since each model has different size and therefore implies different batch size.
    - Training Wrapper arguments: early stopping thresholds (as fraction of vanilla, stored per-judge), other metadata about how to save checkpoints, etc... (these are basically the same everywhere though; the thresholds differ only between PGD=100% and elicitation=90%)
- Experiment configuration (which has two parts):
    - Sweep configurations (defines calibration size, sparsities, calibration sweep sparsities, etc...). It also says which model configurations to use.
    - Operational configuration: this defines how things are stored, how to deal with possible memory issues, etc...
    - `sft_overrides`: an optional `dict[str, Any]` merged on top of each model config's `SFTPassthroughConfig` at step-launch time. This is how mini_test clamps `max_steps=200, save_steps=200` and mini_real clamps `max_steps=1000, save_steps=500` without forking the model YAMLs.

Because (look at `Testing and Running`) we have 3 full integration tests/runs available we have 3 experiment configurations. We have 4 model configurations since there are 4 models total used in this work.

# Computational Model and Files
As you can see above there are three step TYPES: (1) calibration, (2) pruning, and (3) elicitation. Each individual step INSTANCE may or may not DEPEND on a previous step. Moreover, each individual step takes in (usually) some basic parameters (such as model name or path, domain, etc...) and then outputs another such version. It also has side effects in the artifacts location (at `$SAESCOPING_ARTIFACTS_LOCATION/baselines_2026_05_02_pgd_biology_only`).

The dependency graph is compiled from the experiment YAML into a graph YAML stored in the artifacts state directory (`runtime_state_mirror/dependency_graph.yaml`). Each step instance of a given type depends on the instance of the previous type that produces its arguments:
- **Calibration** instance `(model, scope_domain)`: no prior step dependencies. Takes a base model and a scope domain, produces a saliency map.
- **PGD** instance `(model, scope_domain, sparsity)`: depends on the calibration instance for the same `(model, scope_domain)`. Takes the saliency map + sparsity, produces a pruned-then-PGD-recovered checkpoint.
- **Elicitation** instance `(model, scope_domain, sparsity, elicitation_domain)`: depends on the PGD instance for the same `(model, scope_domain, sparsity)`. Takes a PGD checkpoint + elicitation domain, produces evaluation data (judge logs, metrics). The elicitation script (not the scheduler) performs checkpoint selection internally: it lists `checkpoint-{step}/` directories in the PGD checkpoint dir, parses `step_metadata.jsonl` from the corresponding PGD judge logs dir, and for each saved checkpoint finds the `step_metadata.jsonl` row with the smallest `train_step >= checkpoint_step`. From that row it reads the LLM judge score for the elicitation domain (matching by domain name and judge name `quality`, stripping the scope label). The checkpoint whose closest-after eval has the highest such score is selected. If only one checkpoint exists, it is used directly.

If a job crashes or OOMs, the scheduler marks it as failed and will not run anything that depends on it. The failed job's dependents are skipped (not retried).

There are therefore three scripts for each of the step TYPES:
- `calibrate.py` for the calibration step: this produces calibration data and evaluation results.
- `pgd.py`: just calls `pgd_or_elicit.py` with the right arguments.
- `elicit.py`: just calls `pgd_or_elicit.py` with the right arguments.
- `pgd_or_elicit.py`: invokes PGD with specific arguments for both training and operations.


There is one script for the INTERFACE design/definitions (pydantic models/schemas, validators, etc...):
- `interface.py` for the interface design/definitions

There is one scheduler that works for arbitrary sweep configs:
- `scheduler.py` for the scheduler

Then there is one bash script to invoke the scheduler with the right configurations PER experiment config (read `Configurations` and `Testing and Running` to understand):
- `run_mini_test.sh` to run the mini test experiment (with 2b and 4b models). NOTE: the test experiment is like the real one but with smaller parameters and less domains.
- `run_mini_real.sh` to run the mini real experiment (with 2b and 4b models)
- `run_full_real.sh` to run the full real experiment (with 9b and 12b models)

Obviously, as I said above, there are 3 experiment configs and 4 model configs:
- Experimental Configs
    - `mini_test.yaml`
    - `mini_real.yaml`
    - `full_real.yaml`
- Model Configs
    - `google__gemma-2-2b-it.yaml`
    - `google__gemma-3-4b-it.yaml`
    - `google__gemma-2-9b-it.yaml`
    - `google__gemma-3-12b-it.yaml`

There is also a smoke test for each of the individual steps:
- `smoke_test_calibrate.sh` for the calibration step
- `smoke_test_prune.sh` for the PGD recovery step (prune + recover in one shot)
- `smoke_test_pgd_or_elicit.sh` for the full PGD→elicitation pipeline
(smoke test is like the mini test but only does one step on some fake data)

NOTE: _EVERYTHING REQUIRES CUDA_ (to be fast).

# Storage/Artifact dump format
ALL artifacts, including saliency masks, model generations, LLM judge results, etc... are stored into `$SAESCOPING_ARTIFACTS_LOCATION/baselines_2026_05_02_pgd_biology_only/{subfolder_name}`. So are WANDB logs. The subfolders are:
- `saliency_maps/{model}/{scope_domain}/` — one per (model, scope_domain) pair; 4 domains × N models total.
    - `wanda_saliency.safetensors` — per-weight saliency scores (cached; reused across sparsities)
    - `vanilla_scores.json` — all-domains LLM judge scores + loss at zero sparsity, computed during calibration baseline eval. Keys mirror `OneClickLLMJudgeScopingEval.evaluate()` output: `llm_judge/{domain}/{scope}/{judge_name}` and `loss`. Read by PGD/elicitation callbacks for fraction-of-vanilla early stopping. NOTE: the scope label in keys (e.g. `out_of_scope` vs `attack_scope`) may differ between vanilla and elicitation runs; callbacks must compare by `(domain, judge_name)` only, stripping the scope component.
    - `calibration_sweep.json` — per-sparsity-threshold metrics from the informational sweep (loss, model sparsity, linear sparsity, judge scores per domain). The sparsity thresholds swept here come from `calibration_sweep_sparsities` in the experiment config (a superset of the PGD sparsities, e.g. `[0.0, 0.1, 0.2, ..., 0.9]`)
    - `metadata.json`, `README.md`
- `wandb/` — W&B logs (we set `WANDB_DIR` to point here; W&B separates by project name internally)
- `pgd_checkpoints/{model}/{scope_domain}/{sparsity}/` — PGD recovery outputs.
    - `checkpoint-{step}/` — TRL checkpoint directories (model weights + optimizer + scheduler state)
    - `metadata.json`, `README.md`
- `elicitation_checkpoints/{model}/{scope_domain}/{elicitation_domain}/{sparsity}/` — reserved for elicitation outputs, but NOT USED right now (because we cannot afford to store so many weights).
- `pgd_judge_logs/{model}/{scope_domain}/{sparsity}/` — PGD-phase evaluation artifacts.
    - `step_metadata.jsonl` — **primary queryable log**; one row per eval step. Schema: `{train_step, nn_linear_sparsity, loss, loss_delta_vs_baseline, model_sparsity, model_sparsity_late_layers (if min_layer_idx), llm_judge: {<scores_dict keyed by "llm_judge/{domain}/{scope}/{judge_or_group}">}}`. This is what checkpoint selection and early stopping read.
    - `judgements.jsonl` — raw per-judge rows from `OneClickLLMJudgeScopingEval`, written via `JsonlSink`. The evaluator writes to the sink internally (it accepts a `Sink` callable in `evaluate(judgement_sink=...)`), but the evaluator does NOT inject `train_step` — it doesn't know what step it's at. The callback must wrap the `JsonlSink` with a thin decorator that prepends `{"train_step": N, ...original_row}` before delegating. Schema after wrapping: `{train_step, canonical_row: {seed, prompt, response, judge_name, judge_template, judgement_score, judgement_explanation}, is_error, judgement_dict}`. Domain is recoverable from `seed` by matching against the domain question lists (the `seed` field is the original raw question text, not a formatted prompt).
    - `inference.jsonl` — raw per-generation rows, also written via `JsonlSink` through `evaluate(inference_sink=...)`. Same wrapping needed to inject `train_step`. Schema after wrapping: `{train_step, request, response}`. Prompts are special-token-stripped (no `<bos>`, `<start_of_turn>`, etc.).
    - `scores.json` — final eval scores snapshot (last `llm_judge` dict from `step_metadata.jsonl`)
    - `metadata.json`, `README.md`
- `elicitation_judge_logs/{model}/{scope_domain}/{elicitation_domain}/{sparsity}/` — elicitation-phase evaluation artifacts.
    - Same file structure as `pgd_judge_logs/`
- `runtime_state_mirror/` — append-only log of scheduler state. Never deletes or overwrites existing files.
    - `dependency_graph.yaml` — serialized DAG (written once at start, never modified)
    - `operations.jsonl` — one line per state transition: `{timestamp, step_id, event: started|completed|failed|skipped, detail}`
    - `state_v{NNN}_{ISO_TIMESTAMP}.yaml` — periodic full-state snapshots (node statuses, GPU assignments, wall-clock)

NOTE: in each folder (except `wandb`) a specific `metadata.json` and `README.md` is automatically created to describe what this is. Metadata includes information such as the sparsity, domain, number of steps/trainer arguments, etc...

NOTE: every evaluation run judges on ALL 4 domains (biology, chemistry, math, physics), regardless of which domain the model was pruned/elicited on. `OneClickLLMJudgeScopingEval.evaluate()` is called with the full set of domain datasets and returns both a scores dict (keyed by domain/judge) and a DataFrame with per-row `seed` and `judge_name` fields. Domain is encoded in the file path structure of the JSONL sinks, not in the evaluator's return keys.

# CUDA Support
This supports running on one or more GPUs. The scheduler achieves this. Each script takes in a GPU (single one) as an argument and the scheduler awaits for completion of jobs, then assigns new ones to the free GPUs. It does this from a finite and constant SET of GPUs that are passed in as arguments using the `--devices` flag.

NOTE: it is assumed that if a GPU is specified as free for this evaluation, then it will remain free FOREVER throughout the entire program.

# Logging
NOTE: as you can see, during the process of each of the steps above, we collect metrics (such as losses and LLM judge results) across multiple steps. Usually we log every ~50-ish steps,
but for cheaper metrics we judge more often and for more expensive ones (LLM Judges) we log less often. We log MORE OFTEN than we save.

# Misc. things you might want to know
NOTE: checkpoint saving is controlled by configuration. If a step is configured to save (via `save_strategy`/`save_steps` in the SFT config), it is guaranteed to actually save — including on early-stop. Our default configs save PGD checkpoints every 1K steps and do not save elicitation checkpoints (only eval data). This is a YAML-level choice, not a code-level one (see agent reminder #13).

NOTE: only a single GPU is supported per step instance. The scheduler dispatches across one or more GPUs for the entire sweep experiment.

NOTE: each training run is expected to take around 10 hours in the worst case scenario (if you think around 1 second/grad. accum. step and the batch size is around 1 for effective batch size of 16, you need to do around 32K such steps for a total 32K seconds; there are 3.6K seconds in an hour, so it's around 10-ish hours). The computational span should be around 20 hours at most, assuming you can get maximum parallelism. If early-stopping is effective, then we can expect 2-10x savings in many cases.

NOTE: if early-stopping triggers and the step is configured to save checkpoints, a final checkpoint is guaranteed to be saved. Eval data (judge logs, step_metadata) is always saved regardless of checkpoint config. You pass in the `--save_final_checkpoint` flag to guarantee this even if your save strategy fails. It only triggers if nothing had previously been written or it was too far off in number of steps (a flag is passed for that).

NOTE: this is designed to be agnostic to the method used to acquire the saliency maps. However, it depends on the method using saliency maps. This means in the future it CAN be extended trivially to:
- Magnitude pruning
- Random pruning
- Gradient/Taylor-approximation based pruning
- Iterative versions of ^ where each iteration ONLY does calibration and where the final output is a mask, such that just applying the masks flat out is the same as the iterative process.
(currently only supports Wanda based pruning)

NOTE: every script is made to be idempotent with the cache input it expects and cache output it produces (specifically it will fail and do nothing if its output is already there; moreover, if the entire experiment runs a second time, it will just skip those finished ones and then complete the rest---this allows a very simple way of doing everything: human just keeps relaunching the entire experiment until it is done).

# Storage Costs
NOTE: this entire process for the full_real experiment (2 models, 4 scope domains, 4 sparsities):
    - Calibration: 2 models × 4 scope domains = 8 saliency maps (negligible, ~100MB each)
    - PGD checkpoints: 2 models × 4 scope domains × 4 sparsities × 2 checkpoints = 64 checkpoints (~20GB each ≈ 1.3TB)
    - Elicitation: no checkpoints saved (only judge logs, which are small text files)
    - Total step instances: 8 calibrations + 32 PGD + 96 elicitations = 136
=> Therefore, we will take roughly **1.3TB** of disk space. The pre-flight disk check (see agent reminders) should verify this before launching.

# WANDB Support
Every specific test/run logs to a specific WANDB project. The projects are named like `deleteme__baselines_2026_05_02_pgd_biology_only__<test_or_run_name>`. For specific details look below at `Testing and Running`.

NOTE: each individual step INSTANCE gets a unique W&B run name by setting `WANDB_NAME` per step to encode the step type, model, domain, and sparsity so runs are human-readable in the W&B dashboard. They are formatted like such:
```
{step_type}__{model_short}__{scope_domain}__sp{sparsity}[__{elicit_domain}]__{hash8}
```
Examples:
- `calibrate__gemma-2-9b-it__biology__c9d1f3a4`
- `pgd__gemma-2-9b-it__biology__sp0.65__a3f2c1d0`
- `elicit__gemma-2-9b-it__biology__sp0.65__chemistry__b7e4a912`

Double underscore as separator so single underscores in model names don't collide. The 8-char hash covers the full config dict so two runs with identical human-readable prefixes but different hyperparams are still distinguishable.

# Testing and Running
There is support for:
    1. Integration testing specific steps. This logs to WANDB project `deleteme__baselines_2026_05_02_pgd_biology_only__<test_name>`
    2. Integration testing the whole pipeline with a shortened set of domains/models/etc... (and smaller ones too). This logs to WANDB project `deleteme__baselines_2026_05_02_pgd_biology_only__full`
        - Models are google/gemma-2-2b-it and google/gemma-3-4b-it
        - Sparsities are just 0.65 and 0.75 (x2 reduction)
        - LLM Judges use gpt-4.1-nano instead of gpt-4.1
        - Domains are just biology, math (x2 reduction)
        - Only train for 200 steps of PGD and 200 steps of elicitation and save the output. (x10 reduction in space and time)
        - Calibrate on 100 samples only (x40 reduction, but this part was really fast anyways since it's just inference without generation)
        - Full parameter finetuning instead of after a certain layer
    3. Same as (2) but with realer parameters: 1K steps of PGD and 1K steps of elicitation instead of 200. Calibrate on 1K samples instead of 100. Use real LLM Judges (gpt-4.1), all domains, etc... 
        This logs to WANDB project `baselines_2026_05_02_pgd_biology_only__small`
    4. Full run with the real models and numbers of steps. This logs to WANDB project `baselines_2026_05_02_pgd_biology_only__large`

# Things to be improved in future versions
This is a log of things that are NOT supported right now and will be improved in the future when we start supporting other pruning methods such as Taylor approximation.

- Lower computational span
- Shrink the search space
- Use more efficient search algorithms, such as binary search on the sparsities
- Understand better what hyperparameters to use by doing data science (i.e. on the weights, etc...)

# Reminders for agents implementing this

1. **No stored "pruned models" before PGD.** We do NOT save or load a separate pruned model. A pruned model is reconstructed on the fly by loading the original base model and then thresholding the cached saliency map at the desired sparsity (i.e. `compute_wanda_masks(saliency, sparsity)` → `apply_masks_to_model(model, masks)`). Provide loader abstractions that encapsulate this: e.g. a helper that takes `(model_id, saliency_path, sparsity, device)` and returns `(model, tokenizer, masks)` ready for PGD. Similarly for elicitation: a helper that loads a PGD checkpoint directly.

2. **Loader abstractions must handle caching.** Saliency maps are loaded via `safetensors` (see `load_or_compute_safetensors` in `sae_scoping.utils.cache`). `baselines_2026_04_29` has examples of this pattern but we are not copying that code verbatim — write fresh, clean abstractions that fit the three-step model here.

3. **Pre-flight disk space check.** Before the scheduler begins dispatching jobs, it must estimate the total disk footprint (number of checkpoints × approximate size per checkpoint, derived from model parameter count × bytes-per-param) and compare against `shutil.disk_usage(artifacts_root).free`. If estimated usage exceeds 90% of free space, abort with a clear error message stating required vs. available space. This is a best-effort guard, not a guarantee.

4. **No slashes in W&B run names or cache folder names.** Model IDs like `google/gemma-2-9b-it` contain `/`. All path components and W&B names must use the slash-safe form (replace `/` with `__`). Use the `_slash_safe(model_id: str) -> str` helper for this — define it once in `interface.py` and import everywhere. Never inline `model_id.replace("/", "__")`.

5. **All constants and magic numbers live in YAMLs.** Nothing is hardcoded in Python scripts. This includes: batch sizes, max_steps, save_steps, learning rates, calibration sample counts, sparsity lists, LLM judge model name, LLM judge n_samples, early stopping thresholds, dataset name, eval split name, warmup ratios, gradient accumulation, logging frequency, eval frequency, max_seq_len, etc. If a number appears in a script, it must trace back to a YAML field.

6. **Dataset contract: single HuggingFace dataset, domain = subset name.** The input dataset is a single HF dataset (configured via `dataset_name` in the experiment YAML, e.g. `4gate/StemQAMixture`) with subsets whose names exactly match the domain names used in the sweep (e.g. subset `"biology"` for domain `"biology"`). Each subset must have `"question"` and `"answer"` columns. If a future experiment needs a merged dataset, produce a new HF dataset with the right subset names upstream — the scripts here always do `load_dataset(dataset_name, subset=domain_name)`.

7. **Runtime state mirror: the YAML is a graph.** The `dependency_graph.yaml` written to `runtime_state_mirror/` is a serialized directed acyclic graph (nodes + edges via `deps` lists), not a flat list. Serialize via `DependencyGraph.model_dump()` → `yaml.safe_dump()`.

8. **Runtime state mirror: append-only, never delete.** The `runtime_state_mirror/` directory is an append-only log. Never delete or overwrite existing files. Each state snapshot is written as a new file with a monotonic version index and ISO timestamp, e.g. `state_v003_2026-05-02T22-15-00.yaml`. The dependency graph YAML is written once at the start and never modified. An `operations.jsonl` log records every state transition (job started, completed, failed, skipped) with timestamps.

9. **All stateful code goes into a class.** The scheduler's dispatch loop, its GPU pool, its job-status tracking — all of this lives in a class (e.g. `SchedulerState`). The per-step scripts (`calibrate.py`, `prune.py`, `elicitation.py`) may use a top-level function as the entry point but any mutable state accumulated during execution (e.g. running eval scores for early-stop comparison) lives in a class instance, not in module-level variables or closures.

10. **Underscore-prefix for helpers and local constants.** Any function or constant that is not part of the script's public contract gets a leading underscore: `_slash_safe()`, `_resolve_artifacts_base()`, `_STEP_TYPE`, etc. Public contract = the CLI entry point, the main `calibrate()` / `prune_and_recover()` / `elicit()` function, and the pydantic models.

11. **Each script defines its contract precisely.** Every script (`calibrate.py`, `prune.py`, `elicitation.py`, `scheduler.py`) must have a module-level docstring that specifies:
    - **Inputs**: what CLI args / files / environment variables it reads.
    - **Outputs**: what files / directories it creates (exact paths relative to `$SAESCOPING_ARTIFACTS_LOCATION`).
    - **Side effects**: what it logs to W&B, what JSONL sinks it writes.
    - **Idempotency**: how it detects that its work is already done and skips.
    - **Failure mode**: what happens on crash (partial outputs? cleaned up? left for retry?).

12. Define schemas for all inputs and outputs at the interface between scripts or large components.

13. It has to be POSSIBLE to edit the code trivially to store elicitation outputs. Specifically, it should be a configuration-level (YAML-level) change. We do NOT store it by default for our configurations since we only need to know the number, it doesn't take that long, and it takes a LOT of space.

14. Scheduler invocation can take `--devices` to overwrite YAML.