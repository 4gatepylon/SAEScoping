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
    - We save every 1000 steps of elicitation and of PGD
    - We always do BOTH elicitation and PGD with PGD algorithm (projected gradient descent)
    - We ignore "SFT'ed Vanilla Model" for now since we already have numbers for those.

And then we take the CARTESIAN PRODUCT of (MODEL, SCOPE DOMAIN, ELICITATION DOMAIN, SPARSITY)---except of course where scope domain = elicitation domain---and
run a full end-to-end run for each. The process is the following:
1. Collect calibration/saliency maps for each (MODEL, SCOPE DOMAIN). This is _cached_. This is where Wanda is used (in other formulations we can use Taylor approximation
    to determine saliency, etc...). This produces the following data:
    - Saves the specific calibration map.
    - We do a sweep of the sparsities to plot (and logs/save) per-sparsity metrics (loss, LLM judge results, etc...) for the original model. This sweep can be larger than the PGD sparsities set and mainly meant to be informative. This is logged to WANDB in a different pane in a pane under `calibration/` and it includes metrics such as sparsity of `nn.Linear` modules (what wanda affects) as well as REAL sparsity. This sweep includes zero sparsity to get baseline values.
2. Prune the model and then perform PGD for some number of steps. This produces the following data:
    - For each PGD run a WANDB plot of multiple metrics over the course of the run; each run is prefixed with `pgd_` and each pane is in the `pgd` namespace. This includes:
        - Train loss
        - Validation loss
        - LLM Judge quality
        - LLM Judge relevance
        - LLM Judge fluency
        - LLM Judge ground truth similarity
    - Intermediate and final checkpoint(s) are saved
3. For each (MODEL, SCOPE DOMAIN) we pick a reasonable checkpoint from the PGD output and then perform elicitation on ALL valid ELICITATION DOMAINS. This produces the following data:
    - Same over-time metrics as (2)
    - Intermediate and final checkpoint(s) are saved (just like 1)

NOTE: for all runs, if it reaches a certain threshold of loss/LLM judge results then the run stops and it is done. For PGD this is in terms of a minimum value allowable for LLM Judge Relevance and Fluency in terms of _vanilla_ numbers (i.e. 100%). For Elicitation this is also in terms of vanilla numbers but a smaller threshold (also minimum; 90%) and for the OOD (elicitation) domain.

NOTE: we pick a reasonable checkpoint like such: (1) if there is only one then pick that one (common tbh), (2) otherwise it depends on who is selecting but PGD elicitation is really the only one that has to pick and it will pick the one that is best OOD on the domain that will be elicited for.

For the paper we intend to report the following results:
- Table per scope domain, per output domain of worst-case elicitation result and PGD result.
    - Specifically, we pick the maximum sparsity such that post-PGD (pre-elicitation) reaches at least 100% LLM judge score (as a fraction of previous score; obviously the score might be less than 100% by a lot)
    - We then pick for each elicitation run on that specific model, the worst-case elicitation result (i.e. best it ever was on that)
This 2D table can then have slashes to seperate between methods once we compare with other methods.

# Configurations
Each configuration is stored as a YAML inside this folder. There are two TYPES of configurations
- Model/training configurations (used for BOTH PGD and Elicitation). This includes:
    - SFTConfig passthrough (PGDTrainer is an SFTTrainer with a few extra bits added to it) and information such as which LLM Judge, samples, evaluation domains, PGDTrainer "train-after" layer (for us it's 31 for 9b and 12b but none for the rest), etc... This is per-model since each model has different size and therefore implies different batch size.
    - Training Wrapper arguments: early stopping thresholds, other metadata about how to save checkpoints, etc... (these are basically the same everywhere though)
- Experiment configuration (which has two parts):
    - Sweep configurations (defines calibration size, sparsities, etc...). It also says which model configurations to use.
    - Operational configuration: this defines how things are stored, how to deal with possible memory issues, etc...

Because (look at `Testing and Running`) we have 3 full integration tests/runs available we have 3 experiment configurations. We have 4 model configurations since there are 4 models total used in this work.

# Computational Model and Files
As you can see above there are three step TYPES: (1) calibration, (2) pruning, and (3) elicitation. Each individual step INSTANCE may or may not DEPEND on a previous step. Moreover, each individual step takes in (usually) some basic parameters (such as model name or path, domain, etc...) and then outputs another such version. It also has side effects in the artifacts location (at `$SAESCOPING_ARTIFACTS_LOCATION/baselines_2026_05_02_pgd_biology_only`).

The dependency graph is compiled from the experiment YAML into a graph YAML stored in the artifacts state directory (`runtime_state_mirror/dependency_graph.yaml`). Each step instance of a given type depends on the instance of the previous type that produces its arguments:
- **Calibration** instance `(model, scope_domain)`: no prior step dependencies. Takes a base model and a scope domain, produces a saliency map.
- **PGD** instance `(model, scope_domain, sparsity)`: depends on the calibration instance for the same `(model, scope_domain)`. Takes the saliency map + sparsity, produces a pruned-then-PGD-recovered checkpoint.
- **Elicitation** instance `(model, scope_domain, sparsity, elicitation_domain)`: depends on the PGD instance for the same `(model, scope_domain, sparsity)`. Takes a PGD checkpoint + elicitation domain, produces an elicited checkpoint.

If a job crashes or OOMs, the scheduler marks it as failed and will not run anything that depends on it. The failed job's dependents are skipped (not retried).

There are therefore three scripts for each of the step TYPES:
- `calibrate.py` for the calibration step
- `prune.py` for the pruning step
- `elicitation.py` for the elicitation step

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
- `smoke_test_calibrate.py` for the calibration step
- `smoke_test_prune.py` for the pruning step
- `smoke_test_elicitation.py` for the elicitation step
(smoke test is like the mini test but only does one step on some fake data)

NOTE: _EVERYTHING REQUIRES CUDA_ (to be fast).

# Storage/Artifact dump format
ALL artifacts, including saliency masks, model generations, LLM judge results, etc... are stored into `$SAESCOPING_ARTIFACTS_LOCATION/baselines_2026_05_02_pgd_biology_only/{subfolder_name}`. So are WANDB logs. The subfolders are:
- `saliency_maps/{model}/{scope_domain}/` for computed saliency maps (cached; reused across sparsities)
- `wandb` for WANDB logs (we set `WANDB_DIR` to point here; W&B separates by project name internally)
- `pgd_checkpoints/{scope_domain}/{sparsity}/` for all PGD weight/checkpoint outputs
- `elicitation_checkpoints/{scope_domain}/{elicitation_domain}/{sparsity}/` for all Elicitation weight/checkpoint outputs
- `pgd_judge_logs/{scope_domain}/{sparsity}/` for all generations and LLM Judgement results in the PGD phase
- `elicitation_judge_logs/{scope_domain}/{elicitation_domain}/{sparsity}/` for all generations and LLM Judgement results in the Elicitation phase
- `runtime_state_mirror` for a sequence of mirrors of the state as stored in the memory of the runner (and other logs). Basically, this just helps us monitor and check for the fact that the scheduler is actually running the right jobs, etc...

NOTE: in each folder (except `wandb`) a specific `metadata.json` and `README.md` is automatically created to describe what this is. Metadata includes information such as the sparsity, domain, number of steps/trainer arguments, etc...

NOTE: every evaluation run judges on ALL 4 domains (biology, chemistry, math, physics), regardless of which domain the model was pruned/elicited on. The judgement JSONL rows contain `seed` and `judge_name` fields to differentiate between judge prompts/types, and domain is encoded in the file path structure. This is what `OneClickLLMJudgeScopingEval` does internally.

# CUDA Support
This supports running on one or more GPUs. The scheduler achieves this. Each script takes in a GPU (single one) as an argument and the scheduler awaits for completion of jobs, then assigns new ones to the free GPUs. It does this from a finite and constant SET of GPUs that are passed in as arguments using the `--devices` flag.

NOTE: it is assumed that if a GPU is specified as free for this evaluation, then it will remain free FOREVER throughout the entire program.

# Logging
NOTE: as you can see, during the process of each of the steps above, we collect metrics (such as losses and LLM judge results) across multiple steps. Usually we log every ~50-ish steps,
but for cheaper metrics we judge more often and for more expensive ones (LLM Judges) we log less often. We log MORE OFTEN than we save.

# Misc. things you might want to know
NOTE: we always save the final checkpoints of the models in each step of the process, but because of our choice of saving every 1K for 2K steps there is no need to manually write code
to save the final model.

NOTE: only a single GPU is supported per step instance. The scheduler dispatches across one or more GPUs for the entire sweep experiment.

NOTE: each training run is expected to take around 10 hours in the worst case scenario (if you think around 1 second/grad. accum. step and the batch size is around 1 for effective batch size of 16, you need to do around 32K such steps for a total 32K seconds; there are 3.6K seconds in an hour, so it's around 10-ish hours). The computational span should be around 20 hours at most, assuming you can get maximum parallelism. If early-stopping is effective, then we can expect 2-10x savings in many cases.

NOTE: If the early-stopping triggers, then the program is guaranteed to save a checkpoint regardless.

NOTE: this is designed to be agnostic to the method used to acquire the saliency maps. However, it depends on the method using saliency maps. This means in the future it CAN be extended trivially to:
- Magnitude pruning
- Random pruning
- Gradient/Taylor-approximation based pruning
- Iterative versions of ^ where each iteration ONLY does calibration and where the final output is a mask, such that just applying the masks flat out is the same as the iterative process.
(currently only supports Wanda based pruning)

# Storage Costs
NOTE: this entire process will produce for (2 models x 4 scope domains x 3 elicitation domains / scope domain x 4 sparsities) = 96 runs.
    - We will save 2 models x 4 scope domains x 2 checkpoints = 16 PGD outputs per run (for a total of around 20GB per output and therefore around 320GB)
    - We will save 2 models x 4 scope domains x 3 elicitation domains / scope domain x 2 checkpoints = 48 elicitation outputs per run (at around 20GB per for a total of around 1TB)
=> Therefore, we will take at most 1.5TB (with a wide margin) of space.

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