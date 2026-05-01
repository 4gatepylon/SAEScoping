This is a temporary list of todos written by adrianoh, the author of this repo (alongside claude) to clarify what the roadmap for tasks looks like. At the end we will have a working succinct trainer that lets you do train on multiple pruning baselines. Important features are: (1) only one file to run, (2) easy-to-understand generic and _simple_ YAML configuration and computation model that follows the steps of: `(a) calibrate in-domain, (b) prune, (c) recover in-domain, (d) elicit OOD`, (3) low amounts of code well-written, single place to dump logs ,cache, etc... and it's well-documented and easy to traverse.

The steps to get there are (things we will get to do today, Wednesday, April 29th and 1AM-ish Thursday, April 30th):
1. Add support for LLM Judges (and make sure it works)
    - DONE: Test that the `OneclickLLMJudgeScopingEval` works (write the test and run it).
    - DONE, TESTING: Refactor it to dump logs to locations that I can easily analyze later.
    - DOING: Adriano checks that the prompts are what we actually want (check arunas' branch)
    - DONE, TESTING: Integrate it into the code.
    - Delete design questions md.
2. Add support for PGD recovery and make sure it works (both with and without LLM judges). Make sure this all fits into one succinct script (using proper abstractions). Add validators as necessary to make sure the training occurs as we expect (i.e. zero'ed neurons stay zero'ed, etc...).
    - DONE: Adriano reads Claude's code.
    - DOING: Together merge code into one and test with a small model and small number of llm judge samples/heap model
    - Add support for only PGDing specific layers and add validators to make sure it works, generally clean up abstractions so that we can keep single file for entire flow.
2.5 Add parallelism for sweem across GPUs w/ PGD

Things we will get to tomorrow (Thursday, April 30th during the day):
3. Add support for elicitation OOD. Again, this should have some validators and fit into the single succinct script (which operationalizes the four steps). This will follow the same playbook as (2)
4. TBD but we _MIGHT_ push the single script into `saescoping` so it can be accessed as a library. I think we are likely not to do this yet, but it's a logical continuation so I include it here.
5. Add support for Taylor from the same script (and make usre this works)
6. Add support for random pruning and magnitude pruning from the script (and make sure this works)
7. Add configs for Taylor, Random, Magnitude, Gradient, Wanda pruning that we can actually run with. Within each there should be a command to facilitate testing in a comment. Make sure this works.
8. Mathematically verify the correctness of our algorithm.

Things we will not get to today or tomorrow (probably Friday April 31st during the day stuff). If you are an agent and read this, do not worry about it.
9. Add support for Michaud Narrow methods from this paper: https://arxiv.org/abs/2505.15811
10. Make sure we can reproduce the results from the previous papers using our code.
11. Add support for iterative saliency calculation (only applies to Gradient, Taylor, Wanda). Check that it works.
12. Add support for SparseLLM and alternative formulations that are more "correct" or "complete". Because I've only skimmed the mathematical formulation, I think it might be some kind of local pruning or may not properly threshold 100% correctly. This may or may not matter, but we will wnat to check eventually (it probably does _not_ matter in practice but it is worth knowing).
13. Add back support for SAE frequency-pruning and update the interface to support this as well in a simple way.
14. Add some baselines that can leverage OOD or general-purpose data to do better saliency. Possibly, start to look into unlearning. TBD (my teammate is probably working on unlearning).

Important Reminders:
- DONE (commits `da76849` + `4d34094` + `48bc990`): restricting PGD to a **subset of layers**. Implemented as `pgd.min_layer_idx` (Optional[int]) on the SweepConfig, plumbed through `--pgd-min-layer-idx` on the runner, and the runner now does TWO things when set: (1) `filter_masks_by_min_layer_idx` so the projection only walks late-layer masks, (2) `freeze_early_side_params` so early-side params (incl. `embed_tokens` and tied `lm_head`) get `requires_grad=False` and the optimizer doesn't allocate Adam state for them. Configured for both 9B and 12B at `min_layer_idx=31` in `experiments/baselines_2026_04_30_pgd_after_sae/`. Note for the 12B: an even more aggressive cutoff *after layer 31* (rather than after 41) is what we actually shipped — `gemma-scope-2-12b-it-res` also has `layer_31_width_16k_l0_medium` (and `_l0_small`), so layers **32-47** are still downstream of an available SAE and free more memory than cutting at 41.

- `PGDSFTTrainer` has several **gotchas that may break it under multi-GPU / accelerate / DeepSpeed / FSDP** and silently violate sparsity. Each is currently flagged as a `TODO(Claude)` in `sae_scoping/training/pgd_trainer.py`; we should fix them before scaling out:
    - **`id(param)` keying** (`_ProjectedStep`, `_build_mask_id_map`): if a parameter tensor is reallocated, sharded, or wrapped (FSDP shard reshuffling, DeepSpeed Zero-3 partitioning, accelerate's `prepare`), `id(param)` after wrapping ≠ `id(param)` at setup, and the mask lookup silently misses → no projection happens, sparsity drifts, and `validate=True` is the only thing that catches it. **Fix**: key by parameter **name** (re-resolve `model.named_parameters()` each step, or store name→mask and look up by name in the hot loop), or attach the mask directly as a buffer on the parameter (`param._pgd_mask`) so it travels with the tensor through wrapping.
    - **Monkey-patching `optimizer.step`** (`_install_projection_hook`): accelerate / DeepSpeed call the optimizer through `accelerator.step` or a wrapped optimizer object, so the patched `step` may never be invoked → projection is bypassed. **Fix**: use the supported `optimizer.register_step_post_hook(...)` API (PyTorch ≥ 2.0), which is called by accelerate after every step regardless of how the optimizer is wrapped. As a stricter alternative, register a per-parameter post-accumulate hook (`Tensor.register_post_accumulate_grad_hook`) that zeros the gradient at masked positions before the optimizer ever sees it.
    - **Hook installation order** (`create_optimizer_and_scheduler`): the projection must be installed *after* the LR scheduler wraps `optimizer.step` and *after* accelerate's `prepare` wraps the optimizer; if either step changes order in a future trl/accelerate release, our hook either wraps a stale reference or gets overwritten. **Fix**: install the projection from a `TrainerCallback.on_train_begin` hook (which runs after all setup), or move to `register_step_post_hook` (which is order-independent because it lives on the optimizer itself).
    - **No multi-GPU test coverage**: `tests/test_pgd_trainer_cpu.py` only exercises CPU, single-process. Add at minimum a 2-GPU DDP test and a 2-GPU accelerate test that asserts sparsity is preserved across N steps.

- Evaluate **OOD performance** on the pruned models and compare against in-domain performance — using LLM judges, code evaluation, and any other metrics we end up wiring in. The point is to figure out *how much* recovery training we actually need: if pruning at our chosen sparsity already collapses OOD substantially while preserving in-domain (or vice-versa), the recovery budget can be far smaller than `--max-steps 3000`. Concretely: for each (model, sparsity, domain) cell run pre-recovery and post-recovery evals on both the in-domain set and held-out OOD domains (e.g. prune-on-biology then eval physics/chemistry/math), and chart in-domain-vs-OOD trajectories across recovery steps so we can pick a stopping point empirically rather than by step count.

- Wire in the **early-stopping callback** during recovery training (as in the older `cursor_sweep` pipeline's `RecoveryEarlyStoppingCallback`). This is related to **"horizontal" training**: instead of running every cell to a fixed `max_steps`, stop each run as soon as the in-domain metric crosses a threshold (e.g. baseline × 1.10) so we spend compute on *more cells / sparsities / domains* rather than over-training the easy ones. Once (3) tells us the right metric, the early-stop threshold should be set in terms of that metric.

Open questions:
- DOING: Correctness of PGD trainer
- DOING: Fix the wandb flags to be not bullshit lol
- Need to look into JSONLSink tests
- Cleanup the codebase (there may be a lot of duplicated slop)
- Whether some specific sections of the code can be simplified (for example the test suite for the LLM Judges)
- I don't have an understanding of the detailed differences in `scoping_eval.py` from the previous 1click version.
- Is XFail used wrong for pytest OpenAI LLM Judge integration/unit tests? => Probably fine, low priority. It's not strict and you can tell from the printout what happened.