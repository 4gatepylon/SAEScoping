# DO NOT SUBMIT — Wanda Sweep + LLM Judge: Design Questions

Please answer inline (replace the `TODO` placeholders). I'll read this file back and implement accordingly.

---

## Judges & Evaluation

### 1. Which judges?
Available: `relevance`, `fluency`, `ground_truth_similarity` (requires answers in dataset).
Use all three, or a subset?

> You should use all 3 an report them seperately, but under a shared namespace in the wandb logs.

### 2. Which domains to evaluate?
Currently the sweep loads one `dataset_subset` (e.g. biology). Should the judge also evaluate on other domains (math, chemistry, physics) to measure cross-domain degradation at each sparsity?

> This should be a flag that can be passed. I think you would pass just a QA dataset to the judgement code, but the sweep script CAN just use the split name. Only the validation split should be used for judgement.

### 3. Judge model?
Default is `gpt-4.1-nano`. Keep that or use something else?

> Let's use it by default but make it possible to use others too.

### 4. How many judge samples per sparsity level?
More samples = more accurate but more expensive (each sample × each judge = 1 API call). Suggest a default for `--n-judge-samples`?

> Sure, default should be 50 (for each judge; they use the same questions and answers from the model).

---

## Storage & Logging

### 5. Output directory structure?
Proposal A — flat: `./results/sweep_{timestamp}.json` (one file, all sparsities)
Proposal B — nested: `./results/{model}/{subset}/{timestamp}/sparsity_{s}/judgements.json`
Proposal C — something else?

> Judgements should be stramed iwth flush to a JSONL and there should be a trace ID that is linked to the specific run metadata. You should store this in the os.environ (read the code now) artifacts folder for saescoping under outputs/{identifier for this run}. The trace ID should not cause you to omit run metadata that is available at the time. For example each run should be dumped to a different file with date timestamp and runner step stamp etc.

### 6. What to persist per sparsity level?
Check all that apply:
- [ ] Summary scores (the `llm_judge/domain/scope/quality` dict)
- [ ] Full judgement DataFrame (every prompt + response + judge score + explanation)
- [ ] Raw model generations (before judging)
- [x] Loss numbers alongside judge scores (both train AND eval)
- [ ] Masks or saliency maps (probably not — they're already cached)

> Remember to log metadata such as which model which sparsity the run date, arguments file, etc... (runner should ave this to artifacts)

### 7. W&B integration?
The existing scoping pipeline logs to W&B. Should this sweep also log, or local-only for now?

> YES. Log to wandb please. You should add flags to wandb and be able to be overriden by the os.environ (args > os.environ (os.environ is default) > config > defaults). Document a command to run with environ having `deleteme` project to do debugging.

---

## Analysis

### 8. Built-in analysis or separate?
Should the script produce analysis artifacts (e.g. a summary CSV/JSON easy to load in a notebook), or do you have a separate analysis pipeline?

> No it should just produce logs stored to the artifacts folder. The data should be logged clearly so in the future we CAN analyze it without rerunning the code.

### 9. Comparison across runs?
Do you need a standard schema so you can diff runs (different models, datasets, sparsity ranges), or is ad-hoc fine?

> Let's pick a schema. It should support taylor, wandb, etc...

### 10. Anything else?
Any other requirements, constraints, or preferences?

> NO
