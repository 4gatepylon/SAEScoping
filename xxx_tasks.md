# Datasets Now:
- **Chemistry:** Find a great chemistry evaluation -> this should be straightforward
  - wdmp
  - smol instruct
  - mol instructions
  - TBD
- **Cyber:** Find great cyber evaluation (should be realistic in addition to MCQ and possible just loss or knowledge questions; 3cb may be an option; ctf environments would be great)
  - 3cb
  - secbench
  - wmdp
- **Math:** Find great math dataset(s) that are not too hard, not too easy, and verifiable/traininable and at least 20K large (can be a mixture of SFT, finetuning, etc...)

# Questions
- Should I just be using WMDP instead of cybermetric for evaluation? What should I be using for evaluation? Probably all 3 I guess?
- What should I be using for elicitation training? What is an effective training regime?

Evaluate on WMDP, SecQ, cybermetric.

# Tasks Now:
=> **SUPER Urgent:** confirm that cyber findings are valid based on formatting
=> **SUPER Urgent:** Evaluate cyber+chemistry utility on vibes and using judges (make sure to support all camel judges in a new script similar to the previously vibecoded biology one) + determine what to do/what conclusion is
=> **SUPER Urgent:** Evaluate math utility on vibes + eval. scripts + determine what to do, what conclusion is

# Tasks Probably next
=> **Urgent:** Launch Llama2 Spylab model for each specific trojan (in biology setting)
=> Launch training on all of the cyber stuff combined and with a couple different requested formats and with a couple system prompts and with a couple of golden examples that include reasoning from different larger LLMs (or from gemma itself, though we should be able to do that quickly by leveraging OpenRouter). This is meant to tackle the question of whether we just did a bad job of SFT (because the SFT on cyber actually lead to decreases in the other benchmarks).
  - Combine existing datasets (make sure to canonicalize and de-duplicate on question and response level; augment with chain of thought):
    - SecQA (we already have)
    - WMDB Cyber (we already have)
    - Cybermetric
     Look into some new ones:
      - MMLU computer Security
      - CTI-Bench: https://huggingface.co/datasets/AI4Sec/cti-bench — CTI-focused: MITRE ATT&CK mapping, CVE-to-CWE, threat actor attribution. More reasoning-heavy than pure knowledge.
      - SecEval: https://huggingface.co/datasets/XuanwuAI/SecEval — ~2k MCQs across 9 domains (web, network, crypto, pentest, memory safety). Note: some questions have multiple correct answers.
      - SecBench: https://huggingface.co/datasets/secbench-hf/SecBench — 2,730 MCQs, bilingual (EN/CN), splits by knowledge retention vs logical reasoning.
      - (Optional) Pre-combined dataset: https://huggingface.co/datasets/tuandunghcmut/combine-llm-security-benchmark — aggregates CyberMetric + SecEval + CTI-Bench with unified format, may save preprocessing time.
=> **Urgent:** (Fix and run, 1 GPU) Start GEPA optimization with the max_completion_tokens support (NOTE: if we are out of GPUs, then make server-pool w/ possibility to change model backend(s); I would recommend doing this in whatever way is simplest tbh but that supports changing model and maybe running more than one model at a time). I simply need to get the numbers for math, cyber, chemistry on the biology scoped model and on the vanilla model then I should be good to go.
=> **Urgent:** Write paper and submit the first draft, update Dylan
=> Make sure there is a clear MMLU evaluation script and fix issues with max_completion_tokens. Showcase the broad unlearning in some sort of plot.


In the end we should have:
- Existing plots
- Plot that showcases that 1e-4 is the right number next to a table that showcases for that specific model: {topics} x {finetuned, vanilla, vanilla gepa, gepa, sae recovery} and show that max(gepa, sae_recovery) << min(finetuned, vanilla, gepa). These can be side by side again.
- A table that shows for many MMLU subjects the fact that performance drops (need to measure MMLU performance before SAE and after)

## Observations
- Unclear if WMDP is too hard for the Gemma model or if I'm doing something wrong.
- We don't evaluate things like classification from latent space or looking at logits to confirm, maybe we should do this to make sure that it's not just a "deep formatting error"

# Later:
BEFORE SLEEP: launch some kind of distribution estimation and/or training using gemma3 and/or other topics (i.e. get the data to finish the second draft tomorrow)
=> Identify and fix issue where we appear to be using ultrachat model (should have put this shit in yamls more ordered)
=> Maybe (determine what we care about) re-launch all training using special hookpoint to limit how many layers we train in the origianl model (launch {vanilla} x {cyber, chemistry}). The reason for this is that the control is not actually 1:1 (on token length and on weight updates)
=> Determine whether to find better Chemistry or other setting(s)
=> Identify future thing sthat need to be trained and launch those trainings (it is really important to pipline this since it is now clearly go-time; we need to submit in a couple days)
=> Write paper using our best current results and submit (we will iterate from this)
=> Test if GRPO works and showcase RLVR/RLHF...?
=> Remove all 5 trojans plz

Find verifiable dataset for each of ^ and launch GEPA optimizer on each of them. Let's try to do MCQ if possible then move on to the rest.

- Finish software (multi-model server, gepa cleanup, trainer cleanup)
- Look into more tasks, more datasets (mainly support algorithm on a few more options, finalize interface; no need to support all flags)
- Ask Dylan about baselines, etc...
- Commit YAMLs for server configuration so it's a 1-click operation; find the best batch-size, etc... based on my hardware
- **Code:** Find a great code evaluation and training corpus (show that it improves gemma)
- **Forbidden Knowledge** (BeaverTails; it's large; it's roughly verifiable, and it has good categories): https://huggingface.co/datasets/PKU-Alignment/BeaverTails
- **Materials Science:** TBD but I think this should be verifiable. Could also look into law/legal, medicalQA, genomic benchmarks, physics, finance, etc... look into this: https://claude.ai/share/ef7590b9-e78b-4b02-bd47-78b7af9c78e6 (also look into Cyber with CTF/environment like 3cb)
- Launch train on math (and for each of these)
- Test GEPA on multiple MMLU and launch multiple MMLU GEPAs (for each of these)
