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
=> Evaluate cyber+chemistry utility on vibes and using judges (make sure to support all camel judges in a new script similar to the previously vibecoded biology one)
=> Start GEPA optimization with the max_completion_tokens support (NOTE: if we are out of GPUs, then make server-pool w/ possibility to change model backend(s))
=> Evaluate math utility on vibes
=> Write paper and submit the first draft
=> Make sure there is a clear MMLU evaluation script and fix issues with max_completion_tokens. Showcase the broad unlearning in some sort of plot.



In the end we should have:
- Existing plots
- Plot that showcases that 1e-4 is the right number next to a table that showcases for that specific model: {topics} x {finetuned, vanilla, vanilla gepa, gepa, sae recovery} and show that max(gepa, sae_recovery) << min(finetuned, vanilla, gepa). These can be side by side again.
- A table that shows for many MMLU subjects the fact that perofrmance drops (need to measure MMLU performance before SAE and after)

## Observations
- Unclear if WMDP is too hard for the Gemma model or if I'm doing something wrong.

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
