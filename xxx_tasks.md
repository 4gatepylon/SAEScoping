Now:
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

=> Find SFT data and train for cyber etc... loop
=> Train on CamelAI for chemistry
=> Add CamelAI evaluation (just copy over the chemistry evaluation)
=> Make sure there is a clear MMLU evaluation script and fix issues with max_completion_tokens


Saturday:
=> Determine whether to find better Chemistry or other setting(s)
=> Identify future thing sthat need to be trained and launch those trainings (it is really important to pipline this since it is now clearly go-time; we need to submit in a couple days)

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
