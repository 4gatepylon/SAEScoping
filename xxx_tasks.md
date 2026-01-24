Now:
- Launch train on math
- Test GEPA on multiple MMLU and launch multiple MMLU GEPAs
- Launch train on 3-4 other topics
- Make sure at least 1-2 topics are clearly safety-relevant (look into WMDP and SecBench, etc...)

Later:
- Finish software (multi-model server, gepa cleanup, trainer cleanup)
- Look into more tasks, more datasets (mainly support algorithm on a few more options, finalize interface; no need to support all flags)
- Ask Dylan about baselines, etc...
- Commit YAMLs for server configuration so it's a 1-click operation; find the best batch-size, etc... based on my hardware