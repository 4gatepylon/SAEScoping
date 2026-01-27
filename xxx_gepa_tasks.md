# GEPA Tasks
TBH this is not about GEPA. This is more broad/general than that.

I have some issues with the code right now. I need to iterate in a few ways: (1) I need to quickly change the datasets that I evaluate and the answer and answer formats that I try to elicit. Everything is 1-turn (or more generally, needs just one response). It's almost always either (a) multiple choice or (b) judge-based without golden answers or (c) judge-based with golden answers, or (d) code evaluation-based. I have 58 hours until submission deadline and I will need to sleep around 18 of those hours so I will have around 40 hours of work. Iterating takes too long and I guess I'm not sure why.

I think the problems are the following:
- I need to write new scripts to evaluate on new datasets.
- I need to support different querying formats (huggingface local, local LLM, openai, openrouter, etc...)
- I need to support different ways of doing ETL (DSPY, my own extraction)
- I need to support new datasets that have different formats for different things
- Code and outputs are duplicated and disorganized across two repositories and it's hard to find what's where.
- I have different visualization code in different places that does kind of similar things but is meant to do the same conceptual thing (just plot accuracy at different points etc....)
    - I think this is slightly les
- Sometimes some things are not logged, such as responses so I cannot judge myself as to whether or not something is reasonable or not, what the issues may be, etc...
- Launching common commands with small variations takes a trillion flags so then I waste a TON of time doing it.
- I have no train/test/validation split (or at least I'm not properly splitting in a standard way where I know whether I'm experiencing data contamination or not... fuck!)
- I am missing certain training runs and stuff... or not; generally unsure what I have and what I may be missing or not
- There isn't a clear way to name models/etc... for agents, there isn't a clear way to define what transformations to apply (there should be a simple and clear set of tarnsformations such as SAE-enhance, prune SAE, SFT, GRPO, GEPA, add transformation, and all of these parameterized by what data)
- I'm not sure what my baselines are
    - I think I want to compare to michaud, leni's paper and some unlearning prior work but I'm not honestly exactly sure aht this should be
- I have very little time and need to prioritize the like the minimal key things
- Naming is not standard
...

It should be possible to ask agents uch as claude code to "evaluate dataset X with model Y" and have it "just happen" (i.e. the agent writes a command runs it and it "just works" and I can inspect the logs etc...)

So it sounds like what I need to do is to:
1. Define my transformations/steps
2. Define my terminology very clearly
3. Standardize the dataset format(s)
4. Inform (3) and my current/initial dataset choices using prior work (specifically I should read and understand WMDP, Distillation Robustifies Unlearning, and Michaud's paper)
5. Ideally put the transformations into one big script that "runs the algorithm". The "algorithm" should be one code file that just does all the things. It can early stop obviously.
6. De-duplicate my dataset(s), files, etc...
7. Probably standardize ETL to DSPY (but I will need to )
8. Probably support multi-model per server and support multi-gpu so I have one one port/base-name and all the models I may possible need. I want to not need to launch new server sintances wirth massive commands.
9. Make sure to write a bunch of tiny scripts with clear names fpr what im doing that invoke the big commands.
10. Move most things out of experiments into sae_scoping (or something like that).
11. Add a clear way to slot in baselines.
12. Define the interfaces in some dclear way ??