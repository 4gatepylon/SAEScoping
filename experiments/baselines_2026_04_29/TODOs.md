This is a temporary list of todos written by adrianoh, the author of this repo (alongside claude) to clarify what the roadmap for tasks looks like. At the end we will have a working succinct trainer that lets you do train on multiple pruning baselines. Important features are: (1) only one file to run, (2) easy-to-understand generic and _simple_ YAML configuration and computation model that follows the steps of: `(a) calibrate in-domain, (b) prune, (c) recover in-domain, (d) elicit OOD`, (3) low amounts of code well-written, single place to dump logs ,cache, etc... and it's well-documented and easy to traverse.

The steps to get there are:
1. Add support for LLM Judges (and make sure it works)
    - Test that the `OneclickLLMJudgeScopingEval` works (write the test and run it).
    - Adriano checks that the prompts are what we actually want (check arunas' branch)
    - Integrate it into the code.
    - Refactor it to dump logs to locations that I can easily analyze later.
    - Delete design questions md.
2. Add support for PGD recovery and make sure it works (both with and without LLM judges). Make sure this all fits into one succinct script (using proper abstractions). Add validators as necessary to make sure the training occurs as we expect (i.e. zero'ed neurons stay zero'ed, etc...).
    - Adriano reads Claude's code.
    - Together merge code into one and test with a small model and small number of llm judge samples/heap model
    - Add support for only PGDing specific layers and add validators to make sure it works, generally clean up abstractions so that we can keep single file for entire flow.
3. Add support for elicitation OOD. Again, this should have some validators and fit into the single succinct script (which operationalizes the four steps). This will follow the same playbook as (2)
4. TBD but we _MIGHT_ push the single script into `saescoping` so it can be accessed as a library. I think we are likely not to do this yet, but it's a logical continuation so I include it here.
5. Add support for Taylor from the same script (and make usre this works)
6. Add support for random pruning and magnitude pruning from the script (and make sure this works)
7. Add configs for Taylor, Random, Magnitude, Gradient, Wanda pruning that we can actually run with. Within each there should be a command to facilitate testing in a comment. Make sure this works.
8. Mathematically verify the correctness of our algorithm.

Things we will not get to today. If you are an agent and read this, do not worry about it.
9. Add support for Michaud Narrow methods from this paper: https://arxiv.org/abs/2505.15811
10. Make sure we can reproduce the results from the previous papers using our code.
11. Add support for iterative saliency calculation (only applies to Gradient, Taylor, Wanda). Check that it works.
12. Add support for SparseLLM and alternative formulations that are more "correct" or "complete". Because I've only skimmed the mathematical formulation, I think it might be some kind of local pruning or may not properly threshold 100% correctly. This may or may not matter, but we will wnat to check eventually (it probably does _not_ matter in practice but it is worth knowing).
13. Add back support for SAE frequency-pruning and update the interface to support this as well in a simple way.
14. Add some baselines that can leverage OOD or general-purpose data to do better saliency. Possibly, start to look into unlearning. TBD (my teammate is probably working on unlearning).