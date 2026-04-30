# What is this?
This is a temporary list of todos written by adrianoh, the author of this repo (alongside claude) to clarify what the roadmap for tasks looks like. At the end we will have a working succinct trainer that lets you do train on multiple pruning baselines. Important features are: (1) only one file to run, (2) easy-to-understand generic and _simple_ YAML configuration and computation model that follows the steps of: `(a) calibrate in-domain, (b) prune, (c) recover in-domain, (d) elicit OOD`, (3) low amounts of code well-written, single place to dump logs ,cache, etc... and it's well-documented and easy to traverse.

# April 29th
The steps to get there are (things we will get to do today, Wednesday, April 29th and 1AM-ish Thursday, April 30th):
1. Add support for LLM Judges (and make sure it works)
    - DONE: Test that the `OneclickLLMJudgeScopingEval` works (write the test and run it).
    - DONE, TESTING: Refactor it to dump logs to locations that I can easily analyze later.
    - DONE: Adriano checks that the prompts are what we actually want (check arunas' branch)*
    - DONE: Integrate it into the code.
    - DONE: Delete design questions md.
2. Add support for PGD recovery and make sure it works (both with and without LLM judges). Make sure this all fits into one succinct script (using proper abstractions). Add validators as necessary to make sure the training occurs as we expect (i.e. zero'ed neurons stay zero'ed, etc...).
    - DONE: Adriano reads Claude's code.
    - DONE: Together merge code into one and test with a small model and small number of llm judge samples/heap model
2.5 DONE: Add parallelism for sweem across GPUs w/ PGD

# April 30th
Things we will get to tomorrow (Thursday, April 30th during the day):
3. Add support for elicitation OOD. Again, this should have some validators and fit into the single succinct script (which operationalizes the four steps). This will follow the same playbook as (2)
    - Add support for only PGDing specific layers and add validators to make sure it works, generally clean up abstractions so that we can keep single file for entire flow.
4. TBD but we _MIGHT_ push the single script into `saescoping` so it can be accessed as a library. I think we are likely not to do this yet, but it's a logical continuation so I include it here.
5. Add support for Taylor from the same script (and make usre this works)
6. Add support for random pruning and magnitude pruning from the script (and make sure this works)
7. Add configs for Taylor, Random, Magnitude, Gradient, Wanda pruning that we can actually run with. Within each there should be a command to facilitate testing in a comment. Make sure this works.
8. Mathematically verify the correctness of our algorithm.

# Apri 31st
Things we will not get to today or tomorrow (probably Friday April 31st during the day stuff). If you are an agent and read this, do not worry about it.
9. Add support for Michaud Narrow methods from this paper: https://arxiv.org/abs/2505.15811
10. Make sure we can reproduce the results from the previous papers using our code.
11. Add support for iterative saliency calculation (only applies to Gradient, Taylor, Wanda). Check that it works.
12. Add support for SparseLLM and alternative formulations that are more "correct" or "complete". Because I've only skimmed the mathematical formulation, I think it might be some kind of local pruning or may not properly threshold 100% correctly. This may or may not matter, but we will wnat to check eventually (it probably does _not_ matter in practice but it is worth knowing).
13. Add back support for SAE frequency-pruning and update the interface to support this as well in a simple way.
14. Add some baselines that can leverage OOD or general-purpose data to do better saliency. Possibly, start to look into unlearning. TBD (my teammate is probably working on unlearning).
15. Clean up (remove NAMING.md etc...)
16. Support better training methodologies (i.e. horizontal instead of veritcal...)

Open questions:
- DOING: Correctness of PGD trainer
- DOING: Fix the wandb flags to be not bullshit lol
- Review parallel mode
- Need to look into JSONLSink tests
- Cleanup the codebase (there may be a lot of duplicated slop)
- Whether some specific sections of the code can be simplified (for example the test suite for the LLM Judges)
- I don't have an understanding of the detailed differences in `scoping_eval.py` from the previous 1click version.
- Is XFail used wrong for pytest OpenAI LLM Judge integration/unit tests? => Probably fine, low priority. It's not strict and you can tell from the printout what happened.

* Verifiation script:
```
(saescoping) 4gate@Mac SAEScoping % cat diff_j2.sh
#!/bin/bash
set -e

git fetch origin aruna

ARUNA="origin/aruna"
BASELINES="origin/adriano/baselines"
ARUNA_DIR="sae_scoping/xxx_evaluation/iclr_judge_prompts"
BASE_DIR="sae_scoping/evaluation/prompts"

for name in fluency_classifier ground_truth_similarity relevance_classifier; do
    echo "=== $name.j2 ==="
    git show "$ARUNA:$ARUNA_DIR/$name.j2" > /tmp/j2_aruna.txt
    git show "$BASELINES:$BASE_DIR/$name.j2" > /tmp/j2_baselines.txt
    diff -u /tmp/j2_aruna.txt /tmp/j2_baselines.txt || true
    echo ""
done
(saescoping) 4gate@Mac SAEScoping % ./diff_j2.sh
From github.com:4gatepylon/SAEScoping
 * branch            aruna      -> FETCH_HEAD
=== fluency_classifier.j2 ===

=== ground_truth_similarity.j2 ===

=== relevance_classifier.j2 ===

(saescoping) 4gate@Mac SAEScoping %
```