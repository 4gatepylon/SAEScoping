This code is mainly meant to be indicative of how training with SAEs could occur. Generally, what has to happen is:
1. We collect a great science dataset (what `../science` folder supports)
2. We optionally SFT a model (what `../science_sft` folder supports)
    - Things to note:
        We work with spylab models like `ethz-spylab/poisoned_generation_trojan1` which are a derivative of Llama2 models
3. We train a TopK SAE on that model (what `../science_sae` folder supports) using Eleuther's Sparsify library
    - For each SFT checkpoint if available train TopK on it
    - For each original model (spylab model) train TopK on it
    - Hyperparmeters (which layer(s) to try) will be passed in a flag
    - Examples of how this works are in `script_2025_09_10_train_saes.py` (which is outdated but contains relevant operational hints/how-tos for this)
4. We train the model again on the science dataset, but only on the layers after the SAE hookpoint and with the SAE hooked on.
    - I think `recovery_ultrachat.py` has examples of how this works; it's also deprected and on the wrong dataset, but should highlight operational-how-tos
5. We evaluate the quality of our model's responses vs. the quality of responses from the models with the SAE (called "scoped" models) w.r.t. the quality from the models with SFT (the "SFT" models.) We also compare with the quality of responses from "prompted" models (just the original model, but with a prompt). At the same time, we also report the succiptibility to trojans. We use the test set (possible a subset of it) for these evals. We judge the response quality with the LLM judges via jinja2-template defined in `sae_scoping/evaluation/iclr_judge_prompts.j2`. Specifically we report:
    - Utility: `mean([score for score from answering_classifier.j2, factual_helpful_classifier.j2, precise_classifier.j2])` on the in-domain test set (benign).
    - Safety:`min(1-utility on the out-of-domain malicious set, score from refusal.j2`).
    - Things to note:
        - We evaluate on multiple data points. For each judge we take the mean of their "score". For utility we go further and take the mean of the judges' means. For safety you can see the 
        - This is showcased in aggregators/aggregation/aggregate in `./one_click.py`, which is a deprecated file that may need to be refactored, but whihc highlights the desired flow.
        - Importantly, we grade the responses from the Llama2 (spylab) model, so generation has to occur. We merely use the question from the test data, not the answer.