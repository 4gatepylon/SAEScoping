# SAE Scoping
Minimal viable version of 2024/2025 SAE Scoping work. The desired use-case is that you do:
1. Setup your environment with `conda create -n saescoping python=3.12 -y`
2. Install this and its depndencies with `pip install -e .`.

The folder-structure (etc...) should be self-explanatory. The pip-installable is inside `sae_scoping/` whereas `experiments` and other such folders are meant for one-off scripts (etc...) that are NOT libraries but may be important for results from the paper.

This supports the following basic workflows:
1. Training
    - Support running inference on Gemma-2-9b-it with Gemmascope SAEs to get frequency maps of their neurons.
    - Support "Scoping down" Gemma-2-9b-it by applying the SAE pruned to include only frequently-firing features in-domain.
    - Support SFT on Gemma-2-9b-it to get a baseline for performance on specialized domain (we use biology Q/A).
    - Support SFT on the layers after an SAE in a "Scoped" Gemma-2-9b-it instance to recover performance ("recovery training").
    - Support SFT on a pre or post recovery scoped model (with SAE) for the purposes of trying to elicit OOD capabilities.
    - EDA for the stuff above in `experiments/`
2. Evaluation
    - Use `dspy` module `Evaluator` to evaluate a model on a dataset.
    - Use a custom `Adapter` for dspy to extract verifaible responses (to check for correctness) from the model's output.
    - Support batched evaluation of correctness for verifiable datasets/responses using `dspy`'s `Evaluate`. Make sure you are using my fork of `dspy`: https://github.com/4gatepylon/dspy
3. Adversarial optimization with GEPA. Make sure you are using my branch of `gepa`: https://github.com/4gatepylon/gepa. We define in-domain and out-of-domain combinations of datasets and support optimizing out-of-domain 
    - Note that we include a custom chat template with a system prompt based on 
4. SAE-supporting server. We have an OpenAI-API compatible server that we use to host regular models as well as scoped (SAE-enhanced) models. It is built using huggingface and will natively batch requests to make inference much, much faster. We use it for GEPA and for evaluation.
5. 