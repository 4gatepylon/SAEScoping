# Objectives
1. Finish the paper as soon as possible and put a draft on Openreview. This will have no baselines, just our past results and our current results with GEPA.
2. Clean up the code as soon as possible and as much as possible so that in the next 48 hours or so I can continually iterate without having to be very awake/aware/good at coding. I want things where I basically just tune parameters in these last couple days.

This is for ICML Draft 1. We will have more drafts with more stuff later on (i.e. baselines etc...).

# Planned Results
- **Result 1:** Trojann'ed models from competition: https://huggingface.co/ethz-spylab/poisoned_generation_trojan1. For each one I have biology utility and trojan safety. I do this before the SAE + training and after the SAE + training. These are sparsify TopK SAEs. I will need to read `ScopeBench` and re-integrate just the relevant bits into `SAEScoping`. I will compare utility with SFT, vanilla, and prompting. For good measure I should also scope on CamelAI physics and CamelAI chemistry.
    - **What needs to be done:**
        - Make sure to leverage a train and validation set for CamelAI Biology, Physics, Chemistry
        - Train 5 SAEs using the exact same hyperparamaters as the first (15 technically, for the 3 datasets)
        - Train the layers after the SAE(s) on each of the 5 models (on the in-domain dataset). I will need to acquire synthetic data for each dataset.
        - Make sure I have SFT checkpoints for each of the 5 models
        - Store SFT checkpoint generations with and without trojans on benign (in-domain) and malicious (out-of-domain) data (4 combinations)
        - Store prompt-engineered generations with and without trojans on benign (in-domain) and malicious (out-of-domain) data (4 combinations)
        - Store SAE-scoped generations with and without trojans on benign (in-domain) and malicious (out-of-domain) data (4 combinations)
        - Store recovery-trained generations with and without trojans on benign (in-domain) and malicious (out-of-domain) data (4 combinations)
        - Grade all the generations and validate the judge on different subjects (less compute intensive).
- **Result 2:** Gemma2 model can be scoped to biology and is broadly poor out-of-domain even after GEPA prompt optimization. We will showcase (1) good performance in-domain via the plot of the sweep (this motivates using `1e-4`), (2) bad perforamnce out-of-domain via the broad MMLU evaluation, (3) bad performance out-of-domain even after applying GEPA on: `{math, cyber, chemistry}`.
    - **What needs to be done:**
        - Add support for servers to change which model they use (just support one model at a time, one port, one GPU + a `/change_model` API that allows you to provide the same arguments you used to launch in a pydantic schema and to start new models;  this is only going to support Gemma2 for now)
        - Define a dataset format for verifiable dataset. These will always have the following items:
            - Request
            - Response (after the LLM generates)
            - Golden answer
            - For the dataset, there is a way of juding `(request, response, golden answer)`, either using a judge or using a regex to extract form the golden answer and the response and comparing the exraction (i.e. the GSM8K format or `\boxed{answer}` are examples).
        - Add support for our existing datasets with caching for test/validation/train split determinism under the format:
            - All WMDP multiplce choice
            - Cyber MCQ: SecQA, Cybermetric (and WMDP obviously from ^, MMLU computer security from v, etc...)
            - MMLU: all MMLU datasets should be supported
            - Science CamelAI + Judge: all CamelAI + any judge that takes in `(user_request, assistant_response, golden_response)` should be supported on CamelAI datasets `(biology, chemistry, physics)`. These should also enable mixing in of `MegaScience/MegaScience` for the corresponding field.
            - Math: GSM8K, NuminaMath, possibly some from https://claude.ai/share/e82207e0-e8e4-49d3-abb2-d776ec1a44ff
    - Create a custom adaptor like https://claude.ai/share/2ef9bc8a-00c3-4085-8e17-7a31016601e4. This should enable us to canonicalize the response format and avoid so many parse errors. Integrate the adaptor into our DSPY optimization pipeline/code/library.
    - Add support for a batched `run_gepa.py` script which will multiprocess with one target LLM server per worker (so instead of `gpu-ids` for example you would use `--ports` and launch one worker per port, trying to minimize the number of model changes per server)
    - Support one evaluation script ONLY for ALL the datasets using the clean abstraction. It should operate on a server, file, or (local) hf generator. If we can turn this into just a call to `Evaluate` in `dspy`  that is ideal, but if not, it's not the end of the world. We can improve the interface tomorrow, worst-case.
    - Troubleshoot why GEPA does not improve and/or does improve in other cases
    - Run a batched GEPA optimization on all of MMLU (or at least all of chemistry, all of computer security, all of biology, all of math, a few others); (for vanilla, scoped)
    - Run batched GEPA optimization on all the cyber datasets (for vanilla, scoped); run an evaluation on the original and optimized prompts
    - Run batched GEPA optimization on all of WMDP (for vanilla, scoped); run an evaluation on the original and optimized prompts
    - Run batched GEPA optimization on all of Math (for vanilla, scoped); run an evaluation on the original and optimized prompts

We will put all of the result-2 stuff (which is for GEPA) inside the `experiments_gemma_gepa` folder. We may delete some of the python scripts and jupyter notebooks in `experiments/` to replace them with this. We will put the result-1 stuff in `experiments_llama_trojans/`.