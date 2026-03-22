# Development principles

- Each module is both exposed as a standalone CLI and as a library/method that can be called. The CLI is a trivial shim that puts in reasonable defaults.
- **Output caching / --force pattern:** Any script or command that writes output to files (safetensors, JSON, etc.) must check whether the output already exists before running. If it does, skip by default and print a message. Always expose a `--force` flag to override this and re-run unconditionally. This applies to both single-run and batch commands. The goal is that re-running a batch after an interruption is safe and fast: already-completed work is skipped automatically. If a cached file is used, a print statement logs it. If a cached file is overwritten, then a print statement logs it as well.
- A clear boundary exists between what is private (underscore) and what is public. All public things are meant to be used/must be exported and used elsewhere. It should be very clear for each module what it exports and the export should be very simple and easy to reason-about in a self-contained fashion.
- All inputs/outputs are via files and/or are serializeable
- All structured data is JSON
- Everything must have an integration or unit test
- All experiments should be stored with numbers and one folder per experiment (ideally with one file to run it inside and one file to store the results)
- All trainers work for any model and any any SAE type from SAELens or eleuther's Sparsify library. They may fail gracefully, but if something is not failing gracefully with a "not implemented error" then it MUST work (same idea for models). This way we will support Gemma3 later.
- **WandB project naming:** Projects are named `saescoping--pruning--{script_name_without_py}`. Examples: `gradients_map.py` → `saescoping--pruning--gradients_map`, `sweep_eval_temp.py` → `saescoping--pruning--sweep_eval_temp`, `prune_and_maybe_recover.py` → `saescoping--pruning--prune_and_maybe_recover`, `grade_chats.py` → `saescoping--pruning--grade_chats`. Run names must be date-prefixed and include the relevant distinguishing parameters, e.g. `2026-03-20_ema_gradient_abs_biology` or `2026-03-19_ema_taylor`.
- Always use `click` over `argparse`
- Always use type annotations
- Always read `README.md` in the directory you are working in to understand the context and goals of this.
- Use `pydantic` to define schemas
- Where possible integration tests or unit tests (we use more integration because research codebase but still want to keep it clean) should be possible to run on CPU. A common way to do this is to have one utility that creates a really small model (i.e. by deleting layers).
- Put integration tests in `tests/` and unit tests in `tests/unit/`. Never put integration tests in the same files they are testing.
- Never define functions inside other functions. Always pick a clean interface for proper, extensible code.
- Never import inside functions or anywhere other than the top of the module. Always import at the module level at the top of the file.
- Never use relative imports (`from .foo import ...` or `from ..foo import ...`). Always use absolute imports rooted at `toy_sweep_2026_03_14/` (the `PYTHONPATH` root), e.g. `from gradients_map.utils import save_saliency_map`.
- All unit tests messages that denote passing use "✅" and all that denote failure use "❌" as the first character of the message. For warning messages always use "⚠️" as the first character of the message. For tests whose outcomes are unclear use "❓" as the first character of the message.
- The best way to highlight the uncertainty of how you want control flow in code to work, is by asking the user whether your pseudocode (which should be succinct) is the actual form of the implementation.
- Always ask for permission to run integration tests, but you may run unit tests (if they exist) by doing `pytest tests/unit`.
- Your unit tests must always test the breadth of possible behavior/inputs/outputs. Your integration tests may just test the most common path. If you make tests, you should always make sure to try and set them up to surface likely bugs and edge cases. Never skip tests. Never change tests to make passing easier. Set up simple tests that catch issues. Never let issues pass through silently.
- Write polymorphic code. Before implementing, always consider what possible abstractions with what possible interfaces may end up being useful in the future (i.e. if there end up being memory or runtime constraints that are not yet the case or if we want to change to a different method). For example, when you need to cache and then re-load weights, you should have a weights handle as opposed to the weights themselves (that way you could support HF, support CPU caching, support GPU or disk caching, etc...). For example, when writing a hyperparameter sweep algorithm, it should be possible to put different optimizers inside the slots.

PYTHONPATH should always be set to `experiments/saliency_pruning/toy_sweep_2026_03_14/` (this folder, relative to the repo root). Use the `saescoping` conda environment.

# FAQ

Over the course of development other developers, agents, and collaborators have asked some common questions. Here are the answers:

**TODO(adrianoh):** Add support for the Jinja2 template for `gemma2-9b-it` that includes a system prompt.

---

**Q1. Dataset interface: Should `gradients_map.py` and `prune_and_maybe_recover.py` accept HuggingFace `Dataset` objects or plain lists of dicts?**

Use HuggingFace `Dataset` objects with `"question"` and `"answer"` keys. Create a `dataset_utils.py` to cover the repeatable workflow of loading the dataset, converting to OpenAI message format, and possibly converting to text/tokens via the chat template. Use the gemma2 chat template at `prompts/gemma2_chat_template_system_prompt.j2`. `StemQAMixture` uses exactly these column names. Include a Pydantic validator that checks for this schema so that swapping datasets produces a controlled failure rather than a silent bug.

---

**Q2. Which parameters should pruning apply to — all params including embeddings and LM head, or only attention/MLP weight matrices?**

All parameters by default, but it should be possible to pass in a regex that defines which parameters to prune (the regex matches the parameter names we DO want; all others are skipped).

---

**Q3. Should the gradient/loss computation cover only answer tokens (masking the question/prompt) or the full sequence?**

This should be a toggle. Loss should be computed over batches via a batch-size argument. The gradient accumulation uses EMA (exponential moving average) with a configurable beta, as implemented in `prune.py`'s `GradCollectTrainer`. The saliency criterion (gradient magnitude vs. Taylor first-order) is applied after accumulation.

---

**Q4. What is the pruning saliency criterion — `|grad|`, `|grad * weight|` (Taylor first-order), or something else?**

Both should be supported as a toggle: plain gradient magnitude (`|grad|`) and Taylor first-order (`|grad * weight|`).

---

**Q5. Recovery SFT: Manual training loop or `transformers.Trainer`?**

Use `transformers.Trainer`. Define a self-contained, modular callback that periodically generates from the model, grades the outputs, and signals early stopping when quality crosses a threshold. The callback must not modify model weights — weight zeroing happens once before recovery begins, and the callback only decides when to stop. This callback should be designed to slot into any training loop.

---

**Q6. Is `generate_chats.py` just a thin bridge that formats question/answer dicts as OpenAI messages and runs `HFGenerator.generate()` on the question portion?**

Yes. The caller is responsible for passing in the dataset — `generate_chats.py` should not handle dataset loading itself.

---

**Q7. The current version of a module does not do what the specification says. Should I change it?**

If the change would break existing callers (i.e., remove or rename a public interface), surface this to the user first. If the change merely adds a new option or makes the module more compliant without removing anything, just do it.

---

**Q8. What order should components be implemented in?**

Always follow dependency order: implement and pass integration tests for small, upstream components before building the larger ones that depend on them. The intended order is: `dataset_utils.py` → `gradients_map.py` → `prune.py` (weight zeroing) → `generate_chats.py` → `prune_and_maybe_recover.py` → sweep. Do not start a downstream component until the upstream one has a passing integration test.

---

**General reminder:** Each component must be individually testable via CLI and must have a small integration test. Code that cannot be tested will lead to compounding errors regardless of developer skill.