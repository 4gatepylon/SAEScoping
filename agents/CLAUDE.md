# Development principles

- Each module is both exposed as a standalone CLI and as a library/method that can be called. The CLI is a trivial shim that puts in reasonable defaults.
- **Output caching / --force pattern:** Any script or command that writes output to files (safetensors, JSON, etc.) must check whether the output already exists before running. If it does, skip by default and print a message. Always expose a `--force` flag to override this and re-run unconditionally. This applies to both single-run and batch commands. The goal is that re-running a batch after an interruption is safe and fast: already-completed work is skipped automatically. If a cached file is used, a print statement logs it. If a cached file is overwritten, then a print statement logs it as well.
- A clear boundary exists between what is private (underscore) and what is public. All public things are meant to be used/must be exported and used elsewhere. It should be very clear for each module what it exports and the export should be very simple and easy to reason-about in a self-contained fashion.
- All inputs/outputs are via files and/or are serializeable
- All structured data is JSON
- Everything must have an integration or unit test
- All experiments should be stored with numbers and one folder per experiment (ideally with one file to run it inside and one file to store the results)
- Always use `click` over `argparse`
- Always use type annotations
- Always read `README.md` in the directory you are working in to understand the context and goals of this.
- Use `pydantic` to define schemas
- Where possible integration tests or unit tests (we use more integration because research codebase but still want to keep it clean) should be possible to run on CPU. A common way to do this is to have one utility that creates a really small model (i.e. by deleting layers).
- Put integration tests in `tests/` and unit tests in `tests/unit/`. Never put integration tests in the same files they are testing.
- Never define functions inside other functions. Always pick a clean interface for proper, extensible code.
- Never import inside functions or anywhere other than the top of the module. Always import at the module level at the top of the file.
- Never use relative imports (`from .foo import ...` or `from ..foo import ...`). Always use absolute imports rooted at the `PYTHONPATH` root.
- All unit tests messages that denote passing use "✅" and all that denote failure use "❌" as the first character of the message. For warning messages always use "⚠️" as the first character of the message. For tests whose outcomes are unclear use "❓" as the first character of the message.
- The best way to highlight the uncertainty of how you want control flow in code to work, is by asking the user whether your pseudocode (which should be succinct) is the actual form of the implementation.
- Always ask for permission to run integration tests, but you may run unit tests (if they exist) by doing `pytest tests/unit`.
- If the change would break existing callers (i.e., remove or rename a public interface), surface this to the user first. If the change merely adds a new option or makes the module more compliant without removing anything, just do it.

# Testing philosophy

**Unit tests** should focus on the **exported/public interface** of a component and exercise it as a whole. Test core library functionality — the computations, transformations, and logic that the module exposes. Only drill into internal/private details when they contain tricky logic that warrants direct coverage.

**Do not test UI/CLI surface choices** like flag names (`--force`, `--output-dir`), default values, or argument parsing behavior. These change frequently during development and are validated naturally when the developer commits and runs the tool. Testing them creates high-maintenance tests that break on harmless renames and don't catch real bugs.

**Integration tests** exercise multiple components working together end-to-end.

Unit tests must cover the breadth of possible inputs/outputs for the public interface. Integration tests may just cover the most common path. Never skip tests. Never change tests to make passing easier. Set up simple tests that catch real issues. Never let issues pass through silently.

## Examples

**Good unit test** — tests the exported function's core behavior:
```python
def test_compute_saliency_scores_returns_correct_shape():
    """Tests the public compute_saliency_scores with a small model."""
    model = make_tiny_model()
    scores = compute_saliency_scores(model, dataset, mode="gradient")
    assert scores.keys() == set(model.state_dict().keys())
    for name, s in scores.items():
        assert s.shape == model.state_dict()[name].shape

def test_compute_saliency_scores_gradient_vs_taylor():
    """Tests that two supported modes produce different results."""
    model = make_tiny_model()
    grad_scores = compute_saliency_scores(model, dataset, mode="gradient")
    taylor_scores = compute_saliency_scores(model, dataset, mode="taylor")
    assert not all(torch.equal(grad_scores[k], taylor_scores[k]) for k in grad_scores)
```

**Good unit test** — tests tricky internal logic that warrants direct coverage:
```python
def test_ema_accumulation_decays_old_values():
    """The EMA helper is private but has subtle math — worth testing directly."""
    acc = _ema_accumulate(prev=torch.tensor(1.0), new=torch.tensor(0.0), beta=0.9)
    assert acc == pytest.approx(0.9)
```

**Bad unit test** — tests CLI flag names (high maintenance, low value):
```python
def test_force_flag_overwrites_existing_file():
    """Don't test this — --force is a UI choice that may be renamed to --overwrite."""
    ...

def test_default_output_dir_is_cwd():
    """Don't test this — default paths are a UX decision, not core logic."""
    ...
```

**Good integration test** — tests multiple components together:
```python
def test_saliency_map_then_prune_reduces_nonzero_params():
    """End-to-end: compute saliency -> prune at 50% -> verify sparsity."""
    model = make_tiny_model()
    scores = compute_saliency_scores(model, dataset, mode="gradient")
    apply_pruning(model, scores, sparsity=0.5)
    total = sum(p.numel() for p in model.parameters())
    nonzero = sum((p != 0).sum().item() for p in model.parameters())
    assert nonzero / total == pytest.approx(0.5, abs=0.05)
```

Each component must be individually testable via CLI and must have a small integration test. Code that cannot be tested will lead to compounding errors regardless of developer skill.
