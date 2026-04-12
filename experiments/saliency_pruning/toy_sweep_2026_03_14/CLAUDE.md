# Development principles

- Each module is both exposed as a standalone CLI and as a library/method that can be called. The CLI is a trivial shim that puts in reasonable defaults.
- A clear boundary exists between what is private (underscore) and what is public. All public things are meant to be used/must be exported and used elsewhere. It should be very clear for each module what it exports and the export should be very simple and easy to reason-about in a self-contained fashion.
- All inputs/outputs are via files and/or are serializeable
- All structured data is JSON
- Everything must have an integration or unit test
- All experiments should be stored with numbers and one folder per experiment (ideally with one file to run it inside and one file to store the results)
- All trainers work for any model and any any SAE type from SAELens or eleuther's Sparsify library. They may fail gracefully, but if something is not failing gracefully with a "not implemented error" then it MUST work (same idea for models). This way we will support Gemma3 later.
- Always use `click` over `argparse`
- Always use type annotations
- Always read `README.md` in the directory you are working in to understand the context and goals of this.
- Use `pydantic` to define schemas
- Where possible integration tests or unit tests (we use more integration because research codebase but still want to keep it clean) should be possible to run on CPU. A common way to do this is to have one utility that creates a really small model (i.e. by deleting layers).
- Put integration tests in `tests/` and unit tests in `tests/unit/`. Never put integration tests in the same files they are testing.
- Never define functions inside other functions. Always pick a clean interface for proper, extensible code.
- Never import inside functions or anywhere other than the top of the module. Always import at the module level at the top of the file.

PYTHONPATH should always be set at this folder.