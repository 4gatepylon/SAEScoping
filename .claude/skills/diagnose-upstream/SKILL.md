---
name: diagnose-upstream
description: |
  Use BEFORE applying any library/config "fix" whose reason is not yet understood —
  setting a flag (remove_unused_columns=False, strict=False, check_same_thread=False),
  adding a kwarg, swapping a class, wrapping a call in try/except. TRIGGER when:
  the traceback ends inside a third-party package (especially transformers, trl,
  accelerate, datasets, peft, torch, pydantic, click); the error message itself
  proposes a flag value; you are about to add a kwarg whose semantics you have only
  inferred from the error text; the user nudges with "are we in the right codepath?"
  / "why is this happening?" / "doesn't this suggest the dispatch is wrong?"; the
  reason a proposed fix would work is not yet established.

  Core rule: every fix must be UNDERSTOOD. If you cannot state, in one sentence, why
  the change works in terms of upstream behavior, you have not found the bug yet —
  only something that suppresses the symptom.
user-invocable: true
---

# diagnose-upstream

## Core principle

**Every fix must be understood.** An error tells you WHAT broke, not WHY you are in
the function that broke. Before patching the failing line, prove the function is the
right one to be in. If you cannot explain why your change works in terms of upstream
code, keep diagnosing — do not patch.

## When to invoke

Apply this skill BEFORE making any change that:

- Sets a flag/option to a value whose upstream effect you have not read
  (`remove_unused_columns=False`, `strict=False`, `verify=False`,
  `check_same_thread=False`, `safe_serialization=False`).
- Adds a kwarg whose semantics you only inferred from the error message.
- Wraps a call in `try/except` to swallow a third-party exception.
- "Just makes the error go away" by switching a class, dtype, or path.

Trigger conditions (any one is enough):

- **Traceback ends in a third-party package.** Especially the HuggingFace stack
  (`transformers`, `trl`, `accelerate`, `datasets`, `tokenizers`, `peft`,
  `safetensors`), but also `torch`, `pydantic`, `click`, `sqlalchemy`, `pytest`.
- **The error message itself proposes a fix** ("you may need to set X=Y", "consider
  passing Z").
- **User pushes back on routing or reasoning** ("are we in the right codepath?",
  "why is this happening at all?", "the error suggests something upstream is
  wrong"). Treat user routing-pushback as a high-signal trigger; it usually
  outranks the error message's literal suggestion.
- **You cannot state in one sentence why a proposed fix works.** If the
  justification is "the error said so" or "this kwarg sounds related," stop.

## What to do

1. **Read upstream, form a hypothesis from the source — not from the error text.**
   Open the failing function in the env's `site-packages/<pkg>/...`. Identify the
   branch you landed in. Trace upward: what dispatched into this function, what
   set the attribute that controls the branch, who passed (or failed to pass) the
   argument that drove the dispatch.
2. **Question the routing, not just the failing line.** A library-internal error
   can mean the dispatch sent you to the wrong function, not that the function
   needs a workaround. A `KeyError` inside a vision collator may mean the vision
   collator was wrongly selected, not that you should add the missing key.
3. **State your hypothesis to the user before patching.** "I think we landed in
   branch X because of Y; the fix the error suggests would do Z; I want to verify
   that's the right cause first." This lets the user redirect early. The narration
   itself often surfaces the gap in your understanding.
4. **Debug prints are allowed with explicit user permission.** Use a unique
   searchable tag (`XXXAGENTDEBUG`, `__CLAUDE_PROBE__`) so you can grep-and-remove
   all of them in one shot. Always remove before declaring done. Never use a tag
   that already appears in the codebase.
5. **No patch without explanation.** When you propose the change, state in one
   sentence what upstream behavior makes the fix correct. If you cannot, you have
   not found the bug.
6. **Close the loop with the user.** After the fix lands, ask:
   > "Want me to add this case as a worked example to the diagnose-upstream skill?
   > I'd add: <one-line description of the dispatch trap and how to recognize it>."

   Append on confirmation. The skill improves only if real cases get added.

## Worked example: trl SFTTrainer model-class dispatch

Saved here as a vivid pattern; pattern-match against this when you see something
shaped similarly.

**Symptom.** Training a multimodal-capable model (`google/gemma-3-12b-it`) on a
text-only QA dataset, `SFTTrainer.train()` crashed with
`KeyError: 'images'` inside `trl/trainer/sft_trainer.py:_collate_language_modeling`.
The dataset had a `text` column, no `images`.

**Wrong fix (band-aid).** "The error suggests `remove_unused_columns=False`" — that
kwarg patches a *different* column-stripping symptom one layer up
(`_remove_unused_columns` against trl's `_signature_columns = ["messages", "prompt",
"completion", "images"]`). It does not address why the vision collator was selected
in the first place; the next failure was waiting one level deeper.

**Real cause.** trl's `SFTTrainer.__init__` decides `self._is_vlm` by
`isinstance(processing_class, ProcessorMixin)`, where `processing_class` is
auto-loaded via `AutoProcessor.from_pretrained(model_id)` if the caller does not
pass one explicitly. For `gemma-3-12b-it` this returns a multimodal
`Gemma3Processor` (a `ProcessorMixin`), so `_is_vlm = True`. That single boolean
then routes into:

- the VLM signature columns (`["messages", "prompt", "completion", "images"]`);
- the `_prepare_dataset` skip (no per-row pre-tokenization);
- the `DataCollatorForVisionLanguageModeling` (which unconditionally reads
  `example["images"]`).

None of the three branches matches a text-only QA dataset. The dataset's `text`
column was never the problem — it was the model-class-based dispatch.

**Real fix.** Forward the existing tokenizer explicitly:
`SFTTrainer(..., processing_class=tokenizer)`. That sets `_is_vlm = False`,
selecting the text-only collator and pre-tokenization path. The `text` column flows
through `_prepare_dataset` into `input_ids` naturally; no other workaround needed.

**Pattern to recognize.**

- Library does class-based dispatch (`isinstance(...)`, `type(...).__name__`,
  `hasattr(...)`).
- The dispatching object was auto-derived by the library, not passed by the caller.
- The fix lives upstream of the failing line: pass the right object in, do not
  patch around the symptom.

## Anti-patterns

- "The error message says set X=False, so I'll do that" — without reading what X
  controls. The message says what failed, not what fixes the cause.
- "I'll add `try/except` so the error doesn't propagate" — this hides the bug and
  almost always creates a worse silent failure later.
- "I'll keep adding kwargs until the error stops" — if each kwarg is not justified
  by upstream code, the fix is luck-based and brittle.
- "It works on my machine / for the other model, so it's fine" — different model
  classes can take different code paths through the same function (see worked
  example: gemma-2 vs gemma-3 in trl).

## Adding new worked examples

Each time the user approves a new case at step (6), append a section to this file
under `## Worked examples` (rename the singular section once a second example
exists). Each new section should follow the four-part shape:

- **Symptom.** What the user (or you) observed.
- **Wrong fix (band-aid).** The plausible patch the error message points at, and
  why it would only mask the symptom.
- **Real cause.** The upstream dispatch / state that put you on the wrong path.
- **Real fix.** The minimum change that addresses the cause, plus a one-line
  pattern-recognition rule.

Keep examples short — a vivid 10-line entry beats a complete write-up.
