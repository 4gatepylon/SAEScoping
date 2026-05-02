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
4. **Debug prints are allowed with explicit user permission.** See the
   "Debug-print discipline" section below for the full protocol — pick a
   uniqueness-checked tag (UUID-fallback if needed), insert single-line
   unambiguous A/B prints only, find-and-remove via the tag before declaring
   done. The tag is what makes the cleanup trivial; pick it carefully.
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

## Debug-print discipline

When the user has approved adding instrumentation (step 4 above), the goal is to
insert **single-line, unambiguous A/B prints** into the running code, observe one
run, then mechanically revert every probe via its unique tag. No multi-line
blocks, no "let me trace through", no prints that interleave with normal output
in ways that confuse the user. One line per print, each prefixed with the same
unique tag.

The most common target is **third-party library code under `site-packages/`**
(e.g. tracing inside `transformers/trl/torch` to verify which branch fires).
That code is not under version control on the user's side, so there's no `git
diff` safety net — the cleanup discipline below is the only thing that
guarantees you didn't leave instrumentation in someone else's installed
package. "Commit / PR / branch" concepts do not apply to those edits; the goal
is byte-for-byte restoration of the original file. The same discipline applies
verbatim if you ever instrument the user's own repo (rare; see step 7).

### 1. Pick a tag — and verify it is unique to your insertion

A "tag" is a string like `__CLAUDE_PROBE__` or `XXXAGENTDEBUG_2024A` that you
prepend to every print. The tag is what makes cleanup mechanical. **Before you
insert anything, grep the codebase to confirm the tag does not already appear**
(in any file the user might consider in scope — typically the repo, but also
relevant config and test files):

```bash
# from repo root, ripgrep is fastest; -F = literal, no regex surprises
rg -F '__CLAUDE_PROBE__' .
# or with grep:
grep -rnF '__CLAUDE_PROBE__' . --include='*.py' --include='*.yaml' --include='*.json'
```

If the grep returns ANY hits, do not use that tag — the cleanup step would
delete or match unrelated lines. Pick a different tag and re-check.

### 2. Use a UUID when in doubt

If you suspect the candidate tag (e.g. `XXXAGENTDEBUG`) is generic enough that
it might collide, or if multiple instrumentation passes are happening in
parallel, generate a UUID and embed it in the tag:

```bash
# bash one-liner; produces e.g. claude_probe_e9f2c81a
python -c "import uuid; print(f'claude_probe_{uuid.uuid4().hex[:8]}')"
```

Then use that exact string as your tag. UUID prefixes make collisions
effectively impossible and let you run two instrumentation rounds in parallel
without their cleanup steps interfering.

### 3. Single-line prints, A/B style

Each print should:

- Be one physical line (no continuations, no triple-quoted strings).
- Start with the tag, then carry the minimum data needed for the A/B
  comparison: a label + a value or boolean.
- Not depend on local state that might not be in scope (e.g. `f"{maybe_undef!r}"`
  will crash if the variable is missing — gate with `try/except: pass` only if
  the user okayed the noise).

Examples:

```python
print(f"__CLAUDE_PROBE_e9f2c81a__ is_vlm={self._is_vlm}")
print(f"__CLAUDE_PROBE_e9f2c81a__ collator={type(self.data_collator).__name__}")
print(f"__CLAUDE_PROBE_e9f2c81a__ signature_columns={self._signature_columns}")
```

Three lines, three independent A/B observations, all greppable by one tag.

### 4. Find what fired

After the run, grep the log (or stdout capture) by the tag:

```bash
grep -F '__CLAUDE_PROBE_e9f2c81a__' run.log
# tally:
grep -cF '__CLAUDE_PROBE_e9f2c81a__' run.log
```

To find the source lines you inserted (in case you need to tweak before the next
A/B run):

```bash
rg -nF '__CLAUDE_PROBE_e9f2c81a__' .
# or:
grep -rnF '__CLAUDE_PROBE_e9f2c81a__' . --include='*.py'
```

### 5. Remove every probe before declaring done

Do this in one sweep against the source tree. The tag is what guarantees
correctness: if `rg -F '<tag>' .` returns zero hits, every probe is gone.

```bash
# Inspect the lines first (sanity):
rg -nF '__CLAUDE_PROBE_e9f2c81a__' .

# Then delete in-place. Pick ONE method:

# (a) git checkout each touched file (cleanest if you haven't otherwise edited):
git diff --name-only | xargs -I{} git checkout HEAD -- {}

# (b) sed -i to drop matching lines (works on cross-edited files):
rg -lF '__CLAUDE_PROBE_e9f2c81a__' . \
  | xargs sed -i '/__CLAUDE_PROBE_e9f2c81a__/d'

# Verify zero hits remain:
rg -F '__CLAUDE_PROBE_e9f2c81a__' . && echo 'STILL PRESENT' || echo 'clean'
```

If `rg` returns hits after deletion, stop and inspect — do not declare done.

### 6. Verify the cleanup did not change anything *else*

Grep-zero-hits proves the tag is gone, but it does NOT prove the surrounding
code is identical to its pre-instrumentation state. A misfired `sed -i` can
delete adjacent lines, mangle indentation, or strip a closing brace that
happened to be on the same physical line as a probe. Add a structural check.

**Option (a) — `git diff` against HEAD if all your instrumentation went into
already-tracked files:**

```bash
# If you started from a clean tree, after cleanup `git diff` should be empty
# (modulo whatever non-probe changes you intentionally made, which you should
# review by eye).
git -c color.ui=never diff --stat HEAD
git -c color.ui=never diff HEAD   # full review
```

If you also intended non-probe edits, the cleanup should leave only those —
inspect each remaining hunk and confirm none of them belongs to a probe.

**Option (b1) — `cp -r` mirror to a tempdir, `diff -r` afterwards:**

The most informative check when you need to *see* any drift, not just detect
it. Snapshot the target subtree to `$TMPDIR` before instrumentation, then
`diff -r` after cleanup. The diff output tells you exactly which files /
hunks changed, so you can decide whether each change was intentional (a real
fix you made alongside the probe) or accidental (cleanup damage).

```bash
TAG=__CLAUDE_PROBE_e9f2c81a__
TARGET_DIR=path/to/subtree
MIRROR=$TMPDIR/claude-probe-mirror-${TAG}

# BEFORE inserting probes:
cp -r "$TARGET_DIR" "$MIRROR"

# ... insert, run, clean up per steps (1)–(5) ...

# AFTER cleanup — recursive diff; expect zero output:
diff -r "$TARGET_DIR" "$MIRROR"
# Or, file list only:
diff -rq "$TARGET_DIR" "$MIRROR"
```

If the diff is non-empty, every hunk should correspond to an intentional
non-probe edit — eyeball each one. When done, `rm -rf "$MIRROR"`.

**Option (b2) — hash-before / hash-after of the touched tree:**

This is the most defensive check, useful when (i) you instrumented files that
were already dirty, (ii) you're not in a git repo, or (iii) you simply want
a single-line equality check that does not depend on git semantics.

```bash
# BEFORE inserting any probes — capture a hash of the files you intend to
# touch (or the whole subtree):
TAG=__CLAUDE_PROBE_e9f2c81a__
TARGET_DIR=path/to/subtree
HASH_BEFORE=$(tar -cf - "$TARGET_DIR" 2>/dev/null | sha256sum | cut -d' ' -f1)
echo "$HASH_BEFORE" > /tmp/claude-probe-${TAG}.before

# ... insert probes, run experiment, remove probes per step (5) ...

# AFTER cleanup — recompute and compare:
HASH_AFTER=$(tar -cf - "$TARGET_DIR" 2>/dev/null | sha256sum | cut -d' ' -f1)
[ "$HASH_AFTER" = "$(cat /tmp/claude-probe-${TAG}.before)" ] \
  && echo "tree unchanged — cleanup verified" \
  || { echo "tree differs from pre-probe state — DO NOT COMMIT"; \
       diff <(cat /tmp/claude-probe-${TAG}.before) <(echo "$HASH_AFTER"); }
```

If the hashes diverge, walk the diff (`git diff` if applicable) and either
restore the file (`git checkout HEAD -- <file>`) or hand-correct. Do not
declare done while the hashes disagree — that is exactly the failure mode
this step is designed to catch.

**Option (c) — quick sanity test.** If the project has a fast smoke (a single
unit test, `python -c "import <pkg>"`, or a known-passing CLI invocation),
re-run it after cleanup. A green smoke is independent evidence that the
cleanup did not break anything.

### 7. Clean up temp artifacts (mirrors, hash files)

The mirror dir from option (b1) and the hash file from option (b2) live in
`$TMPDIR` (which on this host resolves under `/tmp/claude-…/`). They are
helpful for the duration of the diagnosis but become clutter the moment the
probe is removed. Machines here do not reboot frequently, so `/tmp` does NOT
get cleared automatically — orphaned mirrors accumulate over weeks of
diagnoses and eventually cost real disk.

**Convention.** Always name temp artifacts with a stable prefix
(`claude-probe-…`, `claude-probe-mirror-…`) and place them under `$TMPDIR`
(never `/tmp` directly — `$TMPDIR` is the agent-scoped subdir and is the
cleanup-safe target). The unique tag from step (1) goes into the filename so
multiple diagnoses don't collide.

```bash
TAG=__CLAUDE_PROBE_e9f2c81a__
MIRROR=$TMPDIR/claude-probe-mirror-${TAG}
HASHFILE=$TMPDIR/claude-probe-hash-${TAG}.before
```

**At the end of a successful diagnosis, delete them immediately:**

```bash
rm -rf "$MIRROR" "$HASHFILE"
```

**Detect and clean up old / orphaned ones** (from prior sessions that may not
have cleaned up — yours, or some other agent run on the same host):

```bash
# List anything matching the convention, with size and mtime:
ls -la "$TMPDIR"/claude-probe-* /tmp/claude-*/claude-probe-* 2>/dev/null

# Older than N days — find with mtime guard before deleting:
find "$TMPDIR" /tmp -maxdepth 3 -name 'claude-probe-*' -mtime +1 2>/dev/null

# Delete them (only after eyeballing the find output above):
find "$TMPDIR" /tmp -maxdepth 3 -name 'claude-probe-*' -mtime +1 -exec rm -rf {} + 2>/dev/null
```

If you ever have to leave the diagnosis mid-flight, leave a one-line note in
the conversation telling the user the tag and the artifact paths, so cleanup
can happen later by name.

### 8. Probes are scaffolding, not code

Whether you instrumented site-packages (no git tracking) or the user's repo,
probe lines must be gone before the diagnosis ends. For repo edits, an
additional belt-and-braces check is `git diff` before any commit; if you see
your tag, remove it. For site-packages edits, the only check is steps 5 and 6
above — there is no version control to fall back on, which is why the
hash/mirror equality check matters.

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
