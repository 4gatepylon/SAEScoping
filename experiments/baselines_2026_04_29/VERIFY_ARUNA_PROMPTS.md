# DO NOT SUBMIT — Verify Aruna Judge Prompts

Run from the repo root (`~/git/SAEScoping`).

```bash
# 1. When were the iclr_judge_prompts/ templates first added?
git log --all --format="%ai %an %s" --diff-filter=A -- '**/iclr_judge_prompts/*.j2'

# 2. Aruna's commit replacing old judges with relevance/fluency/ground_truth (expect Apr 18 2026)
git show f6101cfb --stat --format="%ai %an %s"

# 3. Your WIP commit copying them to sae_scoping/evaluation/prompts/ (expect Apr 27 2026)
git show dadef528 --stat --format="%ai %an %s" | grep -E '\.j2|Date|Author|dadef'

# 4. Confirm no other commits touched our 3 prompts (expect only dadef528)
git log --all --oneline -- 'sae_scoping/evaluation/prompts/*.j2'

# 5. Confirm contents are identical between aruna and our branch (expect no output = no diff)
for f in relevance_classifier.j2 fluency_classifier.j2 ground_truth_similarity.j2; do
  echo "=== $f ==="
  git diff origin/aruna:sae_scoping/xxx_evaluation/iclr_judge_prompts/$f HEAD:sae_scoping/evaluation/prompts/$f || true
done
```
