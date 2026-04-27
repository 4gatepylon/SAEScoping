# Saliency Baselines (2026-04-26)

Weight-saliency pruning baselines on StemQA (biology, chemistry, math, physics).

## Methods
- **Wanda** — per-row `|weight| * activation_norm` thresholding
- **Taylor** — `|gradient * weight|` (first-order Taylor approximation)
- **Gradient** — EMA `|gradient|`
- **Random** — uniform random scores (control)

## Pipeline
1. **Calibrate** — compute saliency scores (128 samples)
2. **Prune** — threshold into masks at sparsity ∈ {0.3, 0.5, 0.7}
3. **Recover** — PGD SFT on in-domain data (3 epochs, lr=1e-5, 500 samples)
4. **Elicit** — adversarial PGD SFT on each OOD domain (3 epochs, lr=1e-5, 512 samples)
5. **Evaluate** — cross-entropy loss + optional LLM judge (gpt-4.1-nano)

## Models
- `google/gemma-2-9b-it`
- `google/gemma-3-12b-it`

## Usage
```bash
./run.sh calibrate launch --gpus 0,1 --methods wanda --domains biology
./run.sh elicit --launch --gpus 0,1 --methods wanda --domains biology
```
