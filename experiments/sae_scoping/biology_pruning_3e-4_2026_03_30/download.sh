#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

# Downloads into ./downloaded/ (mirroring HF repo structure):
#   ./downloaded/model_layers_31_h0.0003/outputs/checkpoint-3000/  (model weights)
#   ./downloaded/deleteme_cache_bio_only/.../distribution.safetensors  (SAE distribution)

python download.py distribution
python download.py model model_layers_31_h0.0002 checkpoint-2000
python download.py model model_layers_31_h0.0003 checkpoint-2000
python download.py model model_layers_31_h0.0004 checkpoint-2000
