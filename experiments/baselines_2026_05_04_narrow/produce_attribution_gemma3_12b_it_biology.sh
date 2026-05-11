#!/usr/bin/env bash
# Thin wrapper: invokes produce_attribution_gemma3_12b_it_any.sh with
# dataset_config=biology. All real logic lives in the _any.sh script; see it
# for memory/CUDA/usage requirements.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/produce_attribution_gemma3_12b_it_any.sh" biology
