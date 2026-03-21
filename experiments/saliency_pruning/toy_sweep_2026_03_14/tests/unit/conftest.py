"""Ensure the experiment root (toy_sweep_2026_03_14/) is on sys.path."""
import sys
from pathlib import Path

# tests/unit/ → tests/ → toy_sweep_2026_03_14/
_EXPERIMENT_ROOT = Path(__file__).parent.parent.parent
if str(_EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_ROOT))
