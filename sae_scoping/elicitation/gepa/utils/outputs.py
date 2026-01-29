from __future__ import annotations
import hashlib
import warnings
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

"""By Claude-Code. Support a `with ...` context for deciding where to store outputs."""

ALLOWED_TYPES = (dict, list, tuple, int, bool, str, float, type(None))


def stringify(obj: Any) -> str:
    """Stringify an object as <type>:<object> to avoid ambiguity."""
    type_name = type(obj).__name__
    return f"{type_name}:{obj}"


def hash_str(s: str) -> str:
    """Hash a string using SHA256."""
    return hashlib.sha256(s.encode()).hexdigest()


def hash_value(obj: Any) -> str:
    """
    Recursively hash a value using a deterministic strategy.

    Supported types: dict, list, tuple, int, bool, str, float, NoneType
    - dict: sort by key, hash each key-value pair, concatenate, hash
    - list/tuple: hash each element in order, concatenate, hash
    - primitives: hash the stringified value
    """
    if not isinstance(obj, ALLOWED_TYPES):
        raise TypeError(f"Unsupported type for hashing: {type(obj).__name__}")

    if isinstance(obj, dict):
        if not all(isinstance(k, str) for k in obj.keys()):
            raise TypeError("Dict keys must be strings")
        # Sort by key, hash each key-value pair
        pair_hashes = []
        for k in sorted(obj.keys()):
            v_hash = hash_value(obj[k])
            pair_hashes.append(hash_str(f"{k}:{v_hash}"))
        return hash_str("dict:" + ",".join(pair_hashes))

    elif isinstance(obj, (list, tuple)):
        # Hash each element in order
        type_name = type(obj).__name__
        element_hashes = [hash_value(e) for e in obj]
        return hash_str(f"{type_name}:" + ",".join(element_hashes))

    else:
        # Primitives: int, bool, str, float, NoneType
        return hash_str(stringify(obj))


def get_script_hash() -> str:
    """Get SHA256 hash of the current script's absolute path."""
    script_path = Path(sys.argv[0]).resolve()
    return hashlib.sha256(str(script_path).encode()).hexdigest()


def serialize_for_json(obj: Any) -> Any:
    """Convert tuples to JSON-serializable format with type tags."""
    if isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, tuple):
        return {"__type__": "tuple", "__value__": [serialize_for_json(e) for e in obj]}
    elif isinstance(obj, list):
        return [serialize_for_json(e) for e in obj]
    else:
        return obj


@contextmanager
def OutputDir(
    unique_id: str | None = None,  #
    kwargs: dict | list | tuple | int | bool | str | float | None = None,  #
    base_dir: Path | str | None = Path(__file__).parent.parent / "outputs",  # This is where all outputs go
):
    """
    Context manager that provides a unique output directory.

    Args:
        unique_id: Unique identifier for this experiment. Defaults to SHA256 hash
                   of the script's absolute path.
        kwargs: Parameters to hash for the subdirectory. Can be dict, list, tuple,
                or primitive types. Defaults to None.
        base_dir: Base directory for outputs. Defaults to ../outputs (the "official" output directory).

    Yields:
        Path to outputs/<unique_id>/<kwargs_hash>/contents

    Example:
        with OutputDir(kwargs={"lr": 0.001}) as out_dir:
            (out_dir / "results.json").write_text(json.dumps(results))
    """
    stem_name = Path(sys.argv[0]).resolve().stem
    if unique_id is None:
        unique_id = get_script_hash()

    kwargs_hash = hash_value(kwargs)

    if base_dir is None:
        warnings.warn("base_dir is None, using script's directory as base")
        base_dir = Path(sys.argv[0]).resolve().parent / "outputs"
    else:
        base_dir = Path(base_dir)

    output_root = base_dir / stem_name / unique_id / kwargs_hash
    contents_dir = output_root / "contents"
    contents_dir.mkdir(parents=True, exist_ok=True)

    # Save kwargs to JSON
    kwargs_file = output_root / "kwargs.json"
    kwargs_file.write_text(json.dumps(serialize_for_json(kwargs), indent=2))

    yield contents_dir
