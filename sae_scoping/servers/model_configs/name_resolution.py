"""Config path name resolution utilities."""

from pathlib import Path
from typing import Callable

# Directory containing bundled individual model configs
_INDIVIDUAL_CONFIGS_DIR = Path(__file__).parent / "individual_configs"

# Directory containing bundled group configs
_GROUP_CONFIGS_DIR = Path(__file__).parent / "group_configs"

# Default directory for Sparsify SAE outputs
DEFAULT_SPARSIFY_SAE_OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "experiments_llama_trojans" / "science_sae" / "outputs"


def resolve_sae_artifact_path(
    user_input: str,
    file_glob: str,
    default_dir: Path | None = None,
    validator: Callable[[Path], None] | None = None,
) -> Path:
    """Generic resolver for SAE artifact paths (sae.safetensors, distribution.safetensors, etc.).

    Supports:
    - Full absolute/relative paths
    - Parent directories with unique glob-matching descendant
    - Names that resolve relative to default_dir (if provided)

    Args:
        user_input: User-provided path or name
        file_glob: Glob pattern to search for (e.g., "sae.safetensors", "*.safetensors")
        default_dir: Optional default directory to prefix if path doesn't exist
        validator: Optional callable that takes resolved Path and raises on validation failure

    Returns:
        Resolved Path to the parent directory of the matched file

    Raises:
        FileNotFoundError: If artifact cannot be found
        ValueError: If multiple matches found or validation fails
    """
    path = Path(user_input)

    # If path doesn't exist, try prefixing with default directory
    if not path.exists() and default_dir is not None:
        prefixed_path = default_dir / user_input
        if prefixed_path.exists():
            path = prefixed_path
        else:
            raise FileNotFoundError(f"Path not found: {user_input}\nAlso checked: {prefixed_path}")
    elif not path.exists():
        raise FileNotFoundError(f"Path not found: {user_input}")

    # If it's a file matching the glob, return its parent
    if path.is_file():
        if path.match(file_glob):
            result = path.parent
            if validator:
                validator(result)
            return result
        else:
            raise ValueError(f"File does not match expected pattern '{file_glob}': {path}")

    # Check if file exists directly in this directory
    direct_matches = list(path.glob(file_glob))
    if len(direct_matches) == 1:
        result = direct_matches[0].parent
        if validator:
            validator(result)
        return result

    # Search recursively
    all_matches = list(path.rglob(file_glob))
    if len(all_matches) == 0:
        raise FileNotFoundError(f"No files matching '{file_glob}' found in {path} or its subdirectories")
    if len(all_matches) > 1:
        raise ValueError(
            f"Multiple files matching '{file_glob}' found in {path}:\n" + "\n".join(f"  - {f}" for f in all_matches[:5]) + (f"\n  ... and {len(all_matches) - 5} more" if len(all_matches) > 5 else "")
        )

    # Found exactly one
    result = all_matches[0].parent
    if validator:
        validator(result)
    return result


def validate_sparsify_sae_dir(sae_dir: Path) -> None:
    """Validator for Sparsify SAE directories - checks cfg.json exists."""
    cfg_file = sae_dir / "cfg.json"
    if not cfg_file.exists():
        raise ValueError(f"cfg.json not found alongside sae.safetensors in: {sae_dir}")


def resolve_config_path(config: str) -> Path:
    """Resolve a config name or path to an actual file path.

    Supports:
    - Full absolute/relative paths: /path/to/config.json
    - Refer-by-name: "default_model_config" -> "default_model_config.json"
    - Bundled configs: looks in model_configs/ directory if not found locally

    Args:
        config: Config name or path (with or without .json extension)

    Returns:
        Resolved Path to the config file

    Raises:
        FileNotFoundError: If config file cannot be found
    """
    config_path = config
    if not config_path.endswith(".json"):
        # Support refer-by-name
        config_path = config_path + ".json"

    path = Path(config_path)
    if not path.exists():
        # Support relative paths by name to standard configs for the paper
        path = _INDIVIDUAL_CONFIGS_DIR / config_path

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    return path


def resolve_group_config_path(config: str) -> Path:
    """Resolve a group config name or path to an actual file path.

    Supports:
    - Full absolute/relative paths: /path/to/group_config.json
    - Refer-by-name: "spylab_server_group" -> "spylab_server_group.json"
    - Bundled configs: looks in group_configs/ directory if not found locally

    Args:
        config: Group config name or path (with or without .json extension)

    Returns:
        Resolved Path to the group config file

    Raises:
        FileNotFoundError: If group config file cannot be found
    """
    config_path = config
    if not config_path.endswith(".json"):
        # Support refer-by-name
        config_path = config_path + ".json"

    path = Path(config_path)
    if not path.exists():
        # Support relative paths by name to standard group configs
        path = _GROUP_CONFIGS_DIR / config_path

    if not path.exists():
        raise FileNotFoundError(f"Group config file not found: {config_path}")

    return path
