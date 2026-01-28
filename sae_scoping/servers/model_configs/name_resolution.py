"""Config path name resolution utilities."""

from pathlib import Path

# Directory containing bundled model configs
_MODEL_CONFIGS_DIR = Path(__file__).parent


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
        path = _MODEL_CONFIGS_DIR / config_path

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    return path
