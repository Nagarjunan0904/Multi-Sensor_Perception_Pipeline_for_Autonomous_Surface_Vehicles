"""Configuration loading utilities."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


class ConfigLoader:
    """Load and manage YAML configurations."""

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize config loader.

        Args:
            config_dir: Default directory for config files.
        """
        self.config_dir = Path(config_dir) if config_dir else Path("configs")
        self._cache: Dict[str, Dict] = {}

    def load(
        self,
        config_path: Union[str, Path],
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to config file.
            use_cache: Whether to use cached config.

        Returns:
            Configuration dictionary.
        """
        config_path = Path(config_path)

        # Resolve relative paths
        if not config_path.is_absolute():
            # Check if the path already exists as-is (e.g., 'configs/default.yaml')
            if config_path.exists():
                pass  # Use path as-is
            # Check if path starts with config_dir (avoid double-prefixing)
            elif str(config_path).startswith(str(self.config_dir)):
                pass  # Use path as-is
            else:
                # Prepend config_dir only if path is just filename or relative to config_dir
                config_path = self.config_dir / config_path

        cache_key = str(config_path)

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key].copy()

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Process includes
        config = self._process_includes(config, config_path.parent)

        if use_cache:
            self._cache[cache_key] = config

        return config.copy()

    def _process_includes(
        self,
        config: Dict,
        base_dir: Path,
    ) -> Dict:
        """
        Process !include directives in config.

        Args:
            config: Configuration dictionary.
            base_dir: Base directory for relative includes.

        Returns:
            Processed configuration.
        """
        if not isinstance(config, dict):
            return config

        result = {}

        for key, value in config.items():
            if isinstance(value, str) and value.startswith("!include "):
                include_path = base_dir / value[9:]
                with open(include_path, "r") as f:
                    result[key] = yaml.safe_load(f)
            elif isinstance(value, dict):
                result[key] = self._process_includes(value, base_dir)
            else:
                result[key] = value

        return result

    def merge(
        self,
        base: Dict[str, Any],
        override: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Deep merge two configurations.

        Args:
            base: Base configuration.
            override: Override configuration.

        Returns:
            Merged configuration.
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.merge(result[key], value)
            else:
                result[key] = value

        return result

    def save(
        self,
        config: Dict[str, Any],
        path: Union[str, Path],
    ) -> None:
        """
        Save configuration to YAML file.

        Args:
            config: Configuration dictionary.
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    def clear_cache(self) -> None:
        """Clear configuration cache."""
        self._cache.clear()


def load_config(
    config_path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to load configuration.

    Args:
        config_path: Path to config file.
        overrides: Optional overrides to apply.

    Returns:
        Configuration dictionary.
    """
    loader = ConfigLoader()
    config = loader.load(config_path)

    if overrides:
        config = loader.merge(config, overrides)

    return config


def get_nested(
    config: Dict[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    """
    Get nested config value using dot notation.

    Args:
        config: Configuration dictionary.
        key: Dot-separated key (e.g., 'model.learning_rate').
        default: Default value if key not found.

    Returns:
        Config value or default.
    """
    keys = key.split(".")
    value = config

    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default

    return value


def set_nested(
    config: Dict[str, Any],
    key: str,
    value: Any,
) -> None:
    """
    Set nested config value using dot notation.

    Args:
        config: Configuration dictionary.
        key: Dot-separated key.
        value: Value to set.
    """
    keys = key.split(".")
    current = config

    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]

    current[keys[-1]] = value
