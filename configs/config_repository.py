import json
import threading
from pathlib import Path
from typing import Any
from scipy.stats import randint, loguniform
from scipy.stats._distn_infrastructure import rv_frozen

config_file_path = "Project/configs/models_config.json"

class ConfigRepository:
    """
    Manages storing and retrieving model configs in a JSON file.
    Thread-safe for concurrent reads/writes.
    """

    def __init__(self, config_path: str = config_file_path):
        """Initialize repository with optional custom config file path."""
        self._path = Path(config_path)
        self._lock = threading.Lock()
        if not self._path.exists():
            self._write({})

    def _read(self) -> dict:
        try:
            with self._lock, open(self._path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file: {e}")

    def _write(self, data: dict):
        with self._lock, open(self._path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    def add_config(self, model_name: str, config: dict) -> None:
        """Add or update the configuration for a model."""
        data = self._read()
        data[model_name] = self._serialize(config)
        self._write(data)

    def get_config(self, model_name: str) -> dict:
        """Retrieve the configuration for a model."""
        data = self._read()
        if model_name not in data:
            raise KeyError(f"Model '{model_name}' not found.")
        return self._deserialize(data[model_name])

    def remove_config(self, model_name: str) -> None:
        """Remove a model's configuration."""
        data = self._read()
        if model_name not in data:
            raise KeyError(f"Model '{model_name}' not found.")
        data.pop(model_name)
        self._write(data)

    def list_models(self) -> list:
        """List all model names in the repository."""
        return list(self._read().keys())

    def _serialize(self, config: dict) -> Any:
        """Recursively convert non-JSON-serializable objects to primitives."""
        def convert(val: Any) -> Any:
            if isinstance(val, rv_frozen):
                dist_name = getattr(val.dist, 'name', None)
                if dist_name == 'randint':
                    return {"type": "randint", "params": [val.args[0], val.args[1]]}
                if dist_name == 'loguniform':
                    return {"type": "loguniform", "params": [val.args[0], val.args[1]]}
                return {"type": dist_name, "params": list(val.args)}
            if isinstance(val, dict):
                return {k: convert(v) for k, v in val.items()}
            if isinstance(val, (list, tuple)):
                return [convert(v) for v in val]
            return val
        return convert(config)

    def _deserialize(self, config: Any) -> Any:
        """Recursively rebuild complex objects from serialized config."""
        def convert(val: Any) -> Any:
            # If it's a dict with a type indicator
            if isinstance(val, dict) and 'type' in val:
                t = val['type']
                # Categorical choices
                if t == 'categorical':
                    # Return the list of choices directly
                    return val.get('choices', [])
                params = val.get('params', [])
                if t == 'randint':
                    return randint(*params)
                if t == 'loguniform':
                    return loguniform(*params)
                return val  # Fallback for unknown types
            # Recurse into nested dicts
            if isinstance(val, dict):
                return {k: convert(v) for k, v in val.items()}
            # Recurse into lists
            if isinstance(val, list):
                return [convert(v) for v in val]
            return val
        return convert(config)
