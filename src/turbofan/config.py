from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the project config from the YAML file."""
    config_path = Path(path) if path else Path(__file__).resolve().parents[2] / 'configs' / 'config.yaml'
    with config_path.open('r', encoding='utf-8') as file:
        return yaml.safe_load(file) or {}
