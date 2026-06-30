from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "paths.yaml"


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    path = _resolve_path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    config["dataset"]["path"] = _resolve_path(config["dataset"]["path"])
    for key, value in config["outputs"].items():
        config["outputs"][key] = _resolve_path(value)
    return config


def ensure_output_dirs(config: dict[str, Any]) -> None:
    for output_path in config["outputs"].values():
        Path(output_path).mkdir(parents=True, exist_ok=True)
