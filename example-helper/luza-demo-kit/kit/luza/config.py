"""Typed-ish accessors over the YAML config files in ``1-mining/config``.

The reviews flagged these files as "orphaned" — present but never loaded.
Scrapers now read their target lists from here instead of hardcoding them.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import yaml

from luza.paths import CONFIG_DIR


@lru_cache(maxsize=None)
def load_yaml(name: str) -> dict[str, Any]:
    """Load ``1-mining/config/<name>.yaml`` (``.yaml`` optional)."""
    stem = name[:-5] if name.endswith(".yaml") else name
    path = CONFIG_DIR / f"{stem}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def companies() -> dict[str, Any]:
    return load_yaml("companies")


def _find_key(data: dict[str, Any], *needles: str) -> str | None:
    """Return the first key loosely matching every needle (case/space-insensitive)."""
    for key in data:
        norm = key.lower().replace(" ", "").replace("_", "")
        if all(n in norm for n in needles):
            return key
    return None


def oem_names() -> list[str]:
    """Names of the EV OEMs listed in companies.yaml.

    Tolerant of the malformed ``ev_ OEMs`` key in the current file.
    """
    data = companies()
    key = _find_key(data, "ev", "oem") or _find_key(data, "oem")
    entries = data.get(key, []) if key else []
    return [e["name"] for e in entries if isinstance(e, dict) and "name" in e]


def categories() -> dict[str, Any]:
    return load_yaml("categories")


def data_sources() -> dict[str, Any]:
    return load_yaml("data_sources")


def voltage_architectures() -> list[str]:
    """Known pack-voltage architectures, e.g. ['400 V', '800 V']."""
    return ["400 V", "800 V"]
