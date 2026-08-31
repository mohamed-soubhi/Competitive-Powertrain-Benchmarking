"""Loader for the hand-edited YAML in ``1-mining/config``."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import yaml

from powerbench.paths import CONFIG_DIR


@lru_cache(maxsize=None)
def load_yaml(name: str) -> dict[str, Any]:
    stem = name[:-5] if name.endswith(".yaml") else name
    path = CONFIG_DIR / f"{stem}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def eea_hdv_source() -> dict[str, Any]:
    """The ``eea_hdv_co2`` block of ``sources.yaml``."""
    return load_yaml("sources")["eea_hdv_co2"]


def hdv_columns(source: dict[str, Any] | None = None) -> list[str]:
    """Flatten the grouped ``columns:`` mapping into one ordered, de-duped list."""
    src = source or eea_hdv_source()
    seen: dict[str, None] = {}
    for group in src["columns"].values():
        for col in group:
            seen.setdefault(col, None)
    return list(seen)
