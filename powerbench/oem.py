"""Pure helpers: canonicalise messy OEM names and derive a powertrain class.

Discodata's ``Manufacturer`` field on the HDV CO2 table is dirty: legal-entity
variants for the same maker ("Daimler AG" vs "Daimler Truck AG"), case drift
("RENAULT TRUCK" / "RENAULT TRUCKS"), and postal addresses leaked into the name.
The raw value is kept; a ``brand`` / ``oem_group`` pair is added alongside it.

No I/O, no pandas — string in, string out — so it unit-tests offline and is
reused by both the miner and the reclean step.
"""

from __future__ import annotations

import re

# substring (lowercased, whitespace-collapsed) -> (brand, parent group)
_OEM_RULES: list[tuple[str, tuple[str, str]]] = [
    ("scania", ("Scania", "TRATON")),
    ("man truck", ("MAN", "TRATON")),
    ("daf trucks", ("DAF", "PACCAR")),
    ("paccar", ("DAF", "PACCAR")),
    ("volvo truck", ("Volvo Trucks", "Volvo Group")),
    ("renault truck", ("Renault Trucks", "Volvo Group")),
    ("daimler truck", ("Daimler Truck", "Daimler Truck")),
    ("mitsubishi fuso", ("FUSO", "Daimler Truck")),
    ("daimler ag", ("Daimler Truck", "Daimler Truck")),
    ("mercedes", ("Daimler Truck", "Daimler Truck")),
    ("iveco", ("IVECO", "IVECO Group")),
    ("ford otosan", ("Ford Trucks", "Ford Otosan")),
    ("ford", ("Ford Trucks", "Ford Otosan")),
    ("isuzu", ("Isuzu", "Isuzu")),
    ("dongfeng", ("Dongfeng", "Dongfeng")),
]

_UNKNOWN = ("Unknown", "Unknown")

# a Manufacturer value that is really a postal address, not a maker name
_ADDRESS_HINT = re.compile(r"\d{3,}|mah\.|cad\.|str\.|straße|strasse|no:\s*\d", re.I)


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def looks_like_address(value: str | None) -> bool:
    return bool(value) and bool(_ADDRESS_HINT.search(value))


def canonical_oem(value: str | None) -> tuple[str, str]:
    """Return ``(brand, oem_group)`` for a raw Manufacturer string."""
    if not isinstance(value, str) or not value.strip() or looks_like_address(value):
        return _UNKNOWN
    norm = _norm(value)
    for needle, pair in _OEM_RULES:
        if needle in norm:
            return pair
    return _UNKNOWN


# --- powertrain classification -------------------------------------------------

POWERTRAIN_CLASSES = (
    "BEV / Zero-emission",
    "Hybrid electric",
    "Dual-fuel",
    "Gas (CNG/LNG)",
    "Ethanol ICE",
    "Diesel ICE",
    "Other / Unknown",
)


def _truthy_flag(v) -> bool:
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def powertrain_class(
    *,
    zero_emission=None,
    hybrid=None,
    dual_fuel=None,
    fuel_type: str | None = None,
) -> str:
    """Collapse the EEA flag columns + engine fuel type into one label.

    Priority: zero-emission > hybrid > dual-fuel > gas > ethanol > diesel.
    """
    if _truthy_flag(zero_emission):
        return "BEV / Zero-emission"
    if _truthy_flag(hybrid):
        return "Hybrid electric"
    if _truthy_flag(dual_fuel):
        return "Dual-fuel"
    ft = fuel_type.strip().lower() if isinstance(fuel_type, str) else ""
    if not ft:
        return "Other / Unknown"
    if "ng" in ft or "gas" in ft or "cng" in ft or "lng" in ft:
        return "Gas (CNG/LNG)"
    if "ethanol" in ft:
        return "Ethanol ICE"
    if "diesel" in ft or "ci" in ft:
        return "Diesel ICE"
    return "Other / Unknown"
