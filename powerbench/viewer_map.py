"""Translate an EEA HDV *viewer* row into the unified HDVRow field set.

The viewer tables (``HDV_<year>_viewer``) use ``OEM_*`` / ``MS_*`` column names,
carry no ``MS_Year`` (the year is the table's), give mass in different units, and
add a registration country. This module is the single place that reconciles that
shape with the 2019-2020 ``CO2_HeavyDutyVehicles`` schema.

Pure: dict + year in, dict out. No pandas, no I/O.
"""

from __future__ import annotations

from typing import Any

# viewer column -> HDVRow field (straight renames)
_RENAME: dict[str, str] = {
    "vehicle_id": "vehicle_id",
    "OEM_ManufacturerName": "Manufacturer",
    "OEM_VehicleSubGroup": "VehicleSubgroup",
    "OEM_ZeroEmissionVehicle": "ZeroEmissionVehicle",
    "OEM_HybridElectricHDV": "HybridElectricHDV",
    "OEM_DualFuelVehicle": "DualFuelVehicle",
    "OEM_VocationalVehicle": "VocationalVehicle",
    "OEM_Engine_FuelType": "Engine_FuelType",
    "OEM_CorrectedActualMass": "CurbMassChassis_kg",
    "MS_RegistrationCountry": "country",
    "MS_VehicleCategoryCode": "MS_VehicleCategoryCode",
    "CO2v": "CO2v",
}


def _name(make: Any, model: Any) -> str | None:
    parts = [str(p).strip() for p in (make, model) if p not in (None, "")]
    return " ".join(parts) or None


def _reg_date(yyyymmdd: Any) -> str | None:
    """``20230919`` (int/str) -> ``"2023-09-19"``; anything else -> None."""
    if yyyymmdd in (None, ""):
        return None
    s = str(yyyymmdd).strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    return None


def map_viewer_row(raw: dict[str, Any], year: int) -> dict[str, Any]:
    """Return a dict keyed by HDVRow field names, with ``MS_Year`` injected."""
    out: dict[str, Any] = {"MS_Year": int(year), "source_table": "viewer"}

    for src, dst in _RENAME.items():
        if src in raw:
            out[dst] = raw[src]

    out["name"] = _name(raw.get("OEM_Make"), raw.get("OEM_Model"))
    out["VehicleGroup"] = raw.get("OEM_VehicleGroup")  # HDVRow parses leading int
    out["MS_RegistrationDate"] = _reg_date(raw.get("MS_RegistrationDateClean_YYYYMMDD"))

    tpmlm_t = raw.get("MS_TPMLM")
    if tpmlm_t not in (None, ""):
        try:
            t = float(tpmlm_t)
            out["GrossVehicleMass_t"] = t
            out["MS_TechnPermMaxLadenMass"] = t * 1000.0  # 2019-20 field is in kg
        except (TypeError, ValueError):
            pass
    return out
