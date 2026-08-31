"""Pydantic contract for one cleaned HDV CO2 row.

Raw rows come from EEA Discodata (``[CO2Emission].[latest].[CO2_HeavyDutyVehicles]``,
~250 columns). The miner keeps a curated 38-column subset; this gate coerces
types, enforces plausibility bounds, and drops rows that violate them *loudly*
(the caller logs a per-field rejection reason) before anything reaches DuckDB.

Derived columns (``brand``, ``oem_group``, ``powertrain_class``) are added by
``reclean.py`` after validation, not here.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_YEAR_RE = re.compile(r"^\d{4}$")

# Optional measurement columns: a value outside (lo, hi] is treated as
# "not really measured" and coerced to None, rather than rejecting the whole
# row. EEA rows routinely carry 0 / sentinel values in fields an OEM left blank.
_PLAUSIBLE: dict[str, tuple[float, float]] = {
    "Engine_Displacement_ltr": (0.0, 40.0),
    "Engine_RatedPower_kw": (0.0, 1500.0),
    "Engine_RatedSpeed_rpm": (0.0, 6000.0),
    "Engine_IdlingSpeed_rpm": (0.0, 3000.0),
    "GrossVehicleMass_t": (0.0, 120.0),
    "CurbMassChassis_kg": (0.0, 60000.0),
    "COL_TotalVehicleMass_kg": (0.0, 120000.0),
    "MS_TechnPermMaxLadenMass": (0.0, 120000.0),
    "MS_SpecificCO2Emissions": (0.0, 3000.0),
    "CO2v": (0.0, 3000.0),
    "COL_CO2_gtkm": (0.0, 3000.0),
    "COL_CO2_gkm": (0.0, 5000.0),
    "COL_FuelConsumption_l100km": (0.0, 200.0),
    "WHTC_CO2_gkwh": (0.0, 2000.0),
    "WHSC_CO2_gkwh": (0.0, 2000.0),
}


class HDVRow(BaseModel):
    """One validated heavy-duty-vehicle record."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    # --- identity ---
    # MS_PK_Vehicle is present on the 2019-2020 table; the viewer tables carry a
    # hex vehicle_id instead. At least one must be set (checked in clean_hdv_record).
    MS_PK_Vehicle: int | None = None
    OEM_PK_Vehicle: int | None = None
    Meta_OEM_fileId: int | None = None
    vehicle_id: str | None = Field(default=None, max_length=80)
    source_table: str | None = Field(default=None, max_length=20)  # "vehicle" | "viewer"

    # --- time ---
    MS_Year: int | None = Field(default=None, ge=2018, le=2030)
    MS_RegistrationDate: str | None = None

    # --- OEM ---
    name: str | None = Field(default=None, max_length=120)
    Manufacturer: str | None = Field(default=None, max_length=200)
    ManufacturerAddress: str | None = Field(default=None, max_length=300)
    country: str | None = Field(default=None, max_length=2)  # MS registration country

    # --- powertrain ---
    Engine_FuelType: str | None = Field(default=None, max_length=60)
    COL_FuelType: str | None = Field(default=None, max_length=60)
    HybridElectricHDV: bool | None = None
    ZeroEmissionVehicle: bool | None = None
    DualFuelVehicle: bool | None = None
    Engine_Displacement_ltr: float | None = None
    Engine_RatedPower_kw: float | None = None
    Engine_RatedSpeed_rpm: int | None = None
    Engine_IdlingSpeed_rpm: int | None = None

    # --- segment ---
    # viewer tables give strings like "5", "1s", "32d"; a before-validator keeps
    # the leading integer and nulls the exotic bus/trailer groups (which carry no
    # comparable CO2v anyway).
    VehicleGroup: int | None = Field(default=None, ge=0, le=20)
    VehicleSubgroup: str | None = Field(default=None, max_length=20)
    LegislativeClass: str | None = Field(default=None, max_length=20)
    AxleConfiguration: str | None = Field(default=None, max_length=20)
    GrossVehicleMass_t: float | None = None
    MS_VehicleCategoryCode: str | None = Field(default=None, max_length=20)
    VocationalVehicle: bool | None = None
    ExemptedVehicle: bool | None = None

    # --- mass ---
    CurbMassChassis_kg: float | None = None
    COL_TotalVehicleMass_kg: float | None = None
    MS_TechnPermMaxLadenMass: float | None = None

    # --- CO2 / fuel consumption ---
    MS_SpecificCO2Emissions: float | None = None
    CO2v: float | None = None
    COL_CO2_gtkm: float | None = None
    COL_CO2_gkm: float | None = None
    COL_FuelConsumption_l100km: float | None = None
    WHTC_CO2_gkwh: float | None = None
    WHSC_CO2_gkwh: float | None = None

    # --- tech flags ---
    AdvCO2Tech1_AdvancedCO2Technology: str | None = Field(default=None, max_length=120)
    AdvCO2Tech1_Category: str | None = Field(default=None, max_length=120)
    ADAS_PredictiveCruiseControl: bool | None = None
    ADAS_EngineStopStart: bool | None = None

    @field_validator(
        "Manufacturer",
        "ManufacturerAddress",
        "Engine_FuelType",
        "COL_FuelType",
        "VehicleSubgroup",
        "LegislativeClass",
        "AxleConfiguration",
        "MS_VehicleCategoryCode",
        "AdvCO2Tech1_AdvancedCO2Technology",
        "AdvCO2Tech1_Category",
        mode="before",
    )
    @classmethod
    def _blank_to_none(cls, v: Any) -> Any:
        if isinstance(v, str):
            head = v.splitlines()[0].strip() if v.strip() else ""
            return head or None
        return v

    @field_validator("country", mode="before")
    @classmethod
    def _country_code(cls, v: Any) -> Any:
        if isinstance(v, str) and len(v.strip()) == 2 and v.strip().isalpha():
            return v.strip().upper()
        return None

    @field_validator("VehicleGroup", mode="before")
    @classmethod
    def _group_leading_int(cls, v: Any) -> Any:
        """'5' -> 5; '32d' / '1s' / '21'+ -> None (no comparable CO2v for those)."""
        if v in (None, ""):
            return None
        s = str(v).strip()
        if not s.isdigit():
            return None
        n = int(s)
        return n if 0 <= n <= 20 else None

    @field_validator("MS_RegistrationDate", mode="before")
    @classmethod
    def _sane_date(cls, v: Any) -> Any:
        if v in (None, ""):
            return None
        s = str(v).strip()
        if _DATE_RE.match(s):
            return s[:10]
        if _YEAR_RE.match(s) and 2015 <= int(s) <= 2030:
            return s
        return None

    @field_validator(
        "HybridElectricHDV",
        "ZeroEmissionVehicle",
        "DualFuelVehicle",
        "VocationalVehicle",
        "ExemptedVehicle",
        "ADAS_PredictiveCruiseControl",
        "ADAS_EngineStopStart",
        mode="before",
    )
    @classmethod
    def _coerce_bool(cls, v: Any) -> Any:
        if v in (None, ""):
            return None
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return v != 0
        return str(v).strip().lower() in {"1", "true", "yes", "y"}

    @field_validator(*_PLAUSIBLE.keys(), mode="before")
    @classmethod
    def _plausible_or_none(cls, v: Any, info) -> Any:
        """Null a measurement that is missing, non-numeric, or out of range."""
        if v in (None, ""):
            return None
        try:
            x = float(v)
        except (TypeError, ValueError):
            return None
        lo, hi = _PLAUSIBLE[info.field_name]
        return v if lo < x <= hi else None


def clean_hdv_record(raw: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Validate one raw dict. Returns ``(clean, None)`` or ``(None, error_str)``."""
    try:
        rec = HDVRow.model_validate(raw).model_dump()
    except Exception as exc:  # noqa: BLE001 - surfaced to caller as a string
        return None, " ".join(str(exc).split())[:240]
    if rec.get("MS_PK_Vehicle") is None and not rec.get("vehicle_id"):
        return None, "row has neither MS_PK_Vehicle nor vehicle_id"
    return rec, None


def row_key(rec: dict[str, Any]) -> tuple:
    """Stable identity for de-duplication across snapshots / source tables."""
    ident = rec.get("MS_PK_Vehicle")
    if ident is None:
        ident = rec.get("vehicle_id")
    return (rec.get("MS_Year"), ident)
