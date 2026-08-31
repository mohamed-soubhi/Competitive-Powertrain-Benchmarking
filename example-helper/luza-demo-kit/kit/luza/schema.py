"""Pydantic contracts for scraped records.

Rows are validated/coerced through these models *before* being written to the
``data/cleaned`` CSVs, so corruption like the multi-line ``battery_type`` blob
or a patent "year" of ``9331`` never reaches downstream analysis.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# US utility / plant / design / reissue grants + WO/EP publications
PATENT_ID_RE = re.compile(
    r"^(?:US\d{7,8}[AB]\d?|US(?:RE|PP|D)\d{4,6}(?:[AB]\d?)?|US\d{4}/?\d{7}[AB]\d?"
    r"|WO\d{4}/?\d{6}(?:A\d)?|EP\d{7}(?:[AB]\d)?)$"
)

_DRIVE_MAP = {
    "front": "Front", "front-wheel": "Front", "fwd": "Front",
    "rear": "Rear", "rear-wheel": "Rear", "rwd": "Rear",
    "all": "AWD", "awd": "AWD", "front-rear": "AWD", "4wd": "AWD",
}


class EVSpec(BaseModel):
    """One vehicle row of the cleaned EV specs table."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    name: str
    model_year: int | None = Field(default=None, ge=2000, le=2100)
    battery_useable_kwh: float | None = Field(default=None, gt=0, lt=400)
    battery_nominal_kwh: float | None = Field(default=None, gt=0, lt=400)
    battery_chemistry: str | None = None
    battery_type: str | None = Field(default=None, max_length=60)
    architecture_v: int | None = Field(default=None, ge=100, le=1200)
    range_real_km: float | None = Field(default=None, gt=0, lt=1500)
    range_wltp_km: float | None = Field(default=None, gt=0, lt=1500)
    efficiency_wh_km: float | None = Field(default=None, gt=0, lt=600)
    power_kw: float | None = Field(default=None, gt=0, lt=2000)
    torque_nm: float | None = Field(default=None, gt=0, lt=5000)
    drive: str | None = None
    accel_0_100_s: float | None = Field(default=None, gt=1.0, lt=30.0)
    top_speed_kmh: float | None = Field(default=None, gt=50, lt=500)
    charging_dc_kw: float | None = Field(default=None, ge=0, lt=1000)
    charging_ac_kw: float | None = Field(default=None, ge=0, lt=100)
    cargo_volume_l: float | None = Field(default=None, ge=0, lt=3000)
    weight_kg: float | None = Field(default=None, gt=800, lt=4000)
    source: str | None = None
    source_url: str | None = None
    source_date: str | None = None
    citation: str | None = None

    @field_validator("battery_type", mode="before")
    @classmethod
    def _first_line_only(cls, v: Any) -> Any:
        if isinstance(v, str):
            head = v.splitlines()[0].strip() if v.strip() else ""
            return head[:60] or None
        return v

    @field_validator("battery_chemistry", mode="before")
    @classmethod
    def _norm_chem(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = v.strip().upper()
            return v or None
        return v

    @field_validator("drive", mode="before")
    @classmethod
    def _norm_drive(cls, v: Any) -> Any:
        if isinstance(v, str):
            return _DRIVE_MAP.get(v.strip().lower(), v.strip() or None)
        return v


class PatentRecord(BaseModel):
    """One patent row of the cleaned patents table."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    title: str | None = Field(default=None, max_length=300)
    patent_id: str
    assignee: str | None = None
    inventor: str | None = None
    filing_date: str | None = None
    publication_date: str | None = None
    country: str | None = Field(default=None, max_length=2)
    classification: str | None = None
    search_category: str | None = None
    abstract: str | None = None
    source: str | None = None
    source_url: str | None = None
    source_date: str | None = None
    citation: str | None = None

    @field_validator("patent_id", mode="before")
    @classmethod
    def _check_patent_id(cls, v: Any) -> Any:
        if not isinstance(v, str) or not PATENT_ID_RE.match(v.strip().upper()):
            raise ValueError(f"malformed patent id: {v!r}")
        return v.strip().upper()

    @field_validator("publication_date", "filing_date", mode="before")
    @classmethod
    def _sane_date(cls, v: Any) -> Any:
        """Accept a 4-digit year (1990-2100) or an ISO date; drop anything else.

        Guards against the scraper bug where ``US9331552B2`` yielded ``9331``.
        """
        if v in (None, ""):
            return None
        s = str(v).strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            return s
        if re.fullmatch(r"\d{4}", s) and 1990 <= int(s) <= 2100:
            return s
        return None

    @field_validator("country", mode="before")
    @classmethod
    def _norm_country(cls, v: Any) -> Any:
        if isinstance(v, str) and re.fullmatch(r"[A-Za-z]{2}", v.strip()):
            return v.strip().upper()
        return None


def clean_ev_record(raw: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Validate one raw EV dict. Returns (clean_dict, None) or (None, error_str)."""
    try:
        return EVSpec.model_validate(raw).model_dump(), None
    except Exception as exc:  # noqa: BLE001 - surfaced to caller as a string
        return None, " ".join(str(exc).split())[:200]


def clean_patent_record(raw: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Validate one raw patent dict. Returns (clean_dict, None) or (None, error_str)."""
    try:
        return PatentRecord.model_validate(raw).model_dump(), None
    except Exception as exc:  # noqa: BLE001
        return None, " ".join(str(exc).split())[:200]
