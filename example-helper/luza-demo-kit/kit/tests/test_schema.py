"""Schema validation / coercion contracts (S2)."""

from luza.schema import (
    PATENT_ID_RE,
    clean_ev_record,
    clean_patent_record,
)

_MULTILINE_BATTERY_TYPE = (
    "Lithium-ion\nNumber of Cells\nNo Data\nArchitecture\n400 V\n"
    "Warranty Period\n8 years\nWarranty Mileage\n160"
)


def test_battery_type_collapsed_to_first_line():
    clean, err = clean_ev_record(
        {"name": "Tesla Model 3 RWD", "battery_type": _MULTILINE_BATTERY_TYPE}
    )
    assert err is None
    assert clean["battery_type"] == "Lithium-ion"


def test_drive_is_normalised():
    clean, err = clean_ev_record({"name": "X", "drive": "Rear"})
    assert err is None and clean["drive"] == "Rear"
    clean, _ = clean_ev_record({"name": "X", "drive": "all"})
    assert clean["drive"] == "AWD"


def test_implausible_weight_is_rejected():
    # 594 was the cargo-volume value leaking into weight_kg
    clean, err = clean_ev_record({"name": "X", "weight_kg": 594})
    assert clean is None and err is not None


def test_patent_id_grammar():
    for good in ("US9331552B2", "US20200350796A1", "WO2025119898", "EP1234567B1"):
        assert PATENT_ID_RE.match(good), good
    for bad in ("9331", "US93", "", "1013"):
        assert not PATENT_ID_RE.match(bad), bad


def test_patent_row_with_broken_id_is_rejected():
    clean, err = clean_patent_record({"patent_id": "9331", "title": "t"})
    assert clean is None and "malformed patent id" in err


def test_patent_publication_date_sanitised():
    # the scraper bug: US9331552B2 -> "9331"
    clean, err = clean_patent_record({"patent_id": "US9331552B2", "publication_date": "9331"})
    assert err is None
    assert clean["publication_date"] is None

    clean, _ = clean_patent_record({"patent_id": "US20200350796A1", "publication_date": "2020"})
    assert clean["publication_date"] == "2020"

    clean, _ = clean_patent_record(
        {"patent_id": "WO2025119898", "publication_date": "2025-01-15"}
    )
    assert clean["publication_date"] == "2025-01-15"


def test_patent_country_normalised():
    clean, _ = clean_patent_record({"patent_id": "US9331552B2", "country": "us"})
    assert clean["country"] == "US"
    clean, _ = clean_patent_record({"patent_id": "US9331552B2", "country": "93"})
    assert clean["country"] is None
