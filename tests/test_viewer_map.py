"""Offline tests for the viewer-row -> HDVRow mapper + merged-schema handling."""

from powerbench.schema import clean_hdv_record, row_key
from powerbench.viewer_map import map_viewer_row

_VIEWER = {
    "vehicle_id": "0xDEADBEEF",
    "OEM_ManufacturerName": "Scania CV AB",
    "OEM_Make": "Scania",
    "OEM_Model": "R 450",
    "OEM_VehicleGroup": "5",
    "OEM_VehicleSubGroup": "5-LH",
    "OEM_ZeroEmissionVehicle": "0",
    "OEM_HybridElectricHDV": "0",
    "OEM_DualFuelVehicle": "0",
    "OEM_Engine_FuelType": "Diesel CI",
    "OEM_CorrectedActualMass": 7850,
    "MS_RegistrationCountry": "de",
    "MS_VehicleCategoryCode": "N3",
    "MS_TPMLM": 18.0,
    "MS_RegistrationDateClean_YYYYMMDD": 20230919,
    "CO2v": 667.0,
}


def test_map_viewer_row_renames_and_injects_year():
    m = map_viewer_row(_VIEWER, 2023)
    assert m["MS_Year"] == 2023
    assert m["Manufacturer"] == "Scania CV AB"
    assert m["name"] == "Scania R 450"
    assert m["VehicleGroup"] == "5"
    assert m["VehicleSubgroup"] == "5-LH"
    assert m["country"] == "de"
    assert m["CurbMassChassis_kg"] == 7850
    assert m["GrossVehicleMass_t"] == 18.0
    assert m["MS_TechnPermMaxLadenMass"] == 18000.0
    assert m["MS_RegistrationDate"] == "2023-09-19"
    assert m["source_table"] == "viewer"


def test_mapped_viewer_row_passes_schema():
    rec, err = clean_hdv_record(map_viewer_row(_VIEWER, 2023))
    assert err is None
    assert rec["MS_PK_Vehicle"] is None
    assert rec["vehicle_id"] == "0xDEADBEEF"
    assert rec["country"] == "DE"           # upper-cased by the gate
    assert rec["VehicleGroup"] == 5         # "5" -> int
    assert rec["MS_Year"] == 2023


def test_exotic_vehicle_group_nulled_not_rejected():
    rec, err = clean_hdv_record(map_viewer_row({**_VIEWER, "OEM_VehicleGroup": "32d"}, 2023))
    assert err is None and rec["VehicleGroup"] is None
    rec, err = clean_hdv_record(map_viewer_row({**_VIEWER, "OEM_VehicleGroup": "21"}, 2023))
    assert err is None and rec["VehicleGroup"] is None


def test_row_without_any_id_rejected():
    bad = {k: v for k, v in map_viewer_row(_VIEWER, 2023).items() if k != "vehicle_id"}
    rec, err = clean_hdv_record(bad)
    assert rec is None and "vehicle_id" in err


def test_row_key_prefers_pk_then_vehicle_id():
    assert row_key({"MS_Year": 2020, "MS_PK_Vehicle": 43, "vehicle_id": "x"}) == (2020, 43)
    assert row_key({"MS_Year": 2023, "MS_PK_Vehicle": None, "vehicle_id": "abc"}) == (2023, "abc")


def test_bad_reg_date_int_dropped():
    m = map_viewer_row({**_VIEWER, "MS_RegistrationDateClean_YYYYMMDD": 99}, 2023)
    assert m["MS_RegistrationDate"] is None
