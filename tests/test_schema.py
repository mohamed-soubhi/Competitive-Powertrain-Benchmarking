"""Offline tests for the pydantic HDV validation gate."""

from powerbench.schema import HDVRow, clean_hdv_record

_GOOD = {
    "MS_PK_Vehicle": 43,
    "MS_Year": 2019,
    "Manufacturer": "Scania CV AB",
    "Engine_FuelType": "Diesel CI",
    "Engine_RatedPower_kw": 368,
    "VehicleGroup": 5,
    "GrossVehicleMass_t": 20.5,
    "ZeroEmissionVehicle": False,
    "HybridElectricHDV": 0,
    "MS_SpecificCO2Emissions": 0.0,
}


def test_good_row_passes():
    rec, err = clean_hdv_record(_GOOD)
    assert err is None
    assert rec["MS_PK_Vehicle"] == 43
    assert rec["ZeroEmissionVehicle"] is False
    assert rec["HybridElectricHDV"] is False  # 0 -> bool


def test_missing_pk_rejected():
    rec, err = clean_hdv_record({k: v for k, v in _GOOD.items() if k != "MS_PK_Vehicle"})
    assert rec is None
    assert "MS_PK_Vehicle" in err


def test_out_of_range_measurement_nulled_not_rejected():
    # a bad optional measurement must not drop the whole vehicle
    rec, err = clean_hdv_record({**_GOOD, "Engine_RatedPower_kw": 9999})
    assert err is None
    assert rec["Engine_RatedPower_kw"] is None
    assert rec["MS_PK_Vehicle"] == 43


def test_zero_measurement_becomes_none():
    rec, _ = clean_hdv_record({**_GOOD, "Engine_Displacement_ltr": 0, "GrossVehicleMass_t": 0.0})
    assert rec["Engine_Displacement_ltr"] is None
    assert rec["GrossVehicleMass_t"] is None


def test_blank_strings_become_none():
    rec, _ = clean_hdv_record({**_GOOD, "Engine_FuelType": "   ", "VehicleSubgroup": ""})
    assert rec["Engine_FuelType"] is None
    assert rec["VehicleSubgroup"] is None


def test_multiline_string_first_line_only():
    rec, _ = clean_hdv_record({**_GOOD, "AdvCO2Tech1_AdvancedCO2Technology": "Foo\nbar\nbaz"})
    assert rec["AdvCO2Tech1_AdvancedCO2Technology"] == "Foo"


def test_bad_registration_date_dropped():
    assert clean_hdv_record({**_GOOD, "MS_RegistrationDate": "9331"})[0]["MS_RegistrationDate"] is None
    assert clean_hdv_record({**_GOOD, "MS_RegistrationDate": "2019-05-01T00:00:00"})[0][
        "MS_RegistrationDate"
    ] == "2019-05-01"
    assert clean_hdv_record({**_GOOD, "MS_RegistrationDate": "2020"})[0]["MS_RegistrationDate"] == "2020"


def test_bool_coercion_from_strings():
    rec, _ = clean_hdv_record({**_GOOD, "ZeroEmissionVehicle": "true", "DualFuelVehicle": "0"})
    assert rec["ZeroEmissionVehicle"] is True
    assert rec["DualFuelVehicle"] is False


def test_extra_source_columns_ignored():
    rec, err = clean_hdv_record({**_GOOD, "SomeUnmappedVectoColumn": 1.23})
    assert err is None
    assert "SomeUnmappedVectoColumn" not in rec


def test_year_bounds():
    rec, err = clean_hdv_record({**_GOOD, "MS_Year": 1900})
    assert rec is None and "MS_Year" in err
