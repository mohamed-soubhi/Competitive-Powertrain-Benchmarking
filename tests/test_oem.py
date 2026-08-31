"""Offline tests for OEM canonicalisation + powertrain classification."""

import pytest

from powerbench.oem import canonical_oem, looks_like_address, powertrain_class


@pytest.mark.parametrize(
    "raw, brand, group",
    [
        ("Scania CV AB", "Scania", "TRATON"),
        ("MAN Truck & Bus SE", "MAN", "TRATON"),
        ("MAN Truck & Bus AG", "MAN", "TRATON"),
        ("DAF Trucks N.V.", "DAF", "PACCAR"),
        ("VOLVO TRUCK CORPORATION", "Volvo Trucks", "Volvo Group"),
        ("RENAULT TRUCKS", "Renault Trucks", "Volvo Group"),
        ("Daimler AG", "Daimler Truck", "Daimler Truck"),
        ("Daimler Truck AG", "Daimler Truck", "Daimler Truck"),
        ("IVECO S.p.A.", "IVECO", "IVECO Group"),
        ("Mitsubishi Fuso Truck and Bus Corporation", "FUSO", "Daimler Truck"),
    ],
)
def test_canonical_oem_known(raw, brand, group):
    assert canonical_oem(raw) == (brand, group)


def test_canonical_oem_address_junk_is_unknown():
    junk = "Akpinar Mah. Hasan Basri Cad. No:2 34885 Istanbul, TURKEY"
    assert looks_like_address(junk)
    assert canonical_oem(junk) == ("Unknown", "Unknown")


def test_canonical_oem_none():
    assert canonical_oem(None) == ("Unknown", "Unknown")


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (dict(zero_emission=True, fuel_type="Diesel CI"), "BEV / Zero-emission"),
        (dict(zero_emission=False, hybrid=1), "Hybrid electric"),
        (dict(dual_fuel=True, fuel_type="Diesel CI"), "Dual-fuel"),
        (dict(fuel_type="NG PI"), "Gas (CNG/LNG)"),
        (dict(fuel_type="Ethanol CI"), "Ethanol ICE"),
        (dict(fuel_type="Diesel CI"), "Diesel ICE"),
        (dict(fuel_type=""), "Other / Unknown"),
        (dict(), "Other / Unknown"),
    ],
)
def test_powertrain_class(kwargs, expected):
    assert powertrain_class(**kwargs) == expected
