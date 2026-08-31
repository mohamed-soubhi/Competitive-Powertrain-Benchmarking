"""Config-loader contracts (S2)."""

import pytest

from luza import config


def test_oem_names_handles_malformed_key():
    names = config.oem_names()
    # companies.yaml uses the malformed key "ev_ OEMs"
    assert "Tesla" in names
    assert "BMW" in names
    assert len(names) >= 8


def test_categories_and_data_sources_load():
    assert isinstance(config.categories(), dict)
    assert isinstance(config.data_sources(), dict)


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        config.load_yaml("does_not_exist")
