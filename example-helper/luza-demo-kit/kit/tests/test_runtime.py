"""Logging / verbosity helper contracts (S11)."""

import argparse
import logging
import warnings

import pytest

from luza.runtime import (
    add_verbosity_args,
    configure_logging,
    get_logger,
    level_from_args,
    suppress_warnings,
)


def _parser():
    return add_verbosity_args(argparse.ArgumentParser())


def test_verbosity_flags_map_to_levels():
    assert level_from_args(_parser().parse_args([])) == logging.INFO
    assert level_from_args(_parser().parse_args(["-v"])) == logging.DEBUG
    assert level_from_args(_parser().parse_args(["--quiet"])) == logging.WARNING


def test_verbose_and_quiet_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        _parser().parse_args(["-v", "-q"])


def test_configure_logging_sets_root_level_and_returns_it():
    try:
        got = configure_logging(logging.DEBUG)
        assert got == logging.DEBUG
        assert logging.getLogger().level == logging.DEBUG
        assert configure_logging(_parser().parse_args(["-q"])) == logging.WARNING
    finally:
        configure_logging(logging.WARNING)  # restore quiet default for other tests


def test_suppress_warnings_is_scoped_to_named_categories():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with suppress_warnings(DeprecationWarning):
            warnings.warn("dep", DeprecationWarning)      # silenced
            warnings.warn("user", UserWarning)            # still emitted
        warnings.warn("after", DeprecationWarning)        # emitted again (scope ended)
    messages = {str(w.message) for w in caught}
    assert "dep" not in messages
    assert "user" in messages and "after" in messages


def test_suppress_warnings_no_args_is_noop():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with suppress_warnings():
            warnings.warn("kept", UserWarning)
    assert "kept" in {str(w.message) for w in caught}


def test_get_logger_returns_named_logger():
    assert get_logger("luza.test").name == "luza.test"
