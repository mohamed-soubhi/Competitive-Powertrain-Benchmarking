"""Logging + verbosity helpers (FIX_PLAN.md S11).

``ml_prediction.py`` and ``build_dashboard_v2.py`` both open with::

    import warnings
    warnings.filterwarnings("ignore")

which hides real signals — sklearn convergence failures, pandas dtype warnings,
divide-by-zero in efficiency maths. This module gives the scripts a structured
alternative:

- ``get_logger``         module logger; importing this file configures nothing
- ``add_verbosity_args`` adds mutually-exclusive ``-v/--verbose`` / ``-q/--quiet``
- ``configure_logging``  the one place that installs a root handler
- ``suppress_warnings``  context manager that silences *named* warning
                         categories for a block, instead of a global ignore
"""

from __future__ import annotations

import argparse
import logging
import warnings
from contextlib import contextmanager

_FMT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def add_verbosity_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", action="store_true", help="DEBUG-level logging")
    group.add_argument("-q", "--quiet", action="store_true", help="WARNING-level logging only")
    return parser


def level_from_args(args: argparse.Namespace) -> int:
    if getattr(args, "verbose", False):
        return logging.DEBUG
    if getattr(args, "quiet", False):
        return logging.WARNING
    return logging.INFO


def configure_logging(level_or_args: int | argparse.Namespace = logging.INFO) -> int:
    """Install a root handler at the given level (or the level implied by args)."""
    level = (
        level_or_args
        if isinstance(level_or_args, int)
        else level_from_args(level_or_args)
    )
    logging.basicConfig(level=level, format=_FMT, force=True)
    logging.captureWarnings(True)
    return level


@contextmanager
def suppress_warnings(*categories: type[Warning]):
    """Silence only ``categories`` for the block. No args -> no-op.

    Scoped, explicit, and reversible replacement for a module-level
    ``warnings.filterwarnings("ignore")``.
    """
    with warnings.catch_warnings():
        for cat in categories:
            warnings.filterwarnings("ignore", category=cat)
        yield
