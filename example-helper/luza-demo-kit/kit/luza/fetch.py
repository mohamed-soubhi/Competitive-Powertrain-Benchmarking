"""Adaptive page fetching (FIX_PLAN.md S3b).

Prefers ``scrapling`` (realistic headers, adaptive CSS selectors that self-heal
when class names drift). Falls back to ``requests`` when scrapling is not
installed, so the scrapers still import and their pure parsers stay unit-testable
in an offline environment.

``fetch_dom`` returns a scrapling ``Adaptor`` (has ``.css`` / ``.css_first`` with
``::text`` / ``::attr(...)`` pseudo-selectors); ``fetch_text`` returns a
flattened string for the regex fallback extractors.
"""

from __future__ import annotations

import logging

log = logging.getLogger("luza.fetch")

try:  # optional dependency
    from scrapling.fetchers import Fetcher as _Fetcher

    _HAS_SCRAPLING = True
except Exception:  # pragma: no cover - exercised only where scrapling is absent
    _Fetcher = None
    _HAS_SCRAPLING = False


def has_scrapling() -> bool:
    return _HAS_SCRAPLING


def fetch_dom(url: str, timeout: int = 25, retries: int = 2):
    """Return a scrapling ``Adaptor`` for ``url`` on HTTP 200, else ``None``."""
    if not _HAS_SCRAPLING:
        return None
    detail = "no attempt"
    for attempt in range(1, retries + 2):
        try:
            page = _Fetcher.get(url, timeout=timeout)
            if getattr(page, "status", None) == 200:
                return page
            detail = f"HTTP {getattr(page, 'status', '?')}"
        except Exception as exc:  # network / parse
            detail = repr(exc)[:160]
        log.warning("fetch_dom %s attempt %d/%d: %s", url, attempt, retries + 1, detail)
    return None


def fetch_text(url: str, timeout: int = 25) -> str | None:
    """Flattened page text (scrapling if available, else a ``requests`` GET)."""
    dom = fetch_dom(url, timeout=timeout)
    if dom is not None:
        try:
            return dom.get_all_text(ignore_tags=("script", "style"))
        except Exception:  # pragma: no cover
            return str(getattr(dom, "body", "") or "")
    try:
        import requests

        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        return resp.text
    except Exception as exc:
        log.warning("fetch_text fallback failed for %s: %s", url, exc)
        return None
