"""Client for the EEA Discodata SQL-over-HTTP endpoint.

Discodata (https://discodata.eea.europa.eu/) exposes EEA's SQL Server as a
read-only REST endpoint: ``GET /sql?query=<SQL>&nrOfHits=<n>`` returning
``{"results": [...]}`` or ``{"errors": [{"error": ..., "errorcode": ...}]}``.

Quirks discovered against the HDV CO2 database and coded around here:

* The ``p`` (page) parameter is unreliable — repeated pages return identical
  rows. Do **not** paginate with it. Instead filter server-side (WHERE) into
  chunks small enough to return in one response, and pass a high ``nrOfHits``.
* ``OFFSET ... FETCH``, ``TOP`` combined with paging, ``INFORMATION_SCHEMA`` and
  ``sys.*`` are rejected (errorcode 10001 / 10002).
* ``nrOfHits`` caps row count only on unordered queries; an ``ORDER BY`` query
  returns the full result set regardless. Keep chunks bounded by the WHERE.

The SQL builder (:func:`build_select`) is a pure function with no network use,
unit-tested offline.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger("powerbench.discodata")

DISCODATA_SQL_URL = "https://discodata.eea.europa.eu/sql"
DEFAULT_NR_OF_HITS = 500_000


class DiscodataError(RuntimeError):
    """Discodata returned an ``errors`` payload or an unexpected shape."""


class NonRetryableDiscodataError(RuntimeError):
    """A Discodata error that will never succeed on retry (bad SQL / object).

    Deliberately NOT a subclass of :class:`DiscodataError` so the retry
    predicate ignores it and the caller fails fast.
    """


def _quote_literal(value: Any) -> str:
    """Render a Python value as a T-SQL literal (single-quote escaped)."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return repr(value)
    return "N'" + str(value).replace("'", "''") + "'"


def qualified_table(database: str, schema: str, table: str) -> str:
    return f"[{database}].[{schema}].[{table}]"


def build_select(
    table: str,
    columns: Sequence[str],
    *,
    where: Sequence[str] | None = None,
    equals: dict[str, Any] | None = None,
    in_lists: dict[str, Sequence[Any]] | None = None,
    order_by: Sequence[str] | None = None,
    top: int | None = None,
) -> str:
    """Assemble a single-line ``SELECT`` string.

    ``equals`` / ``in_lists`` are convenience predicates ANDed with any raw
    ``where`` fragments. Column and table names are trusted (they come from our
    own config, not user input); scalar values are quoted.
    """
    if not columns:
        raise ValueError("columns must be non-empty")
    cols = ", ".join(f"[{c}]" if not c.startswith("[") else c for c in columns)
    top_sql = f"TOP {int(top)} " if top else ""
    sql = f"SELECT {top_sql}{cols} FROM {table}"

    clauses: list[str] = list(where or [])
    for col, val in (equals or {}).items():
        clauses.append(f"[{col}] = {_quote_literal(val)}")
    for col, values in (in_lists or {}).items():
        rendered = ", ".join(_quote_literal(v) for v in values)
        clauses.append(f"[{col}] IN ({rendered})")
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    if order_by:
        sql += " ORDER BY " + ", ".join(f"[{c}]" for c in order_by)
    return sql


@dataclass
class DiscodataClient:
    """Thin retrying wrapper over the Discodata ``/sql`` endpoint."""

    base_url: str = DISCODATA_SQL_URL
    timeout: int = 120
    session: requests.Session = field(default_factory=requests.Session)
    user_agent: str = "powerbench/0.1 (EU HDV CO2 benchmarking demo)"

    @retry(
        retry=retry_if_exception_type((requests.RequestException, DiscodataError)),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=True,
    )
    def run_sql(self, query: str, nr_of_hits: int = DEFAULT_NR_OF_HITS) -> list[dict[str, Any]]:
        """Execute ``query`` and return the ``results`` list.

        Raises :class:`DiscodataError` on an ``errors`` payload (not retried for
        SQL-syntax errors — those re-raise immediately as they will never pass).
        """
        params = {"query": query, "nrOfHits": nr_of_hits}
        log.debug("discodata query: %s", query)
        resp = self.session.get(
            self.base_url,
            params=params,
            timeout=self.timeout,
            headers={"User-Agent": self.user_agent},
        )
        if resp.status_code >= 500:
            raise DiscodataError(f"HTTP {resp.status_code} from Discodata")
        resp.raise_for_status()

        try:
            payload = resp.json()
        except ValueError as exc:  # non-JSON (HTML error page)
            raise DiscodataError(f"non-JSON response: {resp.text[:200]}") from exc

        if isinstance(payload, dict) and payload.get("errors"):
            err = payload["errors"][0]
            code = err.get("errorcode")
            msg = f"Discodata error {code}: {err.get('error')}"
            # 10002/10003 = query-not-allowed / bad object — deterministic, do not retry
            if code in (10001, 10002, 10003):
                raise NonRetryableDiscodataError(msg)
            raise DiscodataError(msg)

        results = payload.get("results") if isinstance(payload, dict) else None
        if results is None:
            raise DiscodataError(f"unexpected payload shape: {str(payload)[:200]}")
        return results

    def select(
        self,
        table: str,
        columns: Sequence[str],
        *,
        where: Sequence[str] | None = None,
        equals: dict[str, Any] | None = None,
        in_lists: dict[str, Sequence[Any]] | None = None,
        order_by: Sequence[str] | None = None,
        top: int | None = None,
        nr_of_hits: int = DEFAULT_NR_OF_HITS,
    ) -> list[dict[str, Any]]:
        sql = build_select(
            table,
            columns,
            where=where,
            equals=equals,
            in_lists=in_lists,
            order_by=order_by,
            top=top,
        )
        return self.run_sql(sql, nr_of_hits=nr_of_hits)

    def count(
        self,
        table: str,
        *,
        where: Sequence[str] | None = None,
        equals: dict[str, Any] | None = None,
        in_lists: dict[str, Sequence[Any]] | None = None,
    ) -> int:
        sql = build_select(
            table, ["COUNT(*) AS n"], where=where, equals=equals, in_lists=in_lists
        )
        out = self.run_sql(sql, nr_of_hits=1)
        return int(out[0]["n"]) if out else 0
