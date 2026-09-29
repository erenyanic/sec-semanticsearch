"""
Shared test helper utilities for SEC-SemanticSearch tests.

Plain functions (not pytest fixtures) that can be imported directly
by test modules. Kept separate from conftest.py because conftest.py
is for fixtures only — plain helpers must be in a regular module to
be importable via standard Python imports.
"""

from sec_semantic_search.api.tasks import TaskInfo, TaskState
from sec_semantic_search.database.metadata import FilingRecord, MetadataRegistry


def count_segments(registry: MetadataRegistry, accession_number: str | None = None) -> int:
    """Count rows in the registry's ``segments`` table, optionally for one filing.

    Test-only: production reads segments through ``get_parent_segments()``.
    """
    sql = "SELECT COUNT(*) FROM segments"
    params: tuple = ()
    if accession_number is not None:
        sql += " WHERE accession_number = ?"
        params = (accession_number,)
    with registry._read_lock:
        return registry._read_conn.execute(sql, params).fetchone()[0]


def make_filing_record(
    *,
    id: int = 1,
    ticker: str = "AAPL",
    form_type: str = "10-K",
    filing_date: str = "2024-11-01",
    accession_number: str = "0000320193-24-000001",
    chunk_count: int = 100,
    ingested_at: str = "2024-11-15T10:00:00",
) -> FilingRecord:
    """
    Factory for creating FilingRecord instances with sensible defaults.

    Not a fixture — accepts parameters so tests can create records with
    different values.
    """
    return FilingRecord(
        id=id,
        ticker=ticker,
        form_type=form_type,
        filing_date=filing_date,
        accession_number=accession_number,
        chunk_count=chunk_count,
        ingested_at=ingested_at,
    )


def make_task_info(
    *,
    task_id: str = "abc123def456",
    tickers: list[str] | None = None,
    form_types: list[str] | None = None,
    state: TaskState = TaskState.PENDING,
    count_mode: str = "latest",
    count: int | None = None,
    error: str | None = None,
) -> TaskInfo:
    """
    Factory for creating TaskInfo instances with sensible defaults.

    The cancel_event and message_queue are auto-created by the dataclass.
    """
    info = TaskInfo(
        task_id=task_id,
        tickers=tickers or ["AAPL"],
        form_types=form_types or ["10-K", "10-Q"],
        count_mode=count_mode,
        count=count,
    )
    info.state = state
    info.error = error
    return info
