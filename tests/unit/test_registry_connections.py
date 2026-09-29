"""
Tests for the metadata registry's connection layout.

The registry opens two SQLite connections, each behind its own lock:

    - a write connection (WAL, ``synchronous=NORMAL``, ``temp_store=MEMORY``)
      used for every INSERT/UPDATE/DELETE and the schema;
    - a ``query_only`` read connection used by every read path.

In WAL mode readers do not block on writers, but only across connections,
so these tests prove that a read no longer waits for a write in progress,
that reads still see every committed write, that the read connection
cannot write, that ``list_filings_page`` reads its count and page from one
snapshot, and that ``PRAGMA key`` comes first on both connections.
"""

import sqlite3
import threading
import types
from datetime import date
from unittest.mock import patch

import pytest

from sec_semantic_search.core.exceptions import DatabaseError
from sec_semantic_search.core.types import ContentType, FilingIdentifier, Segment
from sec_semantic_search.database.metadata import MetadataRegistry


def _filing(n: int, ticker: str = "AAPL") -> FilingIdentifier:
    return FilingIdentifier(
        ticker=ticker,
        form_type="10-K",
        filing_date=date(2020 + n % 5, 1 + n % 12, 1 + n % 28),
        accession_number=f"0000320193-24-{n:06d}",
    )


def _segments(filing_id: FilingIdentifier, count: int = 3) -> list[Segment]:
    return [
        Segment(
            path="Part I > Item 1",
            content_type=ContentType.TEXT,
            content=f"Segment {i} of {filing_id.accession_number}.",
            filing_id=filing_id,
            segment_index=i,
        )
        for i in range(count)
    ]


@pytest.fixture
def registry(tmp_db_path):
    reg = MetadataRegistry(db_path=tmp_db_path, encryption_key="")
    yield reg
    reg.close()


def _run_with_timeout(fn, timeout: float = 2.0):
    """Run ``fn`` in a thread; return ``(finished, result)``."""
    out: dict = {}

    def target():
        out["result"] = fn()

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    return not thread.is_alive(), out.get("result")


class TestPragmas:
    def test_write_connection_pragmas(self, registry):
        conn = registry._conn
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL
        assert conn.execute("PRAGMA temp_store").fetchone()[0] == 2  # MEMORY
        assert conn.execute("PRAGMA query_only").fetchone()[0] == 0

    def test_read_connection_pragmas(self, registry):
        conn = registry._read_conn
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA query_only").fetchone()[0] == 1
        assert conn.execute("PRAGMA temp_store").fetchone()[0] == 2

    def test_connections_and_locks_are_distinct(self, registry):
        assert registry._read_conn is not registry._conn
        assert registry._read_lock is not registry._lock


class TestReadConnectionCannotWrite:
    @pytest.mark.parametrize(
        "sql",
        [
            "DELETE FROM filings",
            "INSERT INTO task_history (task_id, status, form_types, results) "
            "VALUES ('t', 's', '[]', '[]')",
            "DROP TABLE segments",
        ],
    )
    def test_write_through_read_connection_fails(self, registry, sql):
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            registry._read_conn.execute(sql)


class TestReadsDoNotWaitForWrites:
    """A read must finish while the write lock is held."""

    @pytest.mark.parametrize(
        "read",
        [
            lambda r: r.count(),
            lambda r: r.get_statistics().filing_count,
            lambda r: r.list_filings_page()[1],
            lambda r: len(r.list_filings()),
            lambda r: r.get_filing(_filing(0).accession_number) is not None,
            lambda r: len(r.get_parent_segments([(_filing(0).accession_number, 0)])),
            lambda r: len(r.get_existing_accessions([_filing(0).accession_number])),
            lambda r: r.get_task_history("missing") is None,
        ],
        ids=[
            "count",
            "statistics",
            "page",
            "list",
            "get_filing",
            "parent_segments",
            "existing_accessions",
            "task_history",
        ],
    )
    def test_read_completes_during_open_write_transaction(self, registry, read):
        registry.register_filing(_filing(0), chunk_count=3, segments=_segments(_filing(0)))

        # Hold the write lock with an uncommitted insert in flight, as an
        # ingest does while it writes several hundred segment rows.
        with registry._lock:
            registry._conn.execute("BEGIN IMMEDIATE")
            registry._conn.execute(
                "INSERT INTO filings (ticker, form_type, filing_date, "
                "accession_number, chunk_count, ingested_at) "
                "VALUES ('MSFT', '10-Q', '2024-05-01', '0000789019-24-000001', 1, 'x')"
            )
            try:
                finished, result = _run_with_timeout(lambda: read(registry))
            finally:
                registry._conn.rollback()

        assert finished, "read blocked behind the write lock"
        assert result not in (None, 0, False)

    def test_uncommitted_rows_are_invisible_to_reads(self, registry):
        registry.register_filing(_filing(0), chunk_count=1)
        with registry._lock:
            registry._conn.execute("BEGIN IMMEDIATE")
            registry._conn.execute(
                "INSERT INTO filings (ticker, form_type, filing_date, "
                "accession_number, chunk_count, ingested_at) "
                "VALUES ('MSFT', '10-Q', '2024-05-01', '0000789019-24-000001', 1, 'x')"
            )
            try:
                finished, count = _run_with_timeout(registry.count)
            finally:
                registry._conn.rollback()
        assert finished
        assert count == 1


class TestReadYourWrites:
    def test_committed_writes_are_visible_immediately(self, registry):
        fid = _filing(1)
        registry.register_filing(fid, chunk_count=3, segments=_segments(fid))

        assert registry.count() == 1
        assert registry.get_filing(fid.accession_number).chunk_count == 3
        assert registry.get_existing_accessions([fid.accession_number]) == {fid.accession_number}
        assert len(registry.get_parent_segments([(fid.accession_number, 2)])) == 1

        registry.remove_filing(fid.accession_number)
        assert registry.count() == 0
        assert registry.get_parent_segments([(fid.accession_number, 2)]) == {}

    def test_register_if_new_sees_rows_from_the_same_registry(self, registry):
        fid = _filing(2)
        assert registry.register_filing_if_new(fid, chunk_count=1) is True
        assert registry.register_filing_if_new(fid, chunk_count=1) is False
        assert registry.count() == 1


class TestPageSnapshot:
    """``total`` and the page rows come from one read transaction."""

    def test_commit_between_count_and_page_does_not_split_them(self, registry):
        for n in range(3):
            registry.register_filing(_filing(n), chunk_count=1)

        real = registry._read_conn

        class _CommitAfterCount:
            """Commits a new filing right after the COUNT statement runs."""

            def execute(self, sql, *args):
                cursor = real.execute(sql, *args)
                if sql.startswith("SELECT COUNT(*)"):
                    registry.register_filing(_filing(99, ticker="NEW"), chunk_count=1)
                return cursor

            def __getattr__(self, name):
                return getattr(real, name)

        registry._read_conn = _CommitAfterCount()
        try:
            records, total = registry.list_filings_page(limit=200)
        finally:
            registry._read_conn = real

        assert total == len(records) == 3
        assert "NEW" not in {r.ticker for r in records}
        # The concurrent commit landed and is visible to the next read.
        assert registry.list_filings_page(limit=200)[1] == 4

    def test_transaction_is_closed_after_the_page(self, registry):
        registry.register_filing(_filing(0), chunk_count=1)
        registry.list_filings_page()
        assert registry._read_conn.in_transaction is False

    def test_transaction_is_closed_when_a_statement_fails(self, registry):
        real = registry._read_conn

        class _FailPage:
            def execute(self, sql, *args):
                if sql.startswith("SELECT * FROM filings"):
                    raise sqlite3.OperationalError("disk I/O error")
                return real.execute(sql, *args)

            def __getattr__(self, name):
                return getattr(real, name)

        registry._read_conn = _FailPage()
        try:
            with pytest.raises(DatabaseError, match="Failed to list filings"):
                registry.list_filings_page()
        finally:
            registry._read_conn = real
        assert real.in_transaction is False
        assert registry.count() == 0


class TestEncryptedConnections:
    """``PRAGMA key`` is the first statement on both connections."""

    def test_pragma_key_first_on_every_connection(self, tmp_db_path):
        statements: list[list[str]] = []

        class _Recording:
            def __init__(self, *args, **kwargs):
                object.__setattr__(self, "_real", sqlite3.connect(*args, **kwargs))
                object.__setattr__(self, "_log", [])
                statements.append(self._log)

            def execute(self, sql, *args):
                self._log.append(sql)
                if sql.startswith("PRAGMA key"):
                    return self._real.execute("SELECT 1")
                return self._real.execute(sql, *args)

            def __getattr__(self, name):
                return getattr(self._real, name)

            def __setattr__(self, name, value):
                setattr(self._real, name, value)

            def __enter__(self):
                return self._real.__enter__()

            def __exit__(self, *exc):
                return self._real.__exit__(*exc)

        module = types.ModuleType("pysqlcipher3.dbapi2")
        module.connect = _Recording
        module.Error = sqlite3.Error
        module.IntegrityError = sqlite3.IntegrityError
        module.Row = sqlite3.Row

        with patch(
            "sec_semantic_search.database.metadata._get_sqlite_module",
            return_value=module,
        ):
            registry = MetadataRegistry(db_path=tmp_db_path, encryption_key="k3y")

        try:
            assert len(statements) == 2
            hex_key = b"k3y".hex()
            for log in statements:
                assert log[0].startswith("PRAGMA key")
                assert hex_key in log[0]
                assert "k3y" not in log[0]
            assert registry.encrypted
            registry.register_filing(_filing(0), chunk_count=1)
            assert registry.count() == 1
        finally:
            registry.close()


def test_close_closes_both_connections(tmp_db_path):
    registry = MetadataRegistry(db_path=tmp_db_path, encryption_key="")
    registry.close()
    for conn in (registry._conn, registry._read_conn):
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")
