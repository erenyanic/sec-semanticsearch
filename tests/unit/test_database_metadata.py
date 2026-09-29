"""
Unit tests for MetadataRegistry — get_filing and SQL injection safety.

These tests complement the integration tests in test_pipeline.py by
adding direct coverage for get_filing() (found/not-found) and verifying
that SQL injection via ticker/form_type parameters is blocked by the
parameterised query pattern.
"""

from datetime import date

import pytest

from sec_semantic_search.core.exceptions import DatabaseError
from sec_semantic_search.core.types import FilingIdentifier
from sec_semantic_search.database.metadata import MetadataRegistry


@pytest.fixture
def registry(tmp_db_path) -> MetadataRegistry:
    return MetadataRegistry(db_path=tmp_db_path)


@pytest.fixture
def stored_filing(registry, sample_filing_id) -> FilingIdentifier:
    """Register a filing and return its identifier."""
    registry.register_filing(sample_filing_id, chunk_count=42)
    return sample_filing_id


class TestGetFiling:
    """get_filing() returns a single FilingRecord or None."""

    def test_found(self, registry, stored_filing):
        record = registry.get_filing(stored_filing.accession_number)
        assert record is not None
        assert record.ticker == "AAPL"
        assert record.form_type == "10-K"
        assert record.chunk_count == 42

    def test_not_found(self, registry):
        assert registry.get_filing("NONEXISTENT") is None


class TestListFilingsOrdering:
    """list_filings() returns records in filing_date DESC order."""

    def test_descending_by_default(self, registry):
        fid_old = FilingIdentifier("AAPL", "10-K", date(2023, 1, 1), "ACC-OLD")
        fid_new = FilingIdentifier("AAPL", "10-K", date(2024, 6, 1), "ACC-NEW")
        registry.register_filing(fid_old, chunk_count=10)
        registry.register_filing(fid_new, chunk_count=20)

        filings = registry.list_filings()
        assert filings[0].filing_date > filings[1].filing_date

    def test_combined_filters(self, registry):
        fid1 = FilingIdentifier("AAPL", "10-K", date(2024, 1, 1), "ACC-1")
        fid2 = FilingIdentifier("AAPL", "10-Q", date(2024, 6, 1), "ACC-2")
        fid3 = FilingIdentifier("MSFT", "10-K", date(2024, 3, 1), "ACC-3")
        registry.register_filing(fid1, chunk_count=10)
        registry.register_filing(fid2, chunk_count=20)
        registry.register_filing(fid3, chunk_count=30)

        result = registry.list_filings(ticker="AAPL", form_type="10-K")
        assert len(result) == 1
        assert result[0].accession_number == "ACC-1"


class TestGetStatistics:
    """get_statistics() returns SQL-aggregated database statistics."""

    def test_empty_database(self, registry):
        stats = registry.get_statistics()
        assert stats.filing_count == 0
        assert stats.tickers == []
        assert stats.form_breakdown == {}
        assert stats.ticker_breakdown == []

    def test_single_filing(self, registry, stored_filing):
        stats = registry.get_statistics()
        assert stats.filing_count == 1
        assert stats.tickers == ["AAPL"]
        assert stats.form_breakdown == {"10-K": 1}
        assert len(stats.ticker_breakdown) == 1
        assert stats.ticker_breakdown[0].ticker == "AAPL"
        assert stats.ticker_breakdown[0].filings == 1
        assert stats.ticker_breakdown[0].chunks == 42
        assert stats.ticker_breakdown[0].forms == ["10-K"]

    def test_multiple_tickers_and_forms(self, registry):
        fid1 = FilingIdentifier("AAPL", "10-K", date(2024, 1, 1), "ACC-1")
        fid2 = FilingIdentifier("AAPL", "10-Q", date(2024, 6, 1), "ACC-2")
        fid3 = FilingIdentifier("MSFT", "10-K", date(2024, 3, 1), "ACC-3")
        registry.register_filing(fid1, chunk_count=10)
        registry.register_filing(fid2, chunk_count=20)
        registry.register_filing(fid3, chunk_count=30)

        stats = registry.get_statistics()
        assert stats.filing_count == 3
        assert stats.tickers == ["AAPL", "MSFT"]
        assert stats.form_breakdown == {"10-K": 2, "10-Q": 1}

        # Ticker breakdown — sorted by ticker.
        assert len(stats.ticker_breakdown) == 2
        aapl = stats.ticker_breakdown[0]
        assert aapl.ticker == "AAPL"
        assert aapl.filings == 2
        assert aapl.chunks == 30  # 10 + 20
        assert aapl.forms == ["10-K", "10-Q"]

        msft = stats.ticker_breakdown[1]
        assert msft.ticker == "MSFT"
        assert msft.filings == 1
        assert msft.chunks == 30
        assert msft.forms == ["10-K"]

    def test_chunk_sum_is_correct(self, registry):
        """Verify chunk counts are summed, not counted."""
        fid1 = FilingIdentifier("AAPL", "10-K", date(2024, 1, 1), "ACC-1")
        fid2 = FilingIdentifier("AAPL", "10-K", date(2023, 1, 1), "ACC-2")
        registry.register_filing(fid1, chunk_count=100)
        registry.register_filing(fid2, chunk_count=250)

        stats = registry.get_statistics()
        assert stats.ticker_breakdown[0].chunks == 350


class TestGetExistingAccessions:
    """get_existing_accessions() returns the subset that already exist."""

    def test_empty_input(self, registry):
        """Empty list should return empty set without hitting the database."""
        assert registry.get_existing_accessions([]) == set()

    def test_no_duplicates(self, registry):
        """None of the accession numbers exist — should return empty set."""
        result = registry.get_existing_accessions(["ACC-X", "ACC-Y", "ACC-Z"])
        assert result == set()

    def test_all_duplicates(self, registry, stored_filing):
        """All accession numbers exist — should return all of them."""
        result = registry.get_existing_accessions([stored_filing.accession_number])
        assert result == {stored_filing.accession_number}

    def test_some_duplicates(self, registry):
        """Mix of existing and non-existing accession numbers."""
        fid1 = FilingIdentifier("AAPL", "10-K", date(2024, 1, 1), "ACC-1")
        fid2 = FilingIdentifier("MSFT", "10-K", date(2024, 3, 1), "ACC-2")
        registry.register_filing(fid1, chunk_count=10)
        registry.register_filing(fid2, chunk_count=20)

        result = registry.get_existing_accessions(["ACC-1", "ACC-MISSING", "ACC-2", "ACC-NOPE"])
        assert result == {"ACC-1", "ACC-2"}

    def test_sql_injection_safety(self, registry, stored_filing):
        """Malicious accession numbers should not break the query."""
        result = registry.get_existing_accessions(["' OR '1'='1", "'; DROP TABLE filings; --"])
        assert result == set()
        # Table should still be intact.
        assert registry.count() == 1


class TestSQLInjectionSafety:
    """Malicious inputs should not break queries or leak data."""

    def test_ticker_with_sql_injection(self, registry, stored_filing):
        """SQL injection in ticker param should match nothing, not error."""
        filings = registry.list_filings(ticker="' OR 1=1 --")
        assert filings == []

    def test_form_type_with_sql_injection(self, registry, stored_filing):
        filings = registry.list_filings(form_type="'; DROP TABLE filings; --")
        assert filings == []
        # Table should still exist — verify by counting.
        assert registry.count() == 1

    def test_accession_with_special_characters(self, registry, stored_filing):
        """Special characters in accession number should not break queries."""
        assert registry.get_existing_accessions(["' OR '1'='1"]) == set()
        assert registry.get_filing("' UNION SELECT * FROM filings --") is None

    def test_count_with_sql_injection(self, registry, stored_filing):
        """count() with malicious filters should return 0, not error."""
        assert registry.count(ticker="' OR 1=1 --") == 0
        assert registry.count(form_type="'; DROP TABLE filings; --") == 0
        # Verify table still intact.
        assert registry.count() == 1


class TestRegisterFilingIfNew:
    """register_filing_if_new() atomically checks and inserts."""

    def test_registers_new_filing(self, registry, sample_filing_id):
        """New filing should be registered and return True."""
        result = registry.register_filing_if_new(sample_filing_id, chunk_count=42)
        assert result is True
        assert registry.count() == 1

    def test_returns_false_for_existing(self, registry, sample_filing_id):
        """Existing filing should return False without error."""
        registry.register_filing(sample_filing_id, chunk_count=42)
        result = registry.register_filing_if_new(sample_filing_id, chunk_count=42)
        assert result is False
        # Should still be exactly 1 filing.
        assert registry.count() == 1

    def test_atomicity_under_contention(self, registry):
        """Only one thread should succeed when racing to register the same filing."""
        import threading

        fid = FilingIdentifier("AAPL", "10-K", date(2024, 1, 1), "ACC-RACE")
        results: list[bool] = []
        errors: list[Exception] = []
        barrier = threading.Barrier(10)

        def try_register() -> None:
            try:
                barrier.wait()  # Synchronise start
                registered = registry.register_filing_if_new(fid, chunk_count=10)
                results.append(registered)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=try_register) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Unexpected errors: {errors}"
        # Exactly one thread should have registered the filing.
        assert results.count(True) == 1
        assert results.count(False) == 9
        assert registry.count() == 1

    def test_different_filings_all_succeed(self, registry):
        """Distinct filings should all be registered concurrently."""
        import threading

        results: list[bool] = []
        errors: list[Exception] = []

        def register(i: int) -> None:
            try:
                fid = FilingIdentifier(
                    "AAPL",
                    "10-K",
                    date(2020 + i, 1, 1),
                    f"ACC-ATOMIC-{i}",
                )
                registered = registry.register_filing_if_new(fid, chunk_count=i)
                results.append(registered)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=register, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Unexpected errors: {errors}"
        assert all(results)
        assert registry.count() == 10


class TestPersistentConnection:
    """Verify the persistent connection and close() behaviour."""

    def test_close_prevents_further_operations(self, registry):
        """After close(), operations should raise (connection is closed)."""
        registry.close()
        with pytest.raises(DatabaseError, match="Failed to count filings"):
            registry.count()

    def test_thread_safety(self, registry, sample_filing_id):
        """Concurrent writes from multiple threads should not corrupt data."""
        import threading

        errors: list[Exception] = []

        def register(i: int) -> None:
            try:
                from datetime import date as _date

                fid = FilingIdentifier(
                    "AAPL",
                    "10-K",
                    _date(2020 + i, 1, 1),
                    f"ACC-THREAD-{i}",
                )
                registry.register_filing(fid, chunk_count=i)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=register, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        assert registry.count() == 10

    def test_wal_mode_enabled(self, registry):
        """WAL journal mode should be active on the persistent connection."""
        with registry._lock:
            row = registry._conn.execute("PRAGMA journal_mode").fetchone()
        assert row[0] == "wal"
