"""
Integration tests for the CLI commands via Typer's CliRunner.

CliRunner invokes commands programmatically without spawning subprocesses.
We mock heavy dependencies (fetcher, databases, embedder) to keep tests
fast while verifying exit codes, output messages, and command routing.
"""

import logging
import re
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from sec_semantic_search.cli.main import app
from sec_semantic_search.database import delete_filings_batch
from sec_semantic_search.database.metadata import DatabaseStatistics
from tests.helpers import make_filing_record

runner = CliRunner()


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from Rich-rendered CLI help output."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


# -----------------------------------------------------------------------
# Root app and version
# -----------------------------------------------------------------------


class TestRootApp:
    """The root app should show help and version."""

    def test_no_args_shows_help(self):
        """
        Typer's no_args_is_help=True exits with code 0 when run
        normally, but CliRunner returns exit code 0 or 2 depending on
        Click version. We just verify help text is shown.
        """
        result = runner.invoke(app, [])
        assert "Usage" in result.output or "sec-search" in result.output

    def test_help_flag(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "search" in result.output
        assert "ingest" in result.output
        assert "manage" in result.output

    def test_version_flag(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "sec-search" in result.output


# -----------------------------------------------------------------------
# manage status
# -----------------------------------------------------------------------


class TestManageStatus:
    """manage status should display database statistics."""

    def test_empty_database(self, tmp_db_path, tmp_chroma_path):
        with (
            patch("sec_semantic_search.cli.manage.MetadataRegistry") as MockReg,
            patch("sec_semantic_search.cli.manage.ChromaDBClient") as MockChroma,
            patch("sec_semantic_search.cli.manage.get_settings") as MockSettings,
        ):
            mock_registry = MagicMock()
            mock_registry.get_statistics.return_value = DatabaseStatistics(
                filing_count=0,
                tickers=[],
                form_breakdown={},
                ticker_breakdown=[],
            )
            MockReg.return_value = mock_registry

            mock_chroma = MagicMock()
            mock_chroma.collection_count.return_value = 0
            MockChroma.return_value = mock_chroma

            mock_settings = MagicMock()
            mock_settings.database.max_filings = 20
            MockSettings.return_value = mock_settings

            result = runner.invoke(app, ["manage", "status"])

        assert result.exit_code == 0
        assert "Database Status" in result.output
        assert "0" in result.output


# -----------------------------------------------------------------------
# manage list
# -----------------------------------------------------------------------


class TestManageList:
    """manage list should show filings or a 'no filings' message."""

    def test_empty_list(self):
        with patch("sec_semantic_search.cli.manage.MetadataRegistry") as MockReg:
            mock_registry = MagicMock()
            mock_registry.list_filings.return_value = []
            MockReg.return_value = mock_registry

            result = runner.invoke(app, ["manage", "list"])

        assert result.exit_code == 0
        assert "No filings found" in result.output


# -----------------------------------------------------------------------
# manage remove
# -----------------------------------------------------------------------


class TestManageRemove:
    """manage remove should handle not-found, successful, and cancelled removal."""

    def test_not_found(self):
        with patch("sec_semantic_search.cli.manage.MetadataRegistry") as MockReg:
            mock_registry = MagicMock()
            mock_registry.get_filing.return_value = None
            MockReg.return_value = mock_registry

            result = runner.invoke(app, ["manage", "remove", "NONEXISTENT"])

        assert result.exit_code == 1
        assert "not found" in result.output.lower()
        assert "NONEXISTENT" in result.output

    def test_successful_removal_with_yes(self):
        """--yes bypasses confirmation and removes the filing."""
        record = make_filing_record(accession_number="ACC-001")
        with (
            patch("sec_semantic_search.cli.manage.MetadataRegistry") as MockReg,
            patch("sec_semantic_search.cli.manage.ChromaDBClient") as MockChroma,
        ):
            mock_registry = MagicMock()
            mock_registry.get_filing.return_value = record
            MockReg.return_value = mock_registry

            mock_chroma = MagicMock()
            MockChroma.return_value = mock_chroma

            result = runner.invoke(app, ["manage", "remove", "ACC-001", "--yes"])

        assert result.exit_code == 0
        assert "Removed" in result.output
        assert "100 chunks" in result.output  # from FilingRecord.chunk_count default
        # The shared delete helper: ChromaDB first, then SQLite.
        mock_chroma.delete_filings_batch.assert_called_once_with(["ACC-001"])
        mock_registry.remove_filings_batch.assert_called_once_with(["ACC-001"])

    def test_confirmation_rejected(self):
        """Answering 'n' to the confirmation prompt should cancel removal."""
        record = make_filing_record(accession_number="ACC-001")
        with patch("sec_semantic_search.cli.manage.MetadataRegistry") as MockReg:
            mock_registry = MagicMock()
            mock_registry.get_filing.return_value = record
            MockReg.return_value = mock_registry

            result = runner.invoke(app, ["manage", "remove", "ACC-001"], input="n\n")

        assert "Cancelled" in result.output


# -----------------------------------------------------------------------
# manage remove — bulk deletion
# -----------------------------------------------------------------------


class TestBulkRemove:
    """manage remove --ticker/--form should delete matching filings in bulk."""

    def test_bulk_remove_by_ticker(self):
        records = [
            make_filing_record(id=1, accession_number="ACC-001"),
            make_filing_record(id=2, accession_number="ACC-002"),
        ]
        with (
            patch("sec_semantic_search.cli.manage.MetadataRegistry") as MockReg,
            patch("sec_semantic_search.cli.manage.ChromaDBClient") as MockChroma,
        ):
            mock_registry = MagicMock()
            mock_registry.list_filings.return_value = records
            MockReg.return_value = mock_registry

            mock_chroma = MagicMock()
            MockChroma.return_value = mock_chroma

            result = runner.invoke(app, ["manage", "remove", "--ticker", "AAPL", "--yes"])

        assert result.exit_code == 0
        assert "2 filing(s) removed" in result.output

    def test_bulk_remove_no_matches(self):
        with patch("sec_semantic_search.cli.manage.MetadataRegistry") as MockReg:
            mock_registry = MagicMock()
            mock_registry.list_filings.return_value = []
            MockReg.return_value = mock_registry

            result = runner.invoke(app, ["manage", "remove", "--ticker", "ZZZZ", "--yes"])

        assert "No filings found" in result.output

    def test_mutual_exclusion_accession_and_ticker(self):
        """Providing both an accession number and --ticker should fail."""
        result = runner.invoke(app, ["manage", "remove", "ACC-001", "--ticker", "AAPL"])
        assert result.exit_code == 1
        assert "Cannot combine" in result.output

    def test_no_args_no_filters(self):
        """Providing neither accession nor filters should fail."""
        result = runner.invoke(app, ["manage", "remove"])
        assert result.exit_code == 1
        assert (
            "Provide an accession" in result.output.lower()
            or "provide an accession" in result.output.lower()
        )

    def test_bulk_remove_cancelled(self):
        """Answering 'n' to bulk remove confirmation should cancel."""
        records = [make_filing_record(accession_number="ACC-001")]
        with (
            patch("sec_semantic_search.cli.manage.MetadataRegistry") as MockReg,
            patch("sec_semantic_search.cli.manage.ChromaDBClient") as MockChroma,
        ):
            mock_registry = MagicMock()
            mock_registry.list_filings.return_value = records
            MockReg.return_value = mock_registry
            MockChroma.return_value = MagicMock()

            result = runner.invoke(app, ["manage", "remove", "--ticker", "AAPL"], input="n\n")

        assert "Cancelled" in result.output


# -----------------------------------------------------------------------
# manage clear
# -----------------------------------------------------------------------


class TestManageClear:
    """manage clear should delete all filings or report empty database."""

    def test_clear_with_yes(self):
        records = [
            make_filing_record(id=1, accession_number="ACC-001"),
            make_filing_record(id=2, accession_number="ACC-002"),
        ]
        with (
            patch("sec_semantic_search.cli.manage.MetadataRegistry") as MockReg,
            patch("sec_semantic_search.cli.manage.ChromaDBClient") as MockChroma,
        ):
            mock_registry = MagicMock()
            mock_registry.list_filings.return_value = records
            MockReg.return_value = mock_registry

            mock_chroma = MagicMock()
            MockChroma.return_value = mock_chroma

            result = runner.invoke(app, ["manage", "clear", "--yes"])

        assert result.exit_code == 0
        assert "Database cleared" in result.output

    def test_clear_empty_database(self):
        with patch("sec_semantic_search.cli.manage.MetadataRegistry") as MockReg:
            mock_registry = MagicMock()
            mock_registry.list_filings.return_value = []
            MockReg.return_value = mock_registry

            result = runner.invoke(app, ["manage", "clear", "--yes"])

        assert result.exit_code == 0
        assert "already empty" in result.output.lower()

    def test_clear_cancelled(self):
        records = [make_filing_record(accession_number="ACC-001")]
        with (
            patch("sec_semantic_search.cli.manage.MetadataRegistry") as MockReg,
            patch("sec_semantic_search.cli.manage.ChromaDBClient") as MockChroma,
        ):
            mock_registry = MagicMock()
            mock_registry.list_filings.return_value = records
            MockReg.return_value = mock_registry
            MockChroma.return_value = MagicMock()

            result = runner.invoke(app, ["manage", "clear"], input="n\n")

        assert "Cancelled" in result.output


# -----------------------------------------------------------------------
# delete_filings_batch() helper
# -----------------------------------------------------------------------


class TestDeleteFilingsBatch:
    """delete_filings_batch() orchestrates deletion across both stores."""

    def test_deletes_multiple_returns_total_chunks(self):
        """Total chunks come from FilingRecord.chunk_count, not ChromaDB return."""
        records = [
            make_filing_record(id=1, accession_number="ACC-001", chunk_count=50),
            make_filing_record(id=2, accession_number="ACC-002", chunk_count=50),
        ]
        mock_chroma = MagicMock()
        mock_registry = MagicMock()

        total = delete_filings_batch(records, registry=mock_registry, chroma=mock_chroma)

        assert total == 100  # 50 + 50 from FilingRecord.chunk_count
        mock_chroma.delete_filings_batch.assert_called_once_with(
            ["ACC-001", "ACC-002"],
        )
        mock_registry.remove_filings_batch.assert_called_once_with(
            ["ACC-001", "ACC-002"],
        )

    def test_chromadb_called_before_sqlite(self):
        """Deletion order must be ChromaDB first, then SQLite."""
        record = make_filing_record(accession_number="ACC-001")
        call_order = []

        mock_chroma = MagicMock()
        mock_chroma.delete_filings_batch.side_effect = lambda accs: call_order.append(
            ("chroma", accs)
        )
        mock_registry = MagicMock()
        mock_registry.remove_filings_batch.side_effect = lambda accs: call_order.append(
            ("registry", accs)
        )

        delete_filings_batch([record], registry=mock_registry, chroma=mock_chroma)

        assert call_order == [
            ("chroma", ["ACC-001"]),
            ("registry", ["ACC-001"]),
        ]

    def test_empty_list_returns_zero(self):
        mock_chroma = MagicMock()
        mock_registry = MagicMock()

        total = delete_filings_batch([], registry=mock_registry, chroma=mock_chroma)

        assert total == 0
        mock_chroma.delete_filings_batch.assert_not_called()
        mock_registry.remove_filings_batch.assert_not_called()

    @staticmethod
    def _records(n):
        return [
            make_filing_record(
                id=i,
                ticker=f"T{i % 3}",
                accession_number=f"ACC-{i:04d}",
                chunk_count=10,
            )
            for i in range(n)
        ]

    def _delete_and_capture(self, n, level):
        """Delete ``n`` filings and return the chunk total and log records."""
        records: list[logging.LogRecord] = []

        class _Handler(logging.Handler):
            def emit(self, record):
                records.append(record)

        logger = logging.getLogger("sec_semantic_search.database")
        handler = _Handler(level=logging.DEBUG)
        previous = logger.level
        logger.addHandler(handler)
        logger.setLevel(level)
        try:
            total = delete_filings_batch(self._records(n), registry=MagicMock(), chroma=MagicMock())
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous)
        return total, records

    def test_bulk_delete_logs_one_info_summary(self):
        """A 500-filing eviction emits one INFO line, not 500."""
        total, records = self._delete_and_capture(500, logging.INFO)

        info = [r for r in records if r.levelno == logging.INFO]
        assert total == 5000
        assert len(info) == 1
        assert info[0].getMessage() == "Deleted 500 filing(s) across 3 ticker(s) — 5000 chunks"

    def test_summary_names_no_ticker(self):
        """Ticker names stay out of the INFO summary (research pattern, AD#29)."""
        _, records = self._delete_and_capture(3, logging.INFO)
        messages = [r.getMessage() for r in records]
        assert not any("T0" in m or "T1" in m for m in messages)

    def test_per_filing_detail_at_debug(self):
        _, records = self._delete_and_capture(4, logging.DEBUG)

        debug = [r for r in records if r.levelno == logging.DEBUG]
        assert len(debug) == 4
        assert all(r.getMessage().startswith("Deleted T") for r in debug)


# -----------------------------------------------------------------------
# ingest add -t (cross-form)
# -----------------------------------------------------------------------


class TestIngestAcrossFormsFetch:
    """``ingest add -t`` fetches from the listing's cached Filing objects."""

    def test_fetches_via_cached_filing_objects(self):
        from datetime import date

        from sec_semantic_search.core.types import FilingIdentifier
        from sec_semantic_search.pipeline.fetch import FilingInfo

        infos = [
            FilingInfo(
                ticker="AAPL",
                form_type=form,
                filing_date=date(2024, month, 1),
                accession_number=f"0000320193-24-00000{month}",
                company_name="Apple Inc.",
                _filing_obj=MagicMock(),
            )
            for month, form in ((3, "10-Q"), (2, "10-K"))
        ]

        def fetched(info):
            return (
                FilingIdentifier(
                    ticker=info.ticker,
                    form_type=info.form_type,
                    filing_date=info.filing_date,
                    accession_number=info.accession_number,
                ),
                "<html></html>",
            )

        with (
            patch("sec_semantic_search.cli.ingest.MetadataRegistry") as registry_cls,
            patch("sec_semantic_search.cli.ingest.ChromaDBClient"),
            patch("sec_semantic_search.cli.ingest.FilingFetcher") as fetcher_cls,
            patch("sec_semantic_search.cli.ingest.PipelineOrchestrator") as orchestrator_cls,
        ):
            registry = registry_cls.return_value
            registry.get_existing_accessions.return_value = set()
            registry.count.return_value = 0
            registry.register_filing_if_new.return_value = True
            fetcher = fetcher_cls.return_value
            fetcher.list_available_across_forms.return_value = infos
            fetcher.fetch_filing_content.side_effect = fetched
            processed = MagicMock()
            processed.ingest_result.segment_count = 2
            processed.ingest_result.chunk_count = 3
            processed.ingest_result.duration_seconds = 0.1
            orchestrator_cls.return_value.process_filing.return_value = processed

            result = runner.invoke(app, ["ingest", "add", "AAPL", "-t", "2"])

        assert result.exit_code == 0, result.output
        assert [c.args[0] for c in fetcher.fetch_filing_content.call_args_list] == infos
        fetcher.fetch_by_accession.assert_not_called()
        assert "2 ingested" in result.output


# -----------------------------------------------------------------------
# ingest add / batch on the shared runner
# -----------------------------------------------------------------------


def _filing_info(ticker, form, n):
    from datetime import date

    from sec_semantic_search.pipeline.fetch import FilingInfo

    return FilingInfo(
        ticker=ticker,
        form_type=form,
        filing_date=date(2024, 1, 1 + n),
        accession_number=f"0000320193-24-{n:06d}",
        company_name=ticker,
        _filing_obj=MagicMock(),
    )


@pytest.fixture
def cli_ingest():
    """Patch the CLI's stores, fetcher and pipeline; storage succeeds."""
    import threading

    from sec_semantic_search.core.types import FilingIdentifier

    fetch_threads: list[str] = []

    def fetched(info):
        fetch_threads.append(threading.current_thread().name)
        return (
            FilingIdentifier(info.ticker, info.form_type, info.filing_date, info.accession_number),
            "<html></html>",
        )

    def processed(filing_id, html, progress_callback=None):
        result = MagicMock()
        result.filing_id = filing_id
        result.ingest_result.segment_count = 2
        result.ingest_result.chunk_count = 3
        result.ingest_result.duration_seconds = 0.1
        return result

    with (
        patch("sec_semantic_search.cli.ingest.MetadataRegistry") as registry_cls,
        patch("sec_semantic_search.cli.ingest.ChromaDBClient") as chroma_cls,
        patch("sec_semantic_search.cli.ingest.FilingFetcher") as fetcher_cls,
        patch("sec_semantic_search.cli.ingest.PipelineOrchestrator") as orchestrator_cls,
    ):
        registry = registry_cls.return_value
        registry.get_existing_accessions.return_value = set()
        registry.count.return_value = 0
        registry.register_filing_if_new.return_value = True
        fetcher = fetcher_cls.return_value
        fetcher.list_available.side_effect = lambda t, f, **kw: [_filing_info(t, f, 1)]
        fetcher.fetch_filing_content.side_effect = fetched
        orchestrator_cls.return_value.process_filing.side_effect = processed
        yield MagicMock(
            registry=registry,
            chroma=chroma_cls.return_value,
            fetcher=fetcher,
            fetch_threads=fetch_threads,
        )


class TestIngestCommands:
    """``ingest add`` and ``ingest batch`` run the shared runner."""

    def test_add_default_lists_the_latest_per_form(self, cli_ingest):
        result = runner.invoke(app, ["ingest", "add", "aapl"])

        assert result.exit_code == 0, result.output
        calls = cli_ingest.fetcher.list_available.call_args_list
        assert [c.args for c in calls] == [("AAPL", "10-K"), ("AAPL", "10-Q")]
        assert {c.kwargs["count"] for c in calls} == {1}
        assert result.output.count("Ingested") == 2
        assert "2 ingested" in result.output

    def test_add_stores_sqlite_first(self, cli_ingest):
        order: list[str] = []
        cli_ingest.registry.register_filing_if_new.side_effect = lambda *a, **k: (
            order.append("sqlite") or True
        )
        cli_ingest.chroma.store_filing.side_effect = lambda r: order.append("chroma")

        result = runner.invoke(app, ["ingest", "add", "AAPL", "-f", "10-K"])

        assert result.exit_code == 0, result.output
        assert order == ["sqlite", "chroma"]
        cli_ingest.registry.register_filing.assert_not_called()

    def test_add_fetches_on_the_prefetch_thread(self, cli_ingest):
        runner.invoke(app, ["ingest", "add", "AAPL"])
        assert cli_ingest.fetch_threads
        assert all(name.startswith("prefetch") for name in cli_ingest.fetch_threads)

    @pytest.mark.parametrize(
        ("args", "count", "year"),
        [
            (["-n", "2"], 2, None),
            (["-y", "2023"], None, 2023),  # filters, no count: all matching
        ],
    )
    def test_add_count_options(self, cli_ingest, args, count, year):
        result = runner.invoke(app, ["ingest", "add", "AAPL", "-f", "10-K", *args])

        assert result.exit_code == 0, result.output
        kwargs = cli_ingest.fetcher.list_available.call_args.kwargs
        assert (kwargs["count"], kwargs["year"]) == (count, year)

    def test_add_skips_duplicates(self, cli_ingest):
        cli_ingest.registry.get_existing_accessions.return_value = {
            _filing_info("AAPL", "10-K", 1).accession_number
        }
        result = runner.invoke(app, ["ingest", "add", "AAPL", "-f", "10-K"])

        assert result.exit_code == 0
        assert "Already ingested" in result.output
        cli_ingest.fetcher.fetch_filing_content.assert_not_called()

    def test_add_exits_1_when_every_fetch_fails(self, cli_ingest):
        from sec_semantic_search.core.exceptions import FetchError

        cli_ingest.fetcher.fetch_filing_content.side_effect = FetchError("offline")
        result = runner.invoke(app, ["ingest", "add", "AAPL"])

        assert result.exit_code == 1
        assert "Fetch failed" in result.output

    def test_add_exits_1_when_every_listing_fails(self, cli_ingest):
        from sec_semantic_search.core.exceptions import FetchError

        cli_ingest.fetcher.list_available.side_effect = FetchError("unknown ticker")
        result = runner.invoke(app, ["ingest", "add", "ZZZZ"])

        assert result.exit_code == 1
        assert "listing failed" in result.output

    def test_add_stops_at_the_filing_limit(self, cli_ingest):
        with patch("sec_semantic_search.cli.ingest.get_settings") as settings:
            settings.return_value.database.max_filings = 0
            result = runner.invoke(app, ["ingest", "add", "AAPL"])

        assert result.exit_code == 1
        assert "Filing limit reached" in result.output
        cli_ingest.chroma.store_filing.assert_not_called()

    def test_add_total_and_number_are_exclusive(self, cli_ingest):
        result = runner.invoke(app, ["ingest", "add", "AAPL", "-t", "2", "-n", "2"])
        assert result.exit_code == 1
        assert "mutually exclusive" in result.output

    def test_batch_total_lists_across_forms_per_ticker(self, cli_ingest):
        cli_ingest.fetcher.list_available_across_forms.side_effect = lambda t, forms, **kw: [
            _filing_info(t, "10-Q", 1 if t == "AAPL" else 2)
        ]
        result = runner.invoke(app, ["ingest", "batch", "AAPL", "MSFT", "-t", "3"])

        assert result.exit_code == 0, result.output
        calls = cli_ingest.fetcher.list_available_across_forms.call_args_list
        assert [c.args[0] for c in calls] == ["AAPL", "MSFT"]
        assert {c.kwargs["count"] for c in calls} == {3}
        assert "Batch complete" in result.output
        assert "2 ingested" in result.output

    def test_no_filings_found(self, cli_ingest):
        cli_ingest.fetcher.list_available.side_effect = lambda *a, **k: []
        result = runner.invoke(app, ["ingest", "add", "AAPL"])

        assert result.exit_code == 0
        assert "No filings found" in result.output


# -----------------------------------------------------------------------
# search
# -----------------------------------------------------------------------


class TestSearchCommand:
    """The search command should display results or 'no results'."""

    @pytest.fixture(autouse=True)
    def _isolate_registry(self):
        """Keep the search command away from the real metadata database.

        ``cli.search`` builds its own ``MetadataRegistry`` so the engine can
        resolve parent segments. The import is function-local, so patch the
        name on ``sec_semantic_search.database`` where it is looked up.
        """
        with patch("sec_semantic_search.database.MetadataRegistry") as MockRegistry:
            MockRegistry.return_value = MagicMock()
            yield MockRegistry

    def test_no_results(self):
        with patch("sec_semantic_search.cli.search.SearchEngine") as MockEngine:
            mock_engine = MagicMock()
            mock_engine.search.return_value = []
            MockEngine.return_value = mock_engine

            result = runner.invoke(app, ["search", "test query"])

        assert result.exit_code == 0
        assert "No results found" in result.output

    def test_registry_database_error_is_reported(self, _isolate_registry):
        """An unreadable metadata database should exit cleanly, not traceback.

        ``MetadataRegistry()`` is constructed inside the search command, so a
        SQLCipher/permissions failure surfaces there rather than from the
        engine. Without explicit handling it escaped as an unhandled
        exception.
        """
        from sec_semantic_search.core.exceptions import DatabaseError

        _isolate_registry.side_effect = DatabaseError("file is not a database")

        with patch("sec_semantic_search.cli.search.SearchEngine"):
            result = runner.invoke(app, ["search", "test query"])

        assert result.exit_code == 1
        assert result.exception is None or isinstance(result.exception, SystemExit)
        assert "Search failed" in result.output
        assert "file is not a database" in result.output

    def test_search_error(self):
        from sec_semantic_search.core.exceptions import SearchError

        with patch("sec_semantic_search.cli.search.SearchEngine") as MockEngine:
            mock_engine = MagicMock()
            mock_engine.search.side_effect = SearchError("Search failed", details="No filings")
            MockEngine.return_value = mock_engine

            result = runner.invoke(app, ["search", "test query"])

        assert result.exit_code == 1
        assert "Search failed" in result.output

    def test_accession_filter_passed_to_engine(self):
        """--accession/-a passes accession_number to SearchEngine.search()."""
        with patch("sec_semantic_search.cli.search.SearchEngine") as MockEngine:
            mock_engine = MagicMock()
            mock_engine.search.return_value = []
            MockEngine.return_value = mock_engine

            result = runner.invoke(
                app, ["search", "test query", "--accession", "0000320193-23-000106"]
            )

        assert result.exit_code == 0
        mock_engine.search.assert_called_once_with(
            query="test query",
            top_k=None,
            ticker=None,
            form_type=None,
            accession_number=["0000320193-23-000106"],
            start_date=None,
            end_date=None,
        )

    def test_accession_short_flag(self):
        """The -a short flag should work identically to --accession."""
        with patch("sec_semantic_search.cli.search.SearchEngine") as MockEngine:
            mock_engine = MagicMock()
            mock_engine.search.return_value = []
            MockEngine.return_value = mock_engine

            result = runner.invoke(app, ["search", "test query", "-a", "0000320193-23-000106"])

        assert result.exit_code == 0
        mock_engine.search.assert_called_once_with(
            query="test query",
            top_k=None,
            ticker=None,
            form_type=None,
            accession_number=["0000320193-23-000106"],
            start_date=None,
            end_date=None,
        )

    def test_accession_combined_with_other_filters(self):
        """--accession can be used alongside --ticker and --form."""
        with patch("sec_semantic_search.cli.search.SearchEngine") as MockEngine:
            mock_engine = MagicMock()
            mock_engine.search.return_value = []
            MockEngine.return_value = mock_engine

            result = runner.invoke(
                app,
                [
                    "search",
                    "test query",
                    "-k",
                    "AAPL",
                    "-f",
                    "10-K",
                    "-a",
                    "0000320193-23-000106",
                    "-t",
                    "3",
                ],
            )

        assert result.exit_code == 0
        mock_engine.search.assert_called_once_with(
            query="test query",
            top_k=3,
            ticker=["AAPL"],
            form_type=["10-K"],
            accession_number=["0000320193-23-000106"],
            start_date=None,
            end_date=None,
        )

    def test_multi_ticker_filter(self):
        """Repeating --ticker/-k passes multiple tickers as a list."""
        with patch("sec_semantic_search.cli.search.SearchEngine") as MockEngine:
            mock_engine = MagicMock()
            mock_engine.search.return_value = []
            MockEngine.return_value = mock_engine

            result = runner.invoke(app, ["search", "test query", "-k", "AAPL", "-k", "MSFT"])

        assert result.exit_code == 0
        mock_engine.search.assert_called_once_with(
            query="test query",
            top_k=None,
            ticker=["AAPL", "MSFT"],
            form_type=None,
            accession_number=None,
            start_date=None,
            end_date=None,
        )

    def test_multi_form_filter(self):
        """Repeating --form/-f passes multiple form types as a list."""
        with patch("sec_semantic_search.cli.search.SearchEngine") as MockEngine:
            mock_engine = MagicMock()
            mock_engine.search.return_value = []
            MockEngine.return_value = mock_engine

            result = runner.invoke(app, ["search", "test query", "-f", "10-K", "-f", "10-Q"])

        assert result.exit_code == 0
        mock_engine.search.assert_called_once_with(
            query="test query",
            top_k=None,
            ticker=None,
            form_type=["10-K", "10-Q"],
            accession_number=None,
            start_date=None,
            end_date=None,
        )

    def test_multi_accession_filter(self):
        """Repeating --accession/-a passes multiple accession numbers."""
        with patch("sec_semantic_search.cli.search.SearchEngine") as MockEngine:
            mock_engine = MagicMock()
            mock_engine.search.return_value = []
            MockEngine.return_value = mock_engine

            result = runner.invoke(
                app,
                [
                    "search",
                    "test query",
                    "-a",
                    "0000320193-23-000106",
                    "-a",
                    "0000320193-23-000107",
                ],
            )

        assert result.exit_code == 0
        mock_engine.search.assert_called_once_with(
            query="test query",
            top_k=None,
            ticker=None,
            form_type=None,
            accession_number=["0000320193-23-000106", "0000320193-23-000107"],
            start_date=None,
            end_date=None,
        )

    def test_accession_appears_in_help(self):
        """--accession should appear in the search --help output."""
        result = runner.invoke(app, ["search", "--help"])
        assert result.exit_code == 0
        output = _strip_ansi(result.output)
        assert "--accession" in output
        assert "-a" in output

    def test_start_date_passed_to_engine(self):
        """--start-date passes start_date to SearchEngine.search()."""
        with patch("sec_semantic_search.cli.search.SearchEngine") as MockEngine:
            mock_engine = MagicMock()
            mock_engine.search.return_value = []
            MockEngine.return_value = mock_engine

            result = runner.invoke(app, ["search", "test query", "--start-date", "2023-01-01"])

        assert result.exit_code == 0
        mock_engine.search.assert_called_once_with(
            query="test query",
            top_k=None,
            ticker=None,
            form_type=None,
            accession_number=None,
            start_date="2023-01-01",
            end_date=None,
        )

    def test_end_date_passed_to_engine(self):
        """--end-date passes end_date to SearchEngine.search()."""
        with patch("sec_semantic_search.cli.search.SearchEngine") as MockEngine:
            mock_engine = MagicMock()
            mock_engine.search.return_value = []
            MockEngine.return_value = mock_engine

            result = runner.invoke(app, ["search", "test query", "--end-date", "2023-12-31"])

        assert result.exit_code == 0
        mock_engine.search.assert_called_once_with(
            query="test query",
            top_k=None,
            ticker=None,
            form_type=None,
            accession_number=None,
            start_date=None,
            end_date="2023-12-31",
        )

    def test_date_range_combined(self):
        """--start-date and --end-date can be used together."""
        with patch("sec_semantic_search.cli.search.SearchEngine") as MockEngine:
            mock_engine = MagicMock()
            mock_engine.search.return_value = []
            MockEngine.return_value = mock_engine

            result = runner.invoke(
                app,
                [
                    "search",
                    "test query",
                    "--start-date",
                    "2023-01-01",
                    "--end-date",
                    "2023-12-31",
                ],
            )

        assert result.exit_code == 0
        mock_engine.search.assert_called_once_with(
            query="test query",
            top_k=None,
            ticker=None,
            form_type=None,
            accession_number=None,
            start_date="2023-01-01",
            end_date="2023-12-31",
        )

    def test_date_range_appears_in_help(self):
        """--start-date and --end-date should appear in search --help."""
        result = runner.invoke(app, ["search", "--help"])
        assert result.exit_code == 0
        output = _strip_ansi(result.output)
        assert "--start-date" in output
        assert "--end-date" in output


# -----------------------------------------------------------------------
# ingest add — validation
# -----------------------------------------------------------------------


class TestIngestAddValidation:
    """ingest add should validate form types before doing work."""

    def test_unsupported_form_type(self):
        result = runner.invoke(app, ["ingest", "add", "AAPL", "--form", "20-F"])
        assert result.exit_code == 1
        assert "Unsupported" in result.output

    def test_multi_form_type_accepted(self):
        """Comma-separated valid forms should pass validation."""
        with patch("sec_semantic_search.cli.ingest.FilingFetcher") as MockFetcher:
            from sec_semantic_search.core.exceptions import FetchError

            mock_fetcher = MagicMock()
            mock_fetcher.fetch_latest.side_effect = FetchError("No network")
            MockFetcher.return_value = mock_fetcher

            result = runner.invoke(app, ["ingest", "add", "AAPL", "--form", "10-K,10-Q"])

        # The form type validation should pass — any error is from fetching,
        # not from form type parsing.
        assert "Unsupported" not in result.output


# -----------------------------------------------------------------------
# search _similarity_text helper
# -----------------------------------------------------------------------


class TestSimilarityText:
    """The _similarity_text helper colour-codes similarity percentages."""

    def test_high_similarity_green(self):
        from sec_semantic_search.cli.search import _similarity_text

        text = _similarity_text(0.45)
        assert "45.0%" in str(text)
        assert text.style == "bold green"

    def test_medium_similarity_yellow(self):
        from sec_semantic_search.cli.search import _similarity_text

        text = _similarity_text(0.30)
        assert "30.0%" in str(text)
        assert text.style == "yellow"

    def test_low_similarity_dim(self):
        from sec_semantic_search.cli.search import _similarity_text

        text = _similarity_text(0.10)
        assert "10.0%" in str(text)
        assert text.style == "dim"


# -----------------------------------------------------------------------
# Rich markup in data (security audit 2026-09-16, observation)
# -----------------------------------------------------------------------


class TestDataIsNotRichMarkup:
    """Filing-derived, stored and exception text must print literally.

    Rich parses ``[...]`` in strings handed to ``console.print`` or
    ``Table.add_row``. A section title such as
    ``[link=https://attacker.example]Item 1A[/link]`` rendered as a
    clickable terminal hyperlink, and lowercase ``[tag]`` sequences vanished.
    """

    LINK = "[link=https://x.io]A[/link]"
    BOLD = "[bold]B[/bold]"

    @staticmethod
    def _wide_console():
        from rich.console import Console

        # Wide enough that no table cell wraps through a payload.
        return Console(width=400)

    @pytest.fixture
    def _isolate_registry(self):
        with patch("sec_semantic_search.database.MetadataRegistry") as MockRegistry:
            MockRegistry.return_value = MagicMock()
            yield MockRegistry

    def _search(self, results, query="risk"):
        with (
            patch("sec_semantic_search.cli.search.console", self._wide_console()),
            patch("sec_semantic_search.cli.search.SearchEngine") as MockEngine,
        ):
            MockEngine.return_value.search.return_value = results
            return runner.invoke(app, ["search", query])

    def test_search_section_and_source_print_literally(self, _isolate_registry):
        from sec_semantic_search.core.types import ContentType, SearchResult

        result = self._search(
            [
                SearchResult(
                    content="Body text.",
                    path=self.LINK,
                    content_type=ContentType.TEXT,
                    ticker="AAPL",
                    form_type="[u]10-K[/u]",
                    similarity=0.5,
                )
            ]
        )
        output = _strip_ansi(result.output)
        assert result.exit_code == 0
        assert self.LINK in output
        assert "AAPL [u]10-K[/u]" in output
        assert "\x1b]8;" not in result.output  # no OSC-8 hyperlink

    def test_search_query_echo_prints_literally(self, _isolate_registry):
        from sec_semantic_search.core.types import ContentType, SearchResult

        hit = SearchResult(
            content="x",
            path="Item 1",
            content_type=ContentType.TEXT,
            ticker="AAPL",
            form_type="10-K",
            similarity=0.5,
        )
        result = self._search([hit], query=self.BOLD)
        assert f"for: {self.BOLD}" in _strip_ansi(result.output)

    def test_search_error_text_prints_literally(self, _isolate_registry):
        from sec_semantic_search.core.exceptions import SearchError

        with (
            patch("sec_semantic_search.cli.search.console", self._wide_console()),
            patch("sec_semantic_search.cli.search.SearchEngine") as MockEngine,
        ):
            MockEngine.return_value.search.side_effect = SearchError(
                f"failed {self.LINK}", details=self.BOLD
            )
            result = runner.invoke(app, ["search", "q"])
        output = _strip_ansi(result.output)
        assert f"failed {self.LINK}" in output
        assert self.BOLD in output

    def test_encryption_extra_hint_keeps_its_brackets(self, _isolate_registry):
        """The CLI's own hint lost "[encryption]" to the markup parser."""
        from sec_semantic_search.core.exceptions import DatabaseError

        _isolate_registry.side_effect = DatabaseError("file is not a database")
        with (
            patch("sec_semantic_search.cli.search.console", self._wide_console()),
            patch("sec_semantic_search.cli.search.SearchEngine"),
        ):
            result = runner.invoke(app, ["search", "q"])
        assert "pip install sec-semantic-search[encryption]" in _strip_ansi(result.output)

    def test_manage_list_cells_print_literally(self):
        with (
            patch("sec_semantic_search.cli.manage.console", self._wide_console()),
            patch("sec_semantic_search.cli.manage.MetadataRegistry") as MockReg,
        ):
            MockReg.return_value.list_filings.return_value = [
                make_filing_record(ticker=self.BOLD, accession_number=self.LINK)
            ]
            result = runner.invoke(app, ["manage", "list"])
        output = _strip_ansi(result.output)
        assert self.BOLD in output
        assert self.LINK in output

    def test_manage_remove_not_found_echoes_literally(self):
        with (
            patch("sec_semantic_search.cli.manage.console", self._wide_console()),
            patch("sec_semantic_search.cli.manage.MetadataRegistry") as MockReg,
        ):
            MockReg.return_value.get_filing.return_value = None
            result = runner.invoke(app, ["manage", "remove", self.BOLD])
        assert f"Filing not found: {self.BOLD}" in _strip_ansi(result.output)

    def test_manage_bulk_remove_lists_filings_literally(self):
        with (
            patch("sec_semantic_search.cli.manage.console", self._wide_console()),
            patch("sec_semantic_search.cli.manage.MetadataRegistry") as MockReg,
        ):
            MockReg.return_value.list_filings.return_value = [
                make_filing_record(form_type=self.BOLD)
            ]
            result = runner.invoke(app, ["manage", "remove", "--ticker", "AAPL"], input="n\n")
        assert f"AAPL {self.BOLD}" in _strip_ansi(result.output)

    def test_ingest_failure_lines_print_literally(self):
        from datetime import date

        from rich.console import Console

        from sec_semantic_search.cli.ingest import _CliReporter, _make_progress
        from sec_semantic_search.core import SECSemanticSearchError
        from sec_semantic_search.pipeline.fetch import FilingInfo

        console = Console(width=400, record=True, file=open("/dev/null", "w"))  # noqa: SIM115
        with patch("sec_semantic_search.cli.ingest.console", console):
            progress = _make_progress()
        reporter = _CliReporter(progress)
        filing = FilingInfo(
            ticker="AAPL",
            form_type=self.BOLD,
            filing_date=date(2024, 11, 1),
            accession_number=self.LINK,
            company_name="Apple Inc.",
        )
        reporter.filing_started(0, 1, filing)
        reporter.failed(
            filing, "processing", SECSemanticSearchError(f"bad {self.LINK}", details=self.BOLD)
        )
        reporter.skipped(filing, "duplicate")
        text = console.export_text()
        console.file.close()
        assert f"AAPL {self.BOLD} — bad {self.LINK}" in text
        assert f"    {self.BOLD}" in text
        assert f"({filing.filing_date.isoformat()}, {self.LINK})" in text
        # Progress descriptions are markup too: they must render to the literal.
        from rich.text import Text

        rendered = Text.from_markup(progress.tasks[0].description).plain
        assert rendered == f"Filings: AAPL {self.BOLD}"
