"""
Integration tests for the CLI commands via Typer's CliRunner.

CliRunner invokes commands programmatically without spawning subprocesses.
We mock heavy dependencies (fetcher, databases, embedder) to keep tests
fast while verifying exit codes, output messages, and command routing.
"""

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
