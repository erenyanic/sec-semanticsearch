"""
Tests for the FilingFetcher pipeline component.

FilingFetcher wraps edgartools to fetch SEC filings over the network.
We mock edgartools' Company and set_identity so tests run without
network access. The class has substantial testable logic in its helper
methods: form type validation, date parsing, date filter formatting,
and the FilingInfo dataclass conversion.
"""

import logging
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from sec_semantic_search.core.exceptions import FetchError
from sec_semantic_search.pipeline.fetch import FilingFetcher, FilingInfo

# -----------------------------------------------------------------------
# FilingInfo dataclass
# -----------------------------------------------------------------------


class TestFilingInfo:
    """FilingInfo is a lightweight preview returned by list_available()."""

    def test_to_identifier(self):
        """to_identifier() should convert to a FilingIdentifier."""
        info = FilingInfo(
            ticker="aapl",
            form_type="10-K",
            filing_date=date(2024, 11, 1),
            accession_number="ACC-001",
            company_name="Apple Inc.",
        )
        fid = info.to_identifier()
        assert fid.ticker == "AAPL"  # Normalised
        assert fid.form_type == "10-K"
        assert fid.filing_date == date(2024, 11, 1)
        assert fid.accession_number == "ACC-001"


# -----------------------------------------------------------------------
# Helper method tests (via a fetcher with mocked edgartools)
# -----------------------------------------------------------------------


@pytest.fixture
def fetcher():
    """A FilingFetcher with mocked EDGAR identity configuration."""
    with patch("sec_semantic_search.pipeline.fetch.set_identity"):
        return FilingFetcher()


class TestValidateFormType:
    """_validate_form_type() normalises and validates."""

    def test_valid_10k(self, fetcher):
        assert fetcher._validate_form_type("10-K") == "10-K"

    def test_valid_10q_lowercase(self, fetcher):
        assert fetcher._validate_form_type("10-q") == "10-Q"

    def test_valid_8k(self, fetcher):
        assert fetcher._validate_form_type("8-K") == "8-K"

    def test_valid_8k_lowercase(self, fetcher):
        assert fetcher._validate_form_type("8-k") == "8-K"

    def test_invalid_form_raises(self, fetcher):
        with pytest.raises(FetchError, match="Unsupported form type"):
            fetcher._validate_form_type("20-F")


class TestParseFilingDate:
    """_parse_filing_date() handles both date objects and strings."""

    def test_date_object_passthrough(self, fetcher):
        d = date(2024, 6, 15)
        assert fetcher._parse_filing_date(d) is d

    def test_string_parsing(self, fetcher):
        result = fetcher._parse_filing_date("2024-06-15")
        assert result == date(2024, 6, 15)

    def test_invalid_string_raises(self, fetcher):
        with pytest.raises(ValueError):
            fetcher._parse_filing_date("not-a-date")


class TestFormatDateFilter:
    """_format_date_filter() builds edgartools date range strings."""

    def test_both_dates(self, fetcher):
        result = fetcher._format_date_filter("2020-01-01", "2024-12-31")
        assert result == "2020-01-01:2024-12-31"

    def test_start_only(self, fetcher):
        result = fetcher._format_date_filter("2020-01-01", None)
        assert result == "2020-01-01:"

    def test_end_only(self, fetcher):
        result = fetcher._format_date_filter(None, "2024-12-31")
        assert result == ":2024-12-31"

    def test_neither(self, fetcher):
        result = fetcher._format_date_filter(None, None)
        assert result is None

    def test_date_objects(self, fetcher):
        result = fetcher._format_date_filter(date(2020, 1, 1), date(2024, 12, 31))
        assert result == "2020-01-01:2024-12-31"


class TestFetchLatest:
    """fetch_latest() delegates to fetch_one(index=0)."""

    def test_delegates_to_fetch_one(self, fetcher):
        """fetch_latest should call fetch_one with index=0."""
        mock_filing = MagicMock()
        mock_filing.accession_no = "ACC-001"
        mock_filing.filing_date = date(2024, 11, 1)
        mock_filing.html.return_value = "<html>content</html>"

        mock_company = MagicMock()
        mock_filings = MagicMock()
        mock_filings.__len__ = lambda self: 1
        mock_filings.__iter__ = lambda self: iter([mock_filing])
        mock_filings.__bool__ = lambda self: True
        mock_company.get_filings.return_value = mock_filings

        with patch.object(fetcher, "_get_company", return_value=mock_company):
            filing_id, html = fetcher.fetch_latest("AAPL", "10-K")

        assert filing_id.ticker == "AAPL"
        assert html == "<html>content</html>"


class TestFetchOneIndexOutOfRange:
    """fetch_one() should raise FetchError when index exceeds available filings."""

    def test_index_out_of_range(self, fetcher):
        mock_company = MagicMock()
        mock_filings = MagicMock()
        mock_filings.__len__ = lambda self: 2
        mock_filings.__iter__ = lambda self: iter([MagicMock(), MagicMock()])
        mock_filings.__bool__ = lambda self: True
        mock_company.get_filings.return_value = mock_filings

        with patch.object(fetcher, "_get_company", return_value=mock_company):
            with pytest.raises(FetchError, match="Index 5 out of range"):
                fetcher.fetch_one("AAPL", "10-K", index=5)


class TestGetCompanyError:
    """_get_company() should wrap edgartools errors as FetchError."""

    def test_invalid_ticker(self, fetcher):
        with patch(
            "sec_semantic_search.pipeline.fetch.Company",
            side_effect=Exception("Unknown ticker"),
        ):
            with pytest.raises(FetchError, match="Invalid ticker"):
                fetcher._get_company("INVALIDTICKER")


# -----------------------------------------------------------------------
# fetch() generator — fault tolerance and count=None
# -----------------------------------------------------------------------


def _make_mock_filing(
    accession_no, filing_date, *, html_content="<html></html>", html_side_effect=None, form="10-K"
):
    """Create a mock filing object with the given properties."""
    filing = MagicMock()
    filing.accession_no = accession_no
    filing.filing_date = filing_date
    filing.form = form
    if html_side_effect is not None:
        filing.html.side_effect = html_side_effect
    else:
        filing.html.return_value = html_content
    return filing


def _make_mock_filings(filing_list):
    """Wrap a list of mock filings to behave like edgartools Filings."""
    mock_filings = MagicMock()
    mock_filings.__len__ = lambda self: len(filing_list)
    mock_filings.__iter__ = lambda self: iter(filing_list)
    mock_filings.__bool__ = lambda self: bool(filing_list)
    return mock_filings


class TestFetchGeneratorFaultTolerance:
    """fetch() should skip filings that raise FetchError and continue."""

    def test_skips_failed_filing_continues(self, fetcher):
        """If one filing fails, the others should still be yielded."""
        filings = [
            _make_mock_filing("ACC-001", date(2024, 1, 1), html_content="<html>1</html>"),
            _make_mock_filing(
                "ACC-002", date(2024, 2, 1), html_side_effect=Exception("Network error")
            ),
            _make_mock_filing("ACC-003", date(2024, 3, 1), html_content="<html>3</html>"),
        ]
        mock_company = MagicMock()
        mock_company.get_filings.return_value = _make_mock_filings(filings)

        with patch.object(fetcher, "_get_company", return_value=mock_company):
            results = list(fetcher.fetch("AAPL", "10-K", count=3))

        assert len(results) == 2
        assert results[0][0].accession_number == "ACC-001"
        assert results[1][0].accession_number == "ACC-003"

    def test_all_filings_fail_yields_nothing(self, fetcher):
        """If every filing fails, the generator should yield nothing."""
        filings = [
            _make_mock_filing("ACC-001", date(2024, 1, 1), html_side_effect=Exception("Fail 1")),
            _make_mock_filing("ACC-002", date(2024, 2, 1), html_side_effect=Exception("Fail 2")),
        ]
        mock_company = MagicMock()
        mock_company.get_filings.return_value = _make_mock_filings(filings)

        with patch.object(fetcher, "_get_company", return_value=mock_company):
            results = list(fetcher.fetch("AAPL", "10-K", count=2))

        assert results == []


class TestFetchLimitingLog:
    """fetch() should log when limiting results (regression test for F1 bug)."""

    def test_logs_limiting_message_when_count_less_than_available(self, fetcher, caplog):
        """With 5 available filings and count=2, the limiting log should fire.

        Regression: previously the generator was consumed twice — once by
        ``list(filings)[:count]`` and again by ``len(list(filings))`` — so
        ``total_available`` was always 0 and this log path was dead code.
        """
        filings = [
            _make_mock_filing(
                f"ACC-{i:03d}",
                date(2024, 1, i + 1),
                html_content=f"<html>{i}</html>",
            )
            for i in range(5)
        ]
        mock_company = MagicMock()
        mock_company.get_filings.return_value = _make_mock_filings(filings)

        pkg_logger = logging.getLogger("sec_semantic_search")
        pkg_logger.propagate = True
        try:
            with patch.object(fetcher, "_get_company", return_value=mock_company):
                with caplog.at_level(logging.INFO, logger="sec_semantic_search"):
                    results = list(fetcher.fetch("AAPL", "10-K", count=2))
        finally:
            pkg_logger.propagate = False

        assert len(results) == 2
        assert "Limiting to 2 10-K filings; more are available" in caplog.text

    def test_no_limiting_log_when_count_equals_available(self, fetcher, caplog):
        """When count equals available filings, no limiting log should appear."""
        filings = [
            _make_mock_filing(
                f"ACC-{i:03d}",
                date(2024, 1, i + 1),
                html_content=f"<html>{i}</html>",
            )
            for i in range(3)
        ]
        mock_company = MagicMock()
        mock_company.get_filings.return_value = _make_mock_filings(filings)

        pkg_logger = logging.getLogger("sec_semantic_search")
        pkg_logger.propagate = True
        try:
            with patch.object(fetcher, "_get_company", return_value=mock_company):
                with caplog.at_level(logging.INFO, logger="sec_semantic_search"):
                    results = list(fetcher.fetch("AAPL", "10-K", count=3))
        finally:
            pkg_logger.propagate = False

        assert len(results) == 3
        assert "Limiting to" not in caplog.text


class _CountingFilings:
    """Iterable standing in for edgartools ``Filings``: counts rows pulled.

    edgartools builds a ``Filing`` object per row on iteration, so rows
    pulled is the cost the lazy filter avoids.
    """

    def __init__(self, n: int, *, amendment_every: int = 0):
        self.n = n
        self.amendment_every = amendment_every
        self.pulled = 0

    def __len__(self):  # cheap in edgartools: a table row count
        return self.n

    def __bool__(self):
        return self.n > 0

    def __iter__(self):
        for i in range(self.n):
            self.pulled += 1
            form = "10-K/A" if self.amendment_every and i % self.amendment_every == 1 else "10-K"
            yield _make_mock_filing(
                f"ACC-{i:05d}", date(2024, 1, 1), html_content=f"<html>{i}</html>", form=form
            )


class TestLazyFilingIteration:
    """Only the rows a call needs are pulled from edgartools."""

    @staticmethod
    def _company(filings):
        company = MagicMock()
        company.get_filings.return_value = filings
        return company

    def test_fetch_pulls_count_plus_one(self, fetcher):
        filings = _CountingFilings(3000)
        with patch.object(fetcher, "_get_company", return_value=self._company(filings)):
            results = list(fetcher.fetch("AAPL", "10-K", count=2))
        assert [fid.accession_number for fid, _ in results] == ["ACC-00000", "ACC-00001"]
        assert filings.pulled == 3  # two kept, one to learn that more exist

    def test_fetch_skips_amendments_lazily(self, fetcher):
        filings = _CountingFilings(3000, amendment_every=2)
        with patch.object(fetcher, "_get_company", return_value=self._company(filings)):
            results = list(fetcher.fetch("AAPL", "10-K", count=3))
        assert [fid.accession_number for fid, _ in results] == [
            "ACC-00000",
            "ACC-00002",
            "ACC-00004",
        ]
        assert filings.pulled <= 7

    def test_fetch_one_latest_pulls_one_row(self, fetcher):
        filings = _CountingFilings(3000)
        with patch.object(fetcher, "_get_company", return_value=self._company(filings)):
            filing_id, _ = fetcher.fetch_latest("AAPL", "10-K")
        assert filing_id.accession_number == "ACC-00000"
        assert filings.pulled == 1

    def test_fetch_one_index_counts_only_originals(self, fetcher):
        filings = _CountingFilings(3000, amendment_every=2)
        with patch.object(fetcher, "_get_company", return_value=self._company(filings)):
            filing_id, _ = fetcher.fetch_one("AAPL", "10-K", index=2)
        assert filing_id.accession_number == "ACC-00004"
        assert filings.pulled == 5

    def test_fetch_one_out_of_range_reports_available(self, fetcher):
        filings = _CountingFilings(2)
        with patch.object(fetcher, "_get_company", return_value=self._company(filings)):
            with pytest.raises(FetchError, match="Index 5 out of range") as exc_info:
                fetcher.fetch_one("AAPL", "10-K", index=5)
        assert exc_info.value.details == "Only 2 filings available."

    def test_list_available_pulls_count_rows(self, fetcher):
        filings = _CountingFilings(3000)
        with patch.object(fetcher, "_get_company", return_value=self._company(filings)):
            available = fetcher.list_available("AAPL", "10-K", count=4)
        assert len(available) == 4
        assert filings.pulled == 4
        assert all(info._filing_obj is not None for info in available)


class TestFetchCountNone:
    """When count is None, fetch() should default to max_filings."""

    def test_count_none_defaults_to_max_filings(self, fetcher):
        """With 10 available filings and max_filings=5, only 5 should be yielded."""
        fetcher.max_filings = 5
        filings = [
            _make_mock_filing(
                f"ACC-{i:03d}", date(2024, 1, i + 1), html_content=f"<html>{i}</html>"
            )
            for i in range(10)
        ]
        mock_company = MagicMock()
        mock_company.get_filings.return_value = _make_mock_filings(filings)

        with patch.object(fetcher, "_get_company", return_value=mock_company):
            results = list(fetcher.fetch("AAPL", "10-K", count=None))

        assert len(results) == 5


# -----------------------------------------------------------------------
# BF-002: Amendment filtering (10-K/A, 10-Q/A silently skipped)
# -----------------------------------------------------------------------


class TestIsAmendment:
    """_is_amendment() detects filings whose actual form ends with /A."""

    def test_original_10k(self, fetcher):
        filing = _make_mock_filing("ACC-001", date(2024, 1, 1), form="10-K")
        assert fetcher._is_amendment(filing) is False

    def test_amendment_10ka(self, fetcher):
        filing = _make_mock_filing("ACC-002", date(2024, 1, 1), form="10-K/A")
        assert fetcher._is_amendment(filing) is True

    def test_amendment_10qa(self, fetcher):
        filing = _make_mock_filing("ACC-003", date(2024, 1, 1), form="10-Q/A")
        assert fetcher._is_amendment(filing) is True

    def test_original_8k(self, fetcher):
        filing = _make_mock_filing("ACC-004", date(2024, 1, 1), form="8-K")
        assert fetcher._is_amendment(filing) is False

    def test_amendment_8ka(self, fetcher):
        filing = _make_mock_filing("ACC-005", date(2024, 1, 1), form="8-K/A")
        assert fetcher._is_amendment(filing) is True

    def test_no_form_attribute(self, fetcher):
        """Filings without a form attribute should not be treated as amendments."""
        filing = MagicMock(spec=[])
        assert fetcher._is_amendment(filing) is False


class TestAmendmentFilteringInListAvailable:
    """list_available() should skip amendments and only return originals."""

    def test_amendments_excluded(self, fetcher):
        """10-K/A filings should be filtered out of list_available() results."""
        filings = [
            _make_mock_filing("ACC-001", date(2024, 11, 5), form="10-K/A"),
            _make_mock_filing("ACC-002", date(2024, 11, 1), form="10-K"),
        ]
        mock_company = MagicMock()
        mock_company.get_filings.return_value = _make_mock_filings(filings)

        with patch.object(fetcher, "_get_company", return_value=mock_company):
            result = fetcher.list_available("AAPL", "10-K", count=10)

        assert len(result) == 1
        assert result[0].accession_number == "ACC-002"

    def test_count_respects_non_amendment_filings_only(self, fetcher):
        """count should apply after amendment filtering, not before."""
        filings = [
            _make_mock_filing("ACC-001", date(2024, 3, 1), form="10-K/A"),
            _make_mock_filing("ACC-002", date(2024, 2, 1), form="10-K"),
            _make_mock_filing("ACC-003", date(2024, 1, 1), form="10-K"),
        ]
        mock_company = MagicMock()
        mock_company.get_filings.return_value = _make_mock_filings(filings)

        with patch.object(fetcher, "_get_company", return_value=mock_company):
            result = fetcher.list_available("AAPL", "10-K", count=1)

        # Should get only 1 original, not the amendment
        assert len(result) == 1
        assert result[0].accession_number == "ACC-002"

    def test_amendment_skip_logged(self, fetcher, caplog):
        """Skipped amendments should produce a debug log message."""
        filings = [
            _make_mock_filing("ACC-001", date(2024, 1, 1), form="10-K/A"),
            _make_mock_filing("ACC-002", date(2024, 1, 1), form="10-K"),
        ]
        mock_company = MagicMock()
        mock_company.get_filings.return_value = _make_mock_filings(filings)

        pkg_logger = logging.getLogger("sec_semantic_search")
        pkg_logger.propagate = True
        try:
            with patch.object(fetcher, "_get_company", return_value=mock_company):
                with caplog.at_level(logging.DEBUG, logger="sec_semantic_search"):
                    fetcher.list_available("AAPL", "10-K", count=10)
        finally:
            pkg_logger.propagate = False

        assert "Skipping amendment ACC-001 (10-K/A)" in caplog.text


class TestAmendmentFilteringInFetch:
    """fetch() generator should skip amendments."""

    def test_amendments_excluded_from_fetch(self, fetcher):
        """Amendments should not be yielded by fetch()."""
        filings = [
            _make_mock_filing(
                "ACC-001", date(2024, 2, 1), form="10-K/A", html_content="<html>amendment</html>"
            ),
            _make_mock_filing(
                "ACC-002", date(2024, 1, 1), form="10-K", html_content="<html>original</html>"
            ),
        ]
        mock_company = MagicMock()
        mock_company.get_filings.return_value = _make_mock_filings(filings)

        with patch.object(fetcher, "_get_company", return_value=mock_company):
            results = list(fetcher.fetch("AAPL", "10-K", count=10))

        assert len(results) == 1
        assert results[0][0].accession_number == "ACC-002"
        assert results[0][1] == "<html>original</html>"


class TestAmendmentFilteringInFetchOne:
    """fetch_one() should skip amendments when indexing."""

    def test_index_counts_non_amendments_only(self, fetcher):
        """Index should refer to the position among non-amendment filings."""
        filings = [
            _make_mock_filing(
                "ACC-001", date(2024, 3, 1), form="10-K/A", html_content="<html>amendment</html>"
            ),
            _make_mock_filing(
                "ACC-002", date(2024, 2, 1), form="10-K", html_content="<html>first</html>"
            ),
            _make_mock_filing(
                "ACC-003", date(2024, 1, 1), form="10-K", html_content="<html>second</html>"
            ),
        ]
        mock_company = MagicMock()
        mock_company.get_filings.return_value = _make_mock_filings(filings)

        with patch.object(fetcher, "_get_company", return_value=mock_company):
            filing_id, html = fetcher.fetch_one("AAPL", "10-K", index=0)

        # index=0 should return ACC-002 (first non-amendment), not ACC-001
        assert filing_id.accession_number == "ACC-002"
        assert html == "<html>first</html>"


class TestAmendmentFilteringInFetchByAccession:
    """fetch_by_accession() should refuse to fetch amendment filings."""

    def test_amendment_accession_not_found(self, fetcher):
        """Requesting an amendment's accession number should raise FetchError."""
        filings = [
            _make_mock_filing(
                "ACC-001", date(2024, 1, 1), form="10-K/A", html_content="<html>amendment</html>"
            ),
            _make_mock_filing(
                "ACC-002", date(2024, 1, 1), form="10-K", html_content="<html>original</html>"
            ),
        ]
        mock_company = MagicMock()
        mock_company.get_filings.return_value = _make_mock_filings(filings)

        with patch.object(fetcher, "_get_company", return_value=mock_company):
            with pytest.raises(FetchError, match="Filing not found"):
                fetcher.fetch_by_accession("AAPL", "10-K", "ACC-001")


# -----------------------------------------------------------------------
# BF-011: _should_skip() and amendment form type support
# -----------------------------------------------------------------------


class TestShouldSkip:
    """_should_skip() conditionally filters amendments based on requested form."""

    def test_base_form_skips_amendment(self, fetcher):
        """When requesting 10-K, 10-K/A filings should be skipped."""
        filing = _make_mock_filing("ACC-001", date(2024, 1, 1), form="10-K/A")
        assert fetcher._should_skip(filing, "10-K") is True

    def test_base_form_keeps_original(self, fetcher):
        """When requesting 10-K, 10-K filings should not be skipped."""
        filing = _make_mock_filing("ACC-001", date(2024, 1, 1), form="10-K")
        assert fetcher._should_skip(filing, "10-K") is False

    def test_amendment_form_keeps_amendment(self, fetcher):
        """When requesting 10-K/A, 10-K/A filings should not be skipped."""
        filing = _make_mock_filing("ACC-001", date(2024, 1, 1), form="10-K/A")
        assert fetcher._should_skip(filing, "10-K/A") is False

    def test_8k_base_skips_8ka(self, fetcher):
        filing = _make_mock_filing("ACC-001", date(2024, 1, 1), form="8-K/A")
        assert fetcher._should_skip(filing, "8-K") is True

    def test_8ka_form_keeps_8ka(self, fetcher):
        filing = _make_mock_filing("ACC-001", date(2024, 1, 1), form="8-K/A")
        assert fetcher._should_skip(filing, "8-K/A") is False

    def test_10q_base_skips_10qa(self, fetcher):
        filing = _make_mock_filing("ACC-001", date(2024, 1, 1), form="10-Q/A")
        assert fetcher._should_skip(filing, "10-Q") is True

    def test_10qa_form_keeps_10qa(self, fetcher):
        filing = _make_mock_filing("ACC-001", date(2024, 1, 1), form="10-Q/A")
        assert fetcher._should_skip(filing, "10-Q/A") is False

    def test_no_form_attribute_not_skipped(self, fetcher):
        filing = MagicMock(spec=[])
        assert fetcher._should_skip(filing, "10-K") is False


class TestAmendmentFormValidation:
    """Amendment form types (10-K/A, 10-Q/A, 8-K/A) are valid in SUPPORTED_FORMS."""

    def test_validate_10ka(self, fetcher):
        assert fetcher._validate_form_type("10-K/A") == "10-K/A"

    def test_validate_10qa(self, fetcher):
        assert fetcher._validate_form_type("10-Q/A") == "10-Q/A"

    def test_validate_8ka(self, fetcher):
        assert fetcher._validate_form_type("8-K/A") == "8-K/A"

    def test_validate_case_insensitive(self, fetcher):
        assert fetcher._validate_form_type("10-k/a") == "10-K/A"


class TestAmendmentListAvailable:
    """list_available() should include amendments when the amendment form is requested."""

    def test_amendments_included_when_requested(self, fetcher):
        """When requesting 10-K/A, amendment filings should be returned."""
        filings = [
            _make_mock_filing("ACC-001", date(2024, 11, 5), form="10-K/A"),
            _make_mock_filing("ACC-002", date(2024, 11, 1), form="10-K/A"),
        ]
        mock_company = MagicMock()
        mock_company.get_filings.return_value = _make_mock_filings(filings)

        with patch.object(fetcher, "_get_company", return_value=mock_company):
            result = fetcher.list_available("AAPL", "10-K/A", count=10)

        assert len(result) == 2
        assert result[0].accession_number == "ACC-001"
        assert result[1].accession_number == "ACC-002"
