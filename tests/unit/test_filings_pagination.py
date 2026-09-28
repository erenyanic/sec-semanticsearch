"""
Tests for ``MetadataRegistry.list_filings_page()`` (F-06).

Covers:
    - Sorting and slicing in SQL, with ``total`` counting every match
    - Deterministic pages when the sort column has ties
    - Filters applied to both the page and the total
    - Only whitelisted columns and directions reach the ``ORDER BY``
"""

from datetime import date

import pytest

from sec_semantic_search.core.types import FilingIdentifier
from sec_semantic_search.database.metadata import MetadataRegistry


@pytest.fixture
def registry(tmp_db_path) -> MetadataRegistry:
    reg = MetadataRegistry(db_path=tmp_db_path)
    rows = [
        ("AAPL", "10-K", date(2024, 11, 1), 30),
        ("MSFT", "10-Q", date(2024, 4, 25), 10),
        ("AAPL", "10-Q", date(2024, 8, 2), 20),
        ("NVDA", "8-K", date(2024, 8, 2), 5),
        ("MSFT", "10-K", date(2024, 7, 30), 40),
    ]
    for i, (ticker, form, filed, chunks) in enumerate(rows):
        fid = FilingIdentifier(
            ticker=ticker,
            form_type=form,
            filing_date=filed,
            accession_number=f"0000000000-24-{i:06d}",
        )
        reg.register_filing(fid, chunk_count=chunks)
    yield reg
    reg.close()


def _tickers(records):
    return [r.ticker for r in records]


class TestPaging:
    def test_default_sort_is_filing_date_desc(self, registry):
        records, total = registry.list_filings_page()
        assert total == 5
        assert [r.filing_date for r in records] == [
            "2024-11-01",
            "2024-08-02",
            "2024-08-02",
            "2024-07-30",
            "2024-04-25",
        ]

    def test_limit_and_offset_slice_in_sql(self, registry):
        page, total = registry.list_filings_page(
            sort_by="chunk_count", order="asc", limit=2, offset=2
        )
        assert total == 5
        assert [r.chunk_count for r in page] == [20, 30]

    def test_offset_past_the_end_returns_empty_page_with_total(self, registry):
        page, total = registry.list_filings_page(limit=10, offset=50)
        assert page == []
        assert total == 5

    def test_ties_are_broken_by_id_so_pages_never_overlap(self, registry):
        seen = []
        for offset in range(5):
            page, _ = registry.list_filings_page(sort_by="ticker", limit=1, offset=offset)
            seen.extend(r.accession_number for r in page)
        assert len(seen) == len(set(seen)) == 5

    def test_every_sortable_column_sorts(self, registry):
        for column in MetadataRegistry.SORTABLE_COLUMNS:
            asc, _ = registry.list_filings_page(sort_by=column, order="asc", limit=5)
            desc, _ = registry.list_filings_page(sort_by=column, order="desc", limit=5)
            values = [getattr(r, column) for r in asc]
            assert values == sorted(values)
            assert [getattr(r, column) for r in desc] == sorted(values, reverse=True)


class TestFilters:
    def test_filters_apply_to_page_and_total(self, registry):
        page, total = registry.list_filings_page(ticker="msft", limit=1)
        assert total == 2
        assert _tickers(page) == ["MSFT"]

    def test_ticker_and_form_combined(self, registry):
        page, total = registry.list_filings_page(ticker="AAPL", form_type="10-q")
        assert total == 1
        assert page[0].form_type == "10-Q"


class TestValidation:
    @pytest.mark.parametrize(
        "sort_by",
        [
            "id",
            "accession_number",
            "filing_date; DROP TABLE filings",
            "filing_date DESC, (SELECT 1)",
        ],
    )
    def test_rejects_unlisted_sort_column(self, registry, sort_by):
        with pytest.raises(ValueError, match="Cannot sort"):
            registry.list_filings_page(sort_by=sort_by)

    @pytest.mark.parametrize("order", ["ASC", "up", "desc; --", ""])
    def test_rejects_unlisted_order(self, registry, order):
        with pytest.raises(ValueError, match="order"):
            registry.list_filings_page(order=order)

    @pytest.mark.parametrize(("limit", "offset"), [(0, 0), (-1, 0), (10, -1)])
    def test_rejects_bad_bounds(self, registry, limit, offset):
        with pytest.raises(ValueError):
            registry.list_filings_page(limit=limit, offset=offset)

    def test_filter_values_are_bound_not_interpolated(self, registry):
        page, total = registry.list_filings_page(ticker="AAPL' OR '1'='1")
        assert (page, total) == ([], 0)
