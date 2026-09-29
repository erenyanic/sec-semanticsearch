"""
SEC filing fetcher using edgartools.

This module wraps the edgartools library to fetch SEC filings (8-K, 10-K, 10-Q)
from the EDGAR database. It provides flexible selection methods including:
    - Single latest filing
    - Specific filing by index position
    - Multiple recent filings
    - Filter by year or year range
    - Filter by date range

Usage:
    from sec_semantic_search.pipeline import FilingFetcher

    fetcher = FilingFetcher()

    # Single latest filing
    filing_id, html = fetcher.fetch_latest("AAPL", "10-K")

    # Multiple filings (last 5 years)
    for filing_id, html in fetcher.fetch("AAPL", "10-K", count=5):
        process(filing_id, html)

    # Specific filing by index (0=most recent, 1=second most recent)
    filing_id, html = fetcher.fetch_one("AAPL", "10-K", index=2)

    # Filter by year range
    for filing_id, html in fetcher.fetch("AAPL", "10-Q", year=range(2020, 2025)):
        process(filing_id, html)

    # Filter by date range
    for filing_id, html in fetcher.fetch("AAPL", "10-K", start_date="2020-01-01"):
        process(filing_id, html)
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import date, datetime
from itertools import islice
from typing import Any

from edgar import Company, set_identity

from sec_semantic_search.config import BASE_FORMS, SUPPORTED_FORMS, get_settings
from sec_semantic_search.core import FetchError, FilingIdentifier, get_logger

logger = get_logger(__name__)


@dataclass
class FilingInfo:
    """
    Summary information about an available filing (without content).

    This lightweight class is used by list_available() to preview
    filings before downloading their full HTML content.

    Attributes:
        ticker: Stock ticker symbol
        form_type: SEC form type (10-K, 10-Q)
        filing_date: Date filed with SEC
        accession_number: SEC-assigned unique identifier
        company_name: Full company name
        _filing_obj: Cached edgartools Filing object for direct content
            fetch (avoids redundant EDGAR API round-trips).  Not part
            of the public API — callers should use
            ``FilingFetcher.fetch_filing_content()`` instead.
    """

    ticker: str
    form_type: str
    filing_date: date
    accession_number: str
    company_name: str
    _filing_obj: Any = field(default=None, repr=False, compare=False)

    def to_identifier(self) -> FilingIdentifier:
        """Convert to FilingIdentifier for pipeline use."""
        return FilingIdentifier(
            ticker=self.ticker,
            form_type=self.form_type,
            filing_date=self.filing_date,
            accession_number=self.accession_number,
        )


class FilingFetcher:
    """
    Fetches SEC filings from EDGAR using edgartools.

    This class provides flexible methods for fetching SEC filings with
    various selection criteria. It handles identity configuration
    automatically using credentials from settings.

    Selection Methods:
        - fetch_latest(): Single most recent filing
        - fetch_one(): Single filing by index position
        - fetch(): Multiple filings with optional filters
        - list_available(): Preview filings without downloading

    Filter Options:
        - count: Maximum number of filings to fetch
        - index: Specific position (0=most recent)
        - year: Single year, list of years, or range
        - start_date/end_date: Date range filtering

    Attributes:
        settings: Application settings instance
        max_filings: Maximum filings limit from settings

    Example:
        >>> fetcher = FilingFetcher()
        >>> # Get last 5 years of 10-K filings
        >>> for filing_id, html in fetcher.fetch("AAPL", "10-K", count=5):
        ...     print(f"Processing {filing_id.date_str}")
    """

    def __init__(self) -> None:
        """Initialise the fetcher and configure EDGAR identity (if available)."""
        self.settings = get_settings()
        self.max_filings = self.settings.database.max_filings

        self._configure_identity()

    def apply_identity(self, name: str | None = None, email: str | None = None) -> None:
        """Apply the effective EDGAR identity for the current operation.

        ``edgar.set_identity()`` mutates process-global state, so callers must
        re-apply the intended identity before every EDGAR-bound operation.
        When per-session credentials are provided, they take precedence.
        Otherwise, the server-side defaults from settings are restored.
        """
        if name and email:
            self.set_identity(name, email)
            return

        self._configure_identity()

    def _configure_identity(self) -> None:
        """Configure SEC EDGAR identity from settings (if both fields are set).

        In web deployments (Scenarios B/C), server-side credentials may be
        unset — each user provides their own via ``set_identity()`` per
        request.  The fetcher still works; the caller is responsible for
        calling ``set_identity()`` before any EDGAR requests.
        """
        name = self.settings.edgar.identity_name
        email = self.settings.edgar.identity_email
        if name and email:
            try:
                set_identity(f"{name} {email}")
                logger.debug("EDGAR identity configured from server-side env vars")
            except Exception as e:
                raise FetchError(
                    "Failed to configure EDGAR identity",
                    details=str(e),
                ) from e
        else:
            logger.debug("EDGAR identity not configured — per-session credentials required")

    def set_identity(self, name: str, email: str) -> None:
        """Set EDGAR identity for the current request (per-session credentials).

        Called by the API layer when users supply credentials via HTTP
        headers (``X-Edgar-Name`` / ``X-Edgar-Email``).

        **Privacy:** EDGAR credentials are never logged — not even at
        DEBUG level.  They are personal identity data that must not
        appear in any log output, database, or file.
        """
        set_identity(f"{name} {email}")
        logger.debug("EDGAR identity set via per-session credentials")

    @staticmethod
    def _is_amendment(filing) -> bool:
        """Check whether a filing is an amendment (e.g. 10-K/A, 10-Q/A).

        edgartools returns amendments alongside their parent form when
        queried by a base form type (e.g. ``form='10-K'`` also returns
        ``10-K/A``).  We read the filing object's actual ``form``
        attribute and check for the ``/A`` suffix.

        Returns:
            True if the filing's actual form type ends with ``/A``.
        """
        actual_form = getattr(filing, "form", "")
        return isinstance(actual_form, str) and actual_form.endswith("/A")

    @staticmethod
    def _should_skip(filing, requested_form: str) -> bool:
        """Decide whether a filing should be skipped for the requested form.

        When the user requests a **base** form (e.g. ``10-K``), edgartools
        returns both originals and amendments.  We skip amendments so they
        don't displace the original via the UNIQUE constraint.

        When the user requests an **amendment** form (e.g. ``10-K/A``),
        edgartools returns only amendments, so nothing is skipped.

        Returns:
            True if the filing should be excluded from results.
        """
        if requested_form in BASE_FORMS:
            actual_form = getattr(filing, "form", "")
            return isinstance(actual_form, str) and actual_form.endswith("/A")
        return False

    def _eligible(self, filings: Iterable[Any], requested_form: str) -> Iterator[Any]:
        """Yield the filings that ``_should_skip`` keeps, lazily.

        edgartools builds a ``Filing`` object per row as it is iterated;
        callers slice this generator so a company with thousands of 8-Ks
        does not build them all to keep a handful.
        """
        for filing in filings:
            if self._should_skip(filing, requested_form):
                logger.debug(
                    "Skipping amendment %s (%s) — original filing preferred",
                    filing.accession_no,
                    getattr(filing, "form", "unknown"),
                )
                continue
            yield filing

    def _validate_form_type(self, form_type: str) -> str:
        """
        Validate and normalise form type.

        Args:
            form_type: SEC form type (e.g., "10-K", "10-Q")

        Returns:
            Normalised form type in uppercase.

        Raises:
            FetchError: If form type is not supported.
        """
        normalised = form_type.upper()
        if normalised not in SUPPORTED_FORMS:
            raise FetchError(
                f"Unsupported form type: {form_type}",
                details=f"Supported forms: {', '.join(SUPPORTED_FORMS)}",
            )
        return normalised

    def _parse_filing_date(self, date_value: str | date) -> date:
        """
        Parse filing date from edgartools.

        edgartools may return dates as strings or date objects depending
        on the version. This method handles both cases.

        Args:
            date_value: Filing date as string (YYYY-MM-DD) or date object.

        Returns:
            Python date object.
        """
        if isinstance(date_value, date):
            return date_value
        return datetime.strptime(date_value, "%Y-%m-%d").date()

    def _format_date_filter(
        self,
        start_date: str | date | None,
        end_date: str | date | None,
    ) -> str | None:
        """
        Format date range for edgartools filing_date parameter.

        edgartools accepts date ranges in format: "YYYY-MM-DD:YYYY-MM-DD"
        For open-ended ranges: "YYYY-MM-DD:" or ":YYYY-MM-DD"

        Args:
            start_date: Range start (inclusive), or None for open start
            end_date: Range end (inclusive), or None for open end

        Returns:
            Formatted date range string, or None if no dates specified.
        """
        if start_date is None and end_date is None:
            return None

        start_str = ""
        end_str = ""

        if start_date is not None:
            if isinstance(start_date, date):
                start_str = start_date.isoformat()
            else:
                start_str = start_date

        if end_date is not None:
            if isinstance(end_date, date):
                end_str = end_date.isoformat()
            else:
                end_str = end_date

        return f"{start_str}:{end_str}"

    def _get_company(self, ticker: str) -> Company:
        """
        Get Company object for ticker with error handling.

        Args:
            ticker: Stock ticker symbol

        Returns:
            edgartools Company object

        Raises:
            FetchError: If ticker is invalid
        """
        try:
            return Company(ticker.upper())
        except Exception as e:
            raise FetchError(
                f"Invalid ticker symbol: {ticker}",
                details=str(e),
            ) from e

    def _get_filings(
        self,
        company: Company,
        form_type: str,
        year: int | list[int] | range | None = None,
        start_date: str | date | None = None,
        end_date: str | date | None = None,
    ):
        """
        Get filings from company with optional filters.

        Args:
            company: edgartools Company object
            form_type: SEC form type
            year: Year filter (single, list, or range)
            start_date: Date range start
            end_date: Date range end

        Returns:
            edgartools Filings object

        Raises:
            FetchError: If no filings found
        """
        # Build filter arguments
        kwargs: dict = {"form": form_type}

        # Add year filter if specified
        if year is not None:
            if isinstance(year, range):
                kwargs["year"] = list(year)
            else:
                kwargs["year"] = year

        # Add date range filter if specified
        date_filter = self._format_date_filter(start_date, end_date)
        if date_filter is not None:
            kwargs["filing_date"] = date_filter

        logger.debug("Fetching filings with filters: %s", kwargs)

        try:
            filings = company.get_filings(**kwargs)

            if not filings or len(filings) == 0:
                filter_desc = []
                if year:
                    filter_desc.append(f"year={year}")
                if date_filter:
                    filter_desc.append(f"date={date_filter}")
                filter_str = ", ".join(filter_desc) if filter_desc else "no filters"

                raise FetchError(
                    f"No {form_type} filings found ({filter_str})",
                    details="Try adjusting your filter criteria.",
                )

            return filings

        except FetchError:
            raise
        except Exception as e:
            raise FetchError(
                "Failed to retrieve filings",
                details=str(e),
            ) from e

    def _fetch_filing_content(
        self,
        filing,
        ticker: str,
        form_type: str,
    ) -> tuple[FilingIdentifier, str]:
        """
        Fetch HTML content for a single filing.

        Args:
            filing: edgartools Filing object
            ticker: Stock ticker symbol
            form_type: SEC form type

        Returns:
            Tuple of (FilingIdentifier, html_content)

        Raises:
            FetchError: If content fetch fails
        """
        try:
            html_content = filing.html()

            if not html_content:
                raise FetchError(
                    "Empty HTML content received",
                    details=f"Filing {filing.accession_no} returned no content.",
                )

            filing_id = FilingIdentifier(
                ticker=ticker.upper(),
                form_type=form_type,
                filing_date=self._parse_filing_date(filing.filing_date),
                accession_number=filing.accession_no,
            )

            return filing_id, html_content

        except FetchError:
            raise
        except Exception as e:
            raise FetchError(
                f"Failed to fetch content for {filing.accession_no}",
                details=str(e),
            ) from e

    # =========================================================================
    # Public Methods (Content Fetch)
    # =========================================================================

    def fetch_filing_content(
        self,
        filing_info: FilingInfo,
    ) -> tuple[FilingIdentifier, str]:
        """
        Fetch HTML content for a filing using its cached edgartools object.

        When ``FilingInfo`` was created by ``list_available()``, the
        original edgartools ``Filing`` object is stored on
        ``_filing_obj``.  This method uses it directly to fetch HTML,
        avoiding the redundant EDGAR API round-trip that
        ``fetch_by_accession()`` would perform (which re-fetches ALL
        filings for the ticker and linear-scans for the accession
        number).

        Falls back to ``fetch_by_accession()`` when ``_filing_obj`` is
        ``None`` (e.g. when ``FilingInfo`` was constructed manually in
        tests).

        Args:
            filing_info: Filing metadata with optional cached filing object.

        Returns:
            Tuple of (FilingIdentifier, html_content).

        Raises:
            FetchError: If content fetch fails.
        """
        if filing_info._filing_obj is not None:
            logger.info(
                "Fetching %s %s content directly (accession: %s)",
                filing_info.ticker,
                filing_info.form_type,
                filing_info.accession_number,
            )
            filing_id, html_content = self._fetch_filing_content(
                filing_info._filing_obj,
                filing_info.ticker,
                filing_info.form_type,
            )
            logger.info(
                "Fetched %s %s (%s): %s characters",
                filing_info.ticker,
                filing_info.form_type,
                filing_id.date_str,
                f"{len(html_content):,}",
            )
            return filing_id, html_content

        # Fallback: no cached object — use the original linear scan.
        return self.fetch_by_accession(
            filing_info.ticker,
            filing_info.form_type,
            filing_info.accession_number,
        )

    # =========================================================================
    # Public Methods (Listing)
    # =========================================================================

    def list_available(
        self,
        ticker: str,
        form_type: str = "10-K",
        *,
        count: int | None = None,
        year: int | list[int] | range | None = None,
        start_date: str | date | None = None,
        end_date: str | date | None = None,
    ) -> list[FilingInfo]:
        """
        List available filings without downloading content.

        Use this method to preview what filings are available before
        downloading. This is useful for showing users what will be
        fetched or for selecting specific filings.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")
            form_type: SEC form type ("8-K", "10-K", or "10-Q")
            count: Maximum number of filings to list (default: max_filings)
            year: Filter by year (single int, list, or range)
            start_date: Filter by date range start (YYYY-MM-DD or date)
            end_date: Filter by date range end (YYYY-MM-DD or date)

        Returns:
            List of FilingInfo objects with filing metadata

        Raises:
            FetchError: If ticker is invalid or no filings found

        Example:
            >>> available = fetcher.list_available("AAPL", "10-K", count=10)
            >>> for info in available:
            ...     print(f"{info.filing_date}: {info.accession_number}")
        """
        form_type = self._validate_form_type(form_type)
        ticker = ticker.upper()

        # Default to max_filings if count not specified
        if count is None:
            count = self.max_filings

        company = self._get_company(ticker)
        filings = self._get_filings(
            company, form_type, year=year, start_date=start_date, end_date=end_date
        )

        # Limit results — islice stops iteration after count items,
        # avoiding materialising the entire filing list from EDGAR.
        # When a base form is requested, amendments are filtered out so
        # they do not displace the original via the UNIQUE constraint.
        result = [
            FilingInfo(
                ticker=ticker,
                form_type=form_type,
                filing_date=self._parse_filing_date(filing.filing_date),
                accession_number=filing.accession_no,
                company_name=getattr(filing, "company", ticker),
                _filing_obj=filing,
            )
            for filing in islice(self._eligible(filings, form_type), count)
        ]

        logger.info(
            "Listed %d available %s filings for %s",
            len(result),
            form_type,
            ticker,
        )

        return result

    def list_available_across_forms(
        self,
        ticker: str,
        form_types: tuple[str, ...],
        *,
        count: int,
        year: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[FilingInfo]:
        """
        List available filings across multiple form types, sorted by date.

        Calls ``list_available()`` per form type, merges all results, sorts
        by ``filing_date`` descending, and returns the top *count* entries.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL").
            form_types: Form types to search across (e.g., ("10-K", "10-Q")).
            count: Maximum number of filings to return (newest first).
            year: Optional filing-year filter.
            start_date: Optional start-date filter (YYYY-MM-DD).
            end_date: Optional end-date filter (YYYY-MM-DD).

        Returns:
            List of ``FilingInfo`` objects, sorted by filing_date descending,
            truncated to *count*.
        """
        all_available: list[FilingInfo] = []
        for form_type in form_types:
            try:
                available = self.list_available(
                    ticker,
                    form_type,
                    count=count,
                    year=year,
                    start_date=start_date,
                    end_date=end_date,
                )
                all_available.extend(available)
            except FetchError:
                continue
        all_available.sort(key=lambda fi: fi.filing_date, reverse=True)
        return all_available[:count]

    def fetch_latest(
        self,
        ticker: str,
        form_type: str = "10-K",
    ) -> tuple[FilingIdentifier, str]:
        """
        Fetch the most recent filing for a company.

        This is a convenience method equivalent to fetch_one(ticker, form, index=0).

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL", "MSFT")
            form_type: SEC form type ("8-K", "10-K", or "10-Q")

        Returns:
            Tuple of (FilingIdentifier, html_content)

        Raises:
            FetchError: If ticker is invalid, no filings found, or fetch fails

        Example:
            >>> filing_id, html = fetcher.fetch_latest("NVDA", "10-Q")
            >>> print(f"Fetched: {filing_id.date_str}, {len(html):,} chars")
        """
        return self.fetch_one(ticker, form_type, index=0)

    def fetch_one(
        self,
        ticker: str,
        form_type: str = "10-K",
        *,
        index: int = 0,
        year: int | list[int] | range | None = None,
        start_date: str | date | None = None,
        end_date: str | date | None = None,
    ) -> tuple[FilingIdentifier, str]:
        """
        Fetch a single filing by index position.

        Index 0 is the most recent filing, index 1 is the second most
        recent, and so on. Filters are applied before indexing.

        Args:
            ticker: Stock ticker symbol
            form_type: SEC form type ("8-K", "10-K", or "10-Q")
            index: Position in filtered results (0=most recent)
            year: Filter by year before selecting index
            start_date: Filter by date range start
            end_date: Filter by date range end

        Returns:
            Tuple of (FilingIdentifier, html_content)

        Raises:
            FetchError: If index out of range or fetch fails

        Example:
            >>> # Get the third most recent 10-K
            >>> filing_id, html = fetcher.fetch_one("AAPL", "10-K", index=2)

            >>> # Get the most recent 10-K from 2023
            >>> filing_id, html = fetcher.fetch_one("AAPL", "10-K", year=2023)
        """
        form_type = self._validate_form_type(form_type)
        ticker = ticker.upper()

        logger.info(
            "Fetching %s %s at index %d",
            ticker,
            form_type,
            index,
        )

        company = self._get_company(ticker)
        filings = self._get_filings(
            company, form_type, year=year, start_date=start_date, end_date=end_date
        )

        # When a base form is requested, filter out amendments before
        # indexing. Stop at the requested position rather than building a
        # ``Filing`` for every row the company ever filed.
        filing = None
        available = 0
        for available, candidate in enumerate(self._eligible(filings, form_type), start=1):
            if available > index:
                filing = candidate
                break

        if filing is None:
            raise FetchError(
                f"Index {index} out of range",
                details=f"Only {available} filings available.",
            )
        filing_id, html_content = self._fetch_filing_content(filing, ticker, form_type)

        logger.info(
            "Fetched %s %s (%s): %s characters",
            ticker,
            form_type,
            filing_id.date_str,
            f"{len(html_content):,}",
        )

        return filing_id, html_content

    def fetch(
        self,
        ticker: str,
        form_type: str = "10-K",
        *,
        count: int | None = None,
        year: int | list[int] | range | None = None,
        start_date: str | date | None = None,
        end_date: str | date | None = None,
    ) -> Iterator[tuple[FilingIdentifier, str]]:
        """
        Fetch multiple filings with flexible filtering.

        This is the main method for batch fetching. It returns a generator
        that yields filings one at a time, allowing incremental processing.

        Args:
            ticker: Stock ticker symbol
            form_type: SEC form type ("8-K", "10-K", or "10-Q")
            count: Maximum number of filings (default: max_filings setting)
            year: Filter by year - accepts:
                  - Single int: year=2023
                  - List: year=[2022, 2023, 2024]
                  - Range: year=range(2020, 2025)
            start_date: Date range start (YYYY-MM-DD string or date object)
            end_date: Date range end (YYYY-MM-DD string or date object)

        Yields:
            Tuples of (FilingIdentifier, html_content)

        Raises:
            FetchError: If ticker is invalid or no filings match filters

        Examples:
            >>> # Last 5 filings
            >>> for fid, html in fetcher.fetch("AAPL", "10-K", count=5):
            ...     print(f"Processing {fid.date_str}")

            >>> # All 10-Ks from 2020-2024
            >>> for fid, html in fetcher.fetch("AAPL", "10-K", year=range(2020, 2025)):
            ...     print(f"Processing {fid.date_str}")

            >>> # Filings since 2022
            >>> for fid, html in fetcher.fetch("MSFT", "10-Q", start_date="2022-01-01"):
            ...     print(f"Processing {fid.date_str}")
        """
        form_type = self._validate_form_type(form_type)
        ticker = ticker.upper()

        # Default to max_filings if count not specified
        if count is None:
            count = self.max_filings

        logger.info(
            "Fetching up to %d %s filings for %s",
            count,
            form_type,
            ticker,
        )

        company = self._get_company(ticker)
        filings = self._get_filings(
            company, form_type, year=year, start_date=start_date, end_date=end_date
        )

        # When a base form is requested, filter out amendments; then limit.
        # Slice lazily: one row past ``count`` is enough to know more exist.
        eligible = self._eligible(filings, form_type)
        filings_list = list(islice(eligible, count))

        if next(eligible, None) is not None:
            logger.info(
                "Limiting to %d %s filings; more are available",
                count,
                form_type,
            )

        fetched_count = 0
        for filing in filings_list:
            try:
                filing_id, html_content = self._fetch_filing_content(filing, ticker, form_type)
                fetched_count += 1
                logger.debug(
                    "Fetched %d/%d: %s",
                    fetched_count,
                    len(filings_list),
                    filing_id.accession_number,
                )
                yield filing_id, html_content

            except FetchError as e:
                logger.warning(
                    "Skipping filing %s: %s",
                    filing.accession_no,
                    e.message,
                )
                continue

        logger.info(
            "Completed: fetched %d %s filings for %s",
            fetched_count,
            form_type,
            ticker,
        )

    def fetch_by_accession(
        self,
        ticker: str,
        form_type: str,
        accession_number: str,
    ) -> tuple[FilingIdentifier, str]:
        """
        Fetch a specific filing by its accession number.

        Use this method when you know the exact accession number of the
        filing you want. This is useful for re-fetching a specific filing
        or when the user has selected from list_available().

        Args:
            ticker: Stock ticker symbol
            form_type: SEC form type ("8-K", "10-K", or "10-Q")
            accession_number: SEC accession number (e.g., "0000320193-23-000077")

        Returns:
            Tuple of (FilingIdentifier, html_content)

        Raises:
            FetchError: If filing not found or fetch fails

        Example:
            >>> filing_id, html = fetcher.fetch_by_accession(
            ...     "AAPL", "10-K", "0000320193-23-000077"
            ... )
        """
        form_type = self._validate_form_type(form_type)
        ticker = ticker.upper()

        logger.info(
            "Fetching %s %s by accession: %s",
            ticker,
            form_type,
            accession_number,
        )

        company = self._get_company(ticker)
        filings = self._get_filings(company, form_type)

        # Search for matching accession number, skipping mismatched forms.
        for filing in filings:
            if self._should_skip(filing, form_type):
                continue
            if filing.accession_no == accession_number:
                filing_id, html_content = self._fetch_filing_content(filing, ticker, form_type)
                logger.info(
                    "Fetched %s %s (%s): %s characters",
                    ticker,
                    form_type,
                    filing_id.date_str,
                    f"{len(html_content):,}",
                )
                return filing_id, html_content

        raise FetchError(
            f"Filing not found: {accession_number}",
            details=f"No {form_type} filing with this accession number for {ticker}.",
        )
