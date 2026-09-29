"""
SQLite metadata registry for tracking ingested SEC filings.

This module provides a lightweight relational layer for operations that
ChromaDB does not handle well: duplicate detection, listing with filters,
aggregation statistics, and filing limit enforcement.

When ``DB_ENCRYPTION_KEY`` is set, the module uses ``pysqlcipher3`` (a
drop-in replacement for Python's ``sqlite3``) and issues ``PRAGMA key``
immediately after opening the connection.  When the key is unset **or**
``pysqlcipher3`` is not installed, standard ``sqlite3`` is used — this
is the default for local development (Scenario A).

Usage:
    from sec_semantic_search.database import MetadataRegistry

    registry = MetadataRegistry()
    registry.register_filing(filing_id, chunk_count=59)
    filings = registry.list_filings(ticker="AAPL")
"""

import json
import os
import re
import sqlite3
import threading
import types
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sec_semantic_search.config import get_settings
from sec_semantic_search.core import (
    DatabaseError,
    FilingIdentifier,
    FilingLimitExceededError,
    Segment,
    get_logger,
)

logger = get_logger(__name__)


def _get_sqlite_module(encryption_key: str | None) -> types.ModuleType:
    """Return the appropriate SQLite driver module.

    When *encryption_key* is set, attempts to import ``pysqlcipher3``.
    Falls back to the standard ``sqlite3`` module if the key is unset
    or if ``pysqlcipher3`` is not installed (with a warning).
    """
    if encryption_key:
        try:
            from pysqlcipher3 import dbapi2 as sqlcipher  # type: ignore[import-untyped]

            logger.info("SQLCipher driver loaded — database encryption enabled")
            return sqlcipher
        except ImportError:
            logger.warning(
                "DB_ENCRYPTION_KEY is set but pysqlcipher3 is not installed. "
                "Install it with: pip install sec-semantic-search[encryption]. "
                "Falling back to unencrypted sqlite3."
            )
    return sqlite3


def _resolve_runtime_encryption_key(default_key: str | None) -> str | None:
    """Resolve the current SQLCipher key from env vars without revalidating paths.

    MetadataRegistry supports tests that override database paths with temporary
    directories outside the project root. Re-instantiating DatabaseSettings()
    would re-run path validation and fail those tests. For the registry we only
    need the encryption key fields, so resolve them directly from the current
    process environment and fall back to the already-loaded settings value.
    """
    from sec_semantic_search.config.settings import resolve_encryption_key_from_values

    env_key = os.environ.get("DB_ENCRYPTION_KEY")
    env_key_file = os.environ.get("DB_ENCRYPTION_KEY_FILE")

    # If either env var is set, resolve from them (ignoring default_key).
    if env_key or env_key_file:
        return resolve_encryption_key_from_values(env_key, env_key_file)

    return default_key


# SEC accession number pattern: NNNNNNNNNN-NN-NNNNNN (with or without dashes).
_ACCESSION_RE = re.compile(r"\b\d{10}-?\d{2}-?\d{6}\b")


def _scrub_error_message(
    error: str | None,
    tickers: list[str],
) -> str | None:
    """Remove ticker symbols and accession numbers from an error message.

    Replaces known ticker symbols (case-insensitive, word-boundary match)
    with ``[TICKER]`` and SEC accession numbers with ``[ACCESSION]``.
    This prevents research-target identifiers from leaking into persisted
    task history even when the error itself is stored.

    Args:
        error: The raw error message (may be ``None``).
        tickers: Ticker symbols associated with the task.

    Returns:
        The scrubbed message, or ``None`` if *error* was ``None``.
    """
    if not error:
        return error

    scrubbed = error

    # Build a single alternation regex for all tickers and apply once,
    # instead of recompiling a separate pattern per ticker.
    valid_tickers = [t for t in tickers if t]
    if valid_tickers:
        alternatives = "|".join(re.escape(t) for t in valid_tickers)
        ticker_pattern = re.compile(
            rf"\b(?:{alternatives})\b",
            re.IGNORECASE,
        )
        scrubbed = ticker_pattern.sub("[TICKER]", scrubbed)

    # Replace accession numbers (NNNNNNNNNN-NN-NNNNNN format).
    scrubbed = _ACCESSION_RE.sub("[ACCESSION]", scrubbed)

    return scrubbed


@dataclass
class TickerStatistics:
    """
    Aggregated statistics for a single ticker.

    Produced by ``MetadataRegistry.get_statistics()`` from a SQL
    ``GROUP BY`` query. Avoids fetching full rows just to count them.

    Attributes:
        ticker: Stock ticker symbol (e.g., "AAPL").
        filings: Total number of filings for this ticker.
        chunks: Total chunks across all filings for this ticker.
        forms: Sorted list of distinct form types (e.g., ["10-K", "10-Q"]).
    """

    ticker: str
    filings: int
    chunks: int
    forms: list[str]


@dataclass
class DatabaseStatistics:
    """
    Aggregated database statistics computed entirely in SQL.

    Produced by ``MetadataRegistry.get_statistics()``. Replaces the
    pattern of fetching all filing rows and iterating in Python.

    Attributes:
        filing_count: Total number of ingested filings.
        tickers: Sorted list of unique ticker symbols.
        form_breakdown: Filing count per form type (e.g., {"10-K": 5}).
        ticker_breakdown: Per-ticker aggregated statistics.
    """

    filing_count: int
    tickers: list[str]
    form_breakdown: dict[str, int]
    ticker_breakdown: list[TickerStatistics]


@dataclass
class FilingRecord:
    """
    A single row from the filings table.

    Provides typed access to filing metadata rather than raw tuples or dicts.
    Used by the CLI ``manage list`` and ``manage status`` commands.

    Attributes:
        id: Auto-increment primary key.
        ticker: Stock ticker symbol (e.g., "AAPL").
        form_type: SEC form type (e.g., "10-K").
        filing_date: Filing date as ISO string (YYYY-MM-DD).
        accession_number: SEC accession number (unique).
        chunk_count: Number of chunks stored in ChromaDB.
        ingested_at: ISO timestamp of when the filing was ingested.
    """

    id: int
    ticker: str
    form_type: str
    filing_date: str
    accession_number: str
    chunk_count: int
    ingested_at: str


class MetadataRegistry:
    """
    SQLite registry for tracking ingested SEC filings.

    This class manages a single ``filings`` table that records which filings
    have been ingested, their chunk counts, and ingestion timestamps. It
    provides duplicate detection, listing, statistics, and filing limit
    enforcement.

    The database file is created automatically on first use. The table
    schema is created via ``CREATE TABLE IF NOT EXISTS``, so repeated
    initialisation is safe.

    Example:
        >>> registry = MetadataRegistry()
        >>> registry.register_filing(filing_id, chunk_count=59)
        >>> print(registry.count())
        1
    """

    def __init__(
        self,
        db_path: str | None = None,
        encryption_key: str | None = None,
    ) -> None:
        """
        Initialise the metadata registry.

        Opens two persistent SQLite connections, each behind its own
        threading lock: one for writes and one, ``query_only``, for reads.
        In WAL mode readers do not block on the writer, but only across
        connections — so a search's parent lookup no longer waits for an
        ingest's segment inserts.

        When *encryption_key* is provided (or ``DB_ENCRYPTION_KEY`` is set),
        both connections use ``pysqlcipher3`` and issue ``PRAGMA key``
        immediately after opening.

        Args:
            db_path: Path to SQLite database file. If None, uses
                     ``settings.database.metadata_db_path``.
            encryption_key: SQLCipher encryption key. If None, reads from
                            ``settings.database.encryption_key``.
        """
        settings = get_settings()
        self._db_path = db_path or settings.database.metadata_db_path
        self._max_filings = settings.database.max_filings
        self._encryption_key = (
            encryption_key
            if encryption_key is not None
            else _resolve_runtime_encryption_key(settings.database.encryption_key)
        )

        # Ensure parent directory exists
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        # Select the appropriate SQLite driver (sqlcipher or sqlite3).
        self._sqlite_module = _get_sqlite_module(self._encryption_key)

        # Write connection — every INSERT/UPDATE/DELETE and the schema.
        # check_same_thread=False allows the API's background worker
        # threads to use the same connection; the lock serialises access.
        self._conn = self._connect()
        self._conn.execute("PRAGMA journal_mode=WAL")
        # In WAL mode NORMAL keeps the database consistent across an
        # application crash and drops the fsync on every commit; a power
        # loss can lose the last transactions. Acceptable for a registry
        # whose filings are re-ingestible (see DEPLOYMENT.md).
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA temp_store=MEMORY")
        self._lock = threading.Lock()

        self._encrypted = self._encryption_key is not None and self._sqlite_module is not sqlite3

        # Cache the driver's exception classes.  When using pysqlcipher3,
        # its Error/IntegrityError are distinct from sqlite3's, so we must
        # catch the right ones.
        self._db_error = self._sqlite_module.Error
        self._db_integrity_error = self._sqlite_module.IntegrityError

        # Create table on init (idempotent)
        self._create_table()

        # Read connection — opened after the schema exists and WAL is on.
        # Autocommit (isolation_level=None): each SELECT sees the latest
        # committed data, and a multi-statement read opens its own
        # transaction (``list_filings_page``). ``query_only`` makes any
        # write through it fail, so reads can never bypass the write lock.
        self._read_conn = self._connect(isolation_level=None)
        self._read_conn.execute("PRAGMA query_only=ON")
        self._read_conn.execute("PRAGMA temp_store=MEMORY")
        self._read_lock = threading.Lock()
        try:
            # Pays the SQLCipher key derivation now and fails fast on a
            # wrong key, rather than on the first search.
            self._read_conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()
        except self._db_error as e:
            self.close()
            raise DatabaseError(
                "Failed to open metadata read connection",
                details=str(e),
            ) from e

        logger.debug(
            "MetadataRegistry initialised: %s (encrypted=%s)",
            self._db_path,
            self._encrypted,
        )

    def _connect(self, **kwargs: Any) -> Any:
        """Open a connection to the registry database.

        When using SQLCipher, ``PRAGMA key`` MUST be the very first
        statement executed on the connection — before any other PRAGMA
        or query. PRAGMA does not support parameter binding, so the key
        is hex-encoded as a blob literal to avoid any SQL injection risk
        (defence in depth, even though the key comes from env vars).
        """
        conn = self._sqlite_module.connect(
            self._db_path,
            check_same_thread=False,
            **kwargs,
        )
        if self._encryption_key and self._sqlite_module is not sqlite3:
            hex_key = self._encryption_key.encode().hex()
            conn.execute(f"PRAGMA key = \"x'{hex_key}'\"")
            logger.debug("SQLCipher PRAGMA key applied")
        conn.row_factory = self._sqlite_module.Row
        return conn

    def close(self) -> None:
        """Close both database connections."""
        self._read_conn.close()
        self._conn.close()
        logger.debug("MetadataRegistry connection closed: %s", self._db_path)

    @property
    def encrypted(self) -> bool:
        """Whether the database connection is using SQLCipher encryption."""
        return self._encrypted

    def _create_table(self) -> None:
        """Create the filings, segments, and task_history tables if missing."""
        filings_sql = """
            CREATE TABLE IF NOT EXISTS filings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                form_type TEXT NOT NULL,
                filing_date TEXT NOT NULL,
                accession_number TEXT NOT NULL UNIQUE,
                chunk_count INTEGER NOT NULL,
                ingested_at TEXT NOT NULL,
                UNIQUE(ticker, form_type, filing_date)
            )
        """
        # ``segments`` holds the full parent-context text keyed by
        # (accession_number, segment_index). Search results are joined
        # against this table at query time so the API can return the
        # broad paragraph for display while keeping the embedded chunk
        # in ChromaDB. Cleanup is driven by the dual-store delete path
        # (no FOREIGN KEY cascade — SQLite requires PRAGMA foreign_keys
        # to be set per connection and the registry already manages
        # filing deletion explicitly).
        segments_sql = """
            CREATE TABLE IF NOT EXISTS segments (
                accession_number TEXT NOT NULL,
                segment_index INTEGER NOT NULL,
                path TEXT NOT NULL,
                content_type TEXT NOT NULL,
                content TEXT NOT NULL,
                PRIMARY KEY (accession_number, segment_index)
            )
        """
        task_history_sql = """
            CREATE TABLE IF NOT EXISTS task_history (
                task_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                tickers TEXT,
                form_types TEXT NOT NULL,
                results TEXT NOT NULL,
                error TEXT,
                started_at TEXT,
                completed_at TEXT,
                filings_done INTEGER NOT NULL DEFAULT 0,
                filings_skipped INTEGER NOT NULL DEFAULT 0,
                filings_failed INTEGER NOT NULL DEFAULT 0
            )
        """
        index_sql = """
            CREATE INDEX IF NOT EXISTS idx_filings_ingested_at
            ON filings (ingested_at)
        """
        try:
            with self._lock, self._conn:
                self._conn.execute(filings_sql)
                self._conn.execute(segments_sql)
                self._conn.execute(index_sql)
                self._conn.execute(task_history_sql)
        except self._db_error as e:
            raise DatabaseError(
                "Failed to create metadata tables",
                details=str(e),
            ) from e

    # ------------------------------------------------------------------
    # Pre-ingestion checks
    # ------------------------------------------------------------------

    def check_filing_limit(self) -> None:
        """
        Raise FilingLimitExceededError if the filing limit is reached.

        This should be called before ingesting a new filing to prevent
        exceeding the configured maximum.

        Raises:
            FilingLimitExceededError: If current count >= max_filings.
            DatabaseError: If the query fails.
        """
        current = self.count()
        if current >= self._max_filings:
            raise FilingLimitExceededError(current, self._max_filings)

    def is_duplicate(self, accession_number: str) -> bool:
        """
        Check whether a filing has already been ingested.

        Args:
            accession_number: SEC accession number to check.

        Returns:
            True if the filing exists in the registry.

        Raises:
            DatabaseError: If the query fails.
        """
        sql = "SELECT 1 FROM filings WHERE accession_number = ? LIMIT 1"
        try:
            with self._read_lock:
                row = self._read_conn.execute(sql, (accession_number,)).fetchone()
            return row is not None
        except self._db_error as e:
            raise DatabaseError(
                "Failed to check for duplicate filing",
                details=str(e),
            ) from e

    def get_existing_accessions(
        self,
        accession_numbers: list[str],
    ) -> set[str]:
        """
        Return the subset of accession numbers that already exist in the registry.

        Performs a single ``SELECT ... WHERE IN (...)`` query instead of
        N individual ``is_duplicate()`` calls, reducing SQLite connection
        overhead from O(N) to O(1) for batch operations.

        Args:
            accession_numbers: Accession numbers to check.

        Returns:
            Set of accession numbers that are already registered.

        Raises:
            DatabaseError: If the query fails.
        """
        if not accession_numbers:
            return set()

        placeholders = ", ".join("?" for _ in accession_numbers)
        sql = f"SELECT accession_number FROM filings WHERE accession_number IN ({placeholders})"
        try:
            with self._read_lock:
                rows = self._read_conn.execute(sql, accession_numbers).fetchall()
            return {row["accession_number"] for row in rows}
        except self._db_error as e:
            raise DatabaseError(
                "Failed to check for existing accessions",
                details=str(e),
            ) from e

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def register_filing(
        self,
        filing_id: FilingIdentifier,
        chunk_count: int,
        segments: list[Segment] | None = None,
    ) -> None:
        """
        Register a newly ingested filing in the metadata registry.

        When ``segments`` is provided, the filing row and all segment rows
        are inserted atomically inside the same transaction — either both
        land or neither does. This preserves the dual-store invariant
        (segments must always be resolvable for every registered filing)
        without introducing a separate failure mode.

        Args:
            filing_id: Identifier of the ingested filing.
            chunk_count: Number of chunks stored in ChromaDB.
            segments: Optional parent segments to persist for parent-context
                resolution at search time. ``None`` is accepted so the
                method stays usable in tests that do not exercise the
                segment-storage path.

        Raises:
            DatabaseError: If the insert fails (e.g., duplicate).
        """
        sql = """
            INSERT INTO filings (ticker, form_type, filing_date,
                                 accession_number, chunk_count, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        ingested_at = datetime.now(UTC).isoformat()

        try:
            with self._lock, self._conn:
                self._conn.execute(
                    sql,
                    (
                        filing_id.ticker,
                        filing_id.form_type,
                        filing_id.date_str,
                        filing_id.accession_number,
                        chunk_count,
                        ingested_at,
                    ),
                )
                if segments:
                    self._insert_segments(filing_id.accession_number, segments)
            logger.info(
                "Registered filing: %s %s (%s) — %d chunks, %d segments",
                filing_id.ticker,
                filing_id.form_type,
                filing_id.date_str,
                chunk_count,
                len(segments) if segments else 0,
            )
        except self._db_integrity_error as e:
            raise DatabaseError(
                f"Filing already exists: {filing_id.accession_number}",
                details=str(e),
            ) from e
        except self._db_error as e:
            raise DatabaseError(
                "Failed to register filing",
                details=str(e),
            ) from e

    def register_filing_if_new(
        self,
        filing_id: FilingIdentifier,
        chunk_count: int,
        segments: list[Segment] | None = None,
    ) -> bool:
        """
        Atomically check for duplicate and register a filing if new.

        Holds the threading lock across both the duplicate check and the
        insert, closing the race window where two threads could both pass
        ``is_duplicate()`` and then both attempt ``register_filing()``.

        When ``segments`` is provided, the filing row and segment rows
        are inserted in the same transaction so the parent-context store
        is consistent with the filings table at all times.

        Args:
            filing_id: Identifier of the filing to register.
            chunk_count: Number of chunks stored in ChromaDB.
            segments: Optional parent segments to persist alongside the
                filing row. See ``register_filing`` for details.

        Returns:
            True if the filing was registered, False if it already existed.

        Raises:
            DatabaseError: If the query or insert fails.
        """
        sql_check = "SELECT 1 FROM filings WHERE accession_number = ? LIMIT 1"
        sql_insert = """
            INSERT INTO filings (ticker, form_type, filing_date,
                                 accession_number, chunk_count, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        ingested_at = datetime.now(UTC).isoformat()

        try:
            with self._lock, self._conn:
                exists = self._conn.execute(
                    sql_check,
                    (filing_id.accession_number,),
                ).fetchone()
                if exists is not None:
                    logger.debug(
                        "Filing already registered (atomic check): %s",
                        filing_id.accession_number,
                    )
                    return False

                self._conn.execute(
                    sql_insert,
                    (
                        filing_id.ticker,
                        filing_id.form_type,
                        filing_id.date_str,
                        filing_id.accession_number,
                        chunk_count,
                        ingested_at,
                    ),
                )
                if segments:
                    self._insert_segments(filing_id.accession_number, segments)
        except self._db_integrity_error:
            # Defensive: UNIQUE constraint caught a race despite the check.
            logger.debug(
                "Filing already registered (integrity constraint): %s",
                filing_id.accession_number,
            )
            return False
        except self._db_error as e:
            raise DatabaseError(
                "Failed to register filing atomically",
                details=str(e),
            ) from e

        logger.info(
            "Registered filing: %s %s (%s) — %d chunks, %d segments",
            filing_id.ticker,
            filing_id.form_type,
            filing_id.date_str,
            chunk_count,
            len(segments) if segments else 0,
        )
        return True

    def remove_filing(self, accession_number: str) -> bool:
        """
        Remove a filing (and its stored segments) from the registry.

        Args:
            accession_number: SEC accession number of the filing to remove.

        Returns:
            True if a filing was removed, False if not found.

        Raises:
            DatabaseError: If the delete fails.
        """
        sql = "DELETE FROM filings WHERE accession_number = ?"
        sql_segments = "DELETE FROM segments WHERE accession_number = ?"
        try:
            with self._lock, self._conn:
                cursor = self._conn.execute(sql, (accession_number,))
                removed = cursor.rowcount > 0
                # Always purge segments — even on a stale row where the
                # filing entry is gone, leaving orphans behind would
                # confuse the next ingest with the same accession.
                self._conn.execute(sql_segments, (accession_number,))
            if removed:
                logger.info("Removed filing from registry: %s", accession_number)
            else:
                logger.warning("Filing not found in registry: %s", accession_number)
            return removed
        except self._db_error as e:
            raise DatabaseError(
                "Failed to remove filing",
                details=str(e),
            ) from e

    def remove_filings_batch(self, accession_numbers: list[str]) -> int:
        """
        Remove multiple filings in batched SQL statements.

        Uses ``DELETE ... WHERE accession_number IN (...)`` to remove
        filings in bulk, reducing SQLite round-trips from O(N) to O(1)
        (or O(N/999) for very large batches due to SQLite's parameter
        limit).

        Args:
            accession_numbers: Accession numbers to delete.

        Returns:
            Total number of rows removed.

        Raises:
            DatabaseError: If any batch delete fails.
        """
        if not accession_numbers:
            return 0

        removed = 0
        # SQLite supports at most 999 bound parameters per statement.
        chunk_size = 999
        for i in range(0, len(accession_numbers), chunk_size):
            batch = accession_numbers[i : i + chunk_size]
            placeholders = ", ".join("?" for _ in batch)
            sql = f"DELETE FROM filings WHERE accession_number IN ({placeholders})"
            sql_segments = f"DELETE FROM segments WHERE accession_number IN ({placeholders})"
            try:
                with self._lock, self._conn:
                    cursor = self._conn.execute(sql, batch)
                    removed += cursor.rowcount
                    # Cascade to segments in the same transaction.
                    self._conn.execute(sql_segments, batch)
            except self._db_error as e:
                raise DatabaseError(
                    "Failed to remove filings batch",
                    details=str(e),
                ) from e

        if removed:
            logger.info("Batch-removed %d filing(s) from registry", removed)
        return removed

    def clear_all(self) -> int:
        """
        Delete all rows from the filings table.

        More efficient than loading all rows via ``list_filings()`` and
        passing them to ``remove_filings_batch()`` — executes a single
        ``DELETE FROM filings`` without fetching any data into memory.

        Returns:
            Number of filings deleted.

        Raises:
            DatabaseError: If the delete fails.
        """
        sql = "DELETE FROM filings"
        sql_segments = "DELETE FROM segments"
        try:
            with self._lock, self._conn:
                cursor = self._conn.execute(sql)
                removed = cursor.rowcount
                # Wipe parent-context store in the same transaction so the
                # two stays consistent.
                self._conn.execute(sql_segments)
            if removed:
                logger.info("Cleared all filings from registry: %d removed", removed)
            return removed
        except self._db_error as e:
            raise DatabaseError(
                "Failed to clear all filings",
                details=str(e),
            ) from e

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_filing(self, accession_number: str) -> FilingRecord | None:
        """
        Retrieve a single filing record by accession number.

        Args:
            accession_number: SEC accession number.

        Returns:
            FilingRecord if found, None otherwise.

        Raises:
            DatabaseError: If the query fails.
        """
        sql = "SELECT * FROM filings WHERE accession_number = ?"
        try:
            with self._read_lock:
                row = self._read_conn.execute(sql, (accession_number,)).fetchone()
            if row is None:
                return None
            return self._row_to_record(row)
        except self._db_error as e:
            raise DatabaseError(
                "Failed to retrieve filing",
                details=str(e),
            ) from e

    def get_filings_by_accessions(
        self,
        accession_numbers: list[str],
    ) -> list[FilingRecord]:
        """
        Retrieve multiple filing records in a single query.

        Uses ``SELECT ... WHERE accession_number IN (...)`` to fetch
        all matching records in one round-trip instead of N individual
        ``get_filing()`` calls.

        Args:
            accession_numbers: Accession numbers to look up.

        Returns:
            List of FilingRecord objects for accession numbers that
            exist.  Order is not guaranteed.  Accession numbers not
            found are silently omitted.

        Raises:
            DatabaseError: If the query fails.
        """
        if not accession_numbers:
            return []

        records: list[FilingRecord] = []
        chunk_size = 999
        for i in range(0, len(accession_numbers), chunk_size):
            batch = accession_numbers[i : i + chunk_size]
            placeholders = ", ".join("?" for _ in batch)
            sql = f"SELECT * FROM filings WHERE accession_number IN ({placeholders})"
            try:
                with self._read_lock:
                    rows = self._read_conn.execute(sql, batch).fetchall()
                records.extend(self._row_to_record(row) for row in rows)
            except self._db_error as e:
                raise DatabaseError(
                    "Failed to retrieve filings batch",
                    details=str(e),
                ) from e
        return records

    def list_filings(
        self,
        ticker: str | None = None,
        form_type: str | None = None,
    ) -> list[FilingRecord]:
        """
        List ingested filings with optional filters.

        Args:
            ticker: Filter by ticker symbol (case-insensitive).
            form_type: Filter by form type (case-insensitive).

        Returns:
            List of FilingRecord objects, ordered by filing_date descending.

        Raises:
            DatabaseError: If the query fails.
        """
        sql = "SELECT * FROM filings WHERE 1=1"
        params: list = []

        if ticker:
            sql += " AND ticker = ?"
            params.append(ticker.upper())
        if form_type:
            sql += " AND form_type = ?"
            params.append(form_type.upper())

        sql += " ORDER BY filing_date DESC"

        try:
            with self._read_lock:
                rows = self._read_conn.execute(sql, params).fetchall()
            return [self._row_to_record(row) for row in rows]
        except self._db_error as e:
            raise DatabaseError(
                "Failed to list filings",
                details=str(e),
            ) from e

    # Columns ``list_filings_page`` may order by. ``ORDER BY`` takes no bound
    # parameters, so the column and direction are checked against these
    # before being written into the SQL.
    SORTABLE_COLUMNS = frozenset(
        {"filing_date", "ticker", "form_type", "chunk_count", "ingested_at"}
    )

    def list_filings_page(
        self,
        ticker: str | None = None,
        form_type: str | None = None,
        *,
        sort_by: str = "filing_date",
        order: str = "desc",
        limit: int = 25,
        offset: int = 0,
    ) -> tuple[list[FilingRecord], int]:
        """
        Return one sorted page of filings and the total matching the filters.

        Sorting and slicing happen in SQLite; ``id`` breaks ties so pages
        never overlap or skip rows. The count and the page are read in
        one read transaction, so ``total`` always describes the same snapshot.

        Args:
            ticker: Filter by ticker symbol (case-insensitive).
            form_type: Filter by form type (case-insensitive).
            sort_by: One of ``SORTABLE_COLUMNS``.
            order: ``"asc"`` or ``"desc"``.
            limit: Page size (≥ 1).
            offset: Rows to skip (≥ 0).

        Returns:
            ``(records, total)``.

        Raises:
            ValueError: If the sort column, order, limit or offset is invalid.
            DatabaseError: If the query fails.
        """
        if sort_by not in self.SORTABLE_COLUMNS:
            raise ValueError(f"Cannot sort filings by {sort_by!r}")
        if order not in ("asc", "desc"):
            raise ValueError(f"Sort order must be 'asc' or 'desc', not {order!r}")
        if limit < 1 or offset < 0:
            raise ValueError("limit must be ≥ 1 and offset ≥ 0")

        where = " WHERE 1=1"
        params: list = []
        if ticker:
            where += " AND ticker = ?"
            params.append(ticker.upper())
        if form_type:
            where += " AND form_type = ?"
            params.append(form_type.upper())

        direction = order.upper()
        page_sql = (
            f"SELECT * FROM filings{where} "
            f"ORDER BY {sort_by} {direction}, id {direction} LIMIT ? OFFSET ?"
        )

        try:
            with self._read_lock:
                # One read transaction: a commit on the write connection
                # between the two statements cannot make them disagree.
                self._read_conn.execute("BEGIN")
                try:
                    total = self._read_conn.execute(
                        f"SELECT COUNT(*) FROM filings{where}", params
                    ).fetchone()[0]
                    rows = self._read_conn.execute(page_sql, [*params, limit, offset]).fetchall()
                finally:
                    self._read_conn.execute("COMMIT")
            return [self._row_to_record(row) for row in rows], total
        except self._db_error as e:
            raise DatabaseError(
                "Failed to list filings",
                details=str(e),
            ) from e

    def list_oldest_filings(self, limit: int) -> list[FilingRecord]:
        """
        Return the oldest filings ordered by ingestion time (ascending).

        Used by FIFO eviction in demo mode to identify which filings
        to delete when the database approaches its capacity limit.

        Args:
            limit: Maximum number of filings to return.

        Returns:
            List of FilingRecord objects, ordered by ``ingested_at`` ASC.

        Raises:
            DatabaseError: If the query fails.
        """
        sql = "SELECT * FROM filings ORDER BY ingested_at ASC LIMIT ?"
        try:
            with self._read_lock:
                rows = self._read_conn.execute(sql, (limit,)).fetchall()
            return [self._row_to_record(row) for row in rows]
        except self._db_error as e:
            raise DatabaseError(
                "Failed to list oldest filings",
                details=str(e),
            ) from e

    def count(
        self,
        ticker: str | None = None,
        form_type: str | None = None,
    ) -> int:
        """
        Count ingested filings with optional filters.

        Args:
            ticker: Filter by ticker symbol.
            form_type: Filter by form type.

        Returns:
            Number of matching filings.

        Raises:
            DatabaseError: If the query fails.
        """
        sql = "SELECT COUNT(*) FROM filings WHERE 1=1"
        params: list = []

        if ticker:
            sql += " AND ticker = ?"
            params.append(ticker.upper())
        if form_type:
            sql += " AND form_type = ?"
            params.append(form_type.upper())

        try:
            with self._read_lock:
                row = self._read_conn.execute(sql, params).fetchone()
            return row[0]
        except self._db_error as e:
            raise DatabaseError(
                "Failed to count filings",
                details=str(e),
            ) from e

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def get_statistics(self) -> DatabaseStatistics:
        """
        Return aggregated database statistics computed in SQL.

        Runs a single ``GROUP BY ticker, form_type`` query and derives
        all aggregates from the result. Much more efficient than fetching
        all rows and iterating in Python.

        Returns:
            DatabaseStatistics with filing count, tickers, form breakdown,
            and per-ticker breakdown.

        Raises:
            DatabaseError: If the query fails.
        """
        sql = """
            SELECT ticker, form_type, COUNT(*) AS filings,
                   SUM(chunk_count) AS chunks
            FROM filings
            GROUP BY ticker, form_type
            ORDER BY ticker, form_type
        """
        try:
            with self._read_lock:
                rows = self._read_conn.execute(sql).fetchall()
        except self._db_error as e:
            raise DatabaseError(
                "Failed to retrieve database statistics",
                details=str(e),
            ) from e

        # Derive all aggregates from the grouped rows.
        filing_count = 0
        form_breakdown: dict[str, int] = {}
        ticker_data: dict[str, dict] = {}

        for row in rows:
            ticker = row["ticker"]
            form_type = row["form_type"]
            filings = row["filings"]
            chunks = row["chunks"]

            filing_count += filings
            form_breakdown[form_type] = form_breakdown.get(form_type, 0) + filings

            if ticker not in ticker_data:
                ticker_data[ticker] = {"filings": 0, "chunks": 0, "forms": []}
            ticker_data[ticker]["filings"] += filings
            ticker_data[ticker]["chunks"] += chunks
            ticker_data[ticker]["forms"].append(form_type)

        # ticker_data is already sorted by ticker because the SQL
        # query uses ORDER BY ticker, form_type and Python 3.7+ dicts
        # preserve insertion order.  No need to re-sort.
        ticker_breakdown = [
            TickerStatistics(
                ticker=ticker,
                filings=data["filings"],
                chunks=data["chunks"],
                forms=sorted(data["forms"]),
            )
            for ticker, data in ticker_data.items()
        ]

        return DatabaseStatistics(
            filing_count=filing_count,
            tickers=list(ticker_data.keys()),
            form_breakdown=dict(sorted(form_breakdown.items())),
            ticker_breakdown=ticker_breakdown,
        )

    # ------------------------------------------------------------------
    # Task history (persistence for completed/failed/cancelled tasks)
    # ------------------------------------------------------------------

    def save_task_history(
        self,
        task_id: str,
        *,
        status: str,
        tickers: list[str],
        form_types: list[str],
        results: list[dict[str, Any]],
        error: str | None = None,
        started_at: str | None = None,
        completed_at: str | None = None,
        filings_done: int = 0,
        filings_skipped: int = 0,
        filings_failed: int = 0,
    ) -> None:
        """Persist a completed task's metadata to SQLite.

        Called by ``TaskManager`` just before pruning the in-memory entry.
        Uses ``INSERT OR REPLACE`` so re-saving is idempotent.

        Privacy controls applied here:

        - **Ticker stripping:** When ``TASK_HISTORY_PERSIST_TICKERS`` is
          ``false`` (the default), the ``tickers`` column stores ``null``
          instead of the actual ticker list.  This prevents research-target
          patterns from being persisted to disk.
        - **Error scrubbing:** Ticker symbols and accession numbers are
          removed from ``error`` text before storage, regardless of the
          ticker-persist setting (error messages can leak identifiers even
          when tickers are stripped).
        """
        settings = get_settings()
        persist_tickers = settings.database.task_history_persist_tickers

        # Privacy: strip tickers when not explicitly opted in.
        tickers_json = json.dumps(tickers) if persist_tickers else None

        # Privacy: scrub error messages of ticker/accession references.
        scrubbed_error = _scrub_error_message(error, tickers) if error else None

        sql = """
            INSERT OR REPLACE INTO task_history
                (task_id, status, tickers, form_types, results, error,
                 started_at, completed_at,
                 filings_done, filings_skipped, filings_failed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        try:
            with self._lock, self._conn:
                self._conn.execute(
                    sql,
                    (
                        task_id,
                        status,
                        tickers_json,
                        json.dumps(form_types),
                        json.dumps(results),
                        scrubbed_error,
                        started_at,
                        completed_at,
                        filings_done,
                        filings_skipped,
                        filings_failed,
                    ),
                )
            logger.debug("Persisted task history: %s", task_id[:8])
        except self._db_error as e:
            raise DatabaseError(
                "Failed to save task history",
                details=str(e),
            ) from e

    def get_task_history(self, task_id: str) -> dict[str, Any] | None:
        """Retrieve a persisted task by ID.

        Returns a dict matching the ``TaskStatus`` schema fields, or
        ``None`` if the task was never persisted.
        """
        sql = "SELECT * FROM task_history WHERE task_id = ?"
        try:
            with self._read_lock:
                row = self._read_conn.execute(sql, (task_id,)).fetchone()
            if row is None:
                return None
            return {
                "task_id": row["task_id"],
                "status": row["status"],
                "tickers": json.loads(row["tickers"]) if row["tickers"] else [],
                "form_types": json.loads(row["form_types"]),
                "results": json.loads(row["results"]),
                "error": row["error"],
                "started_at": row["started_at"],
                "completed_at": row["completed_at"],
                "filings_done": row["filings_done"],
                "filings_skipped": row["filings_skipped"],
                "filings_failed": row["filings_failed"],
            }
        except self._db_error as e:
            raise DatabaseError(
                "Failed to retrieve task history",
                details=str(e),
            ) from e

    def prune_task_history(self, max_age_days: int | None = None) -> int:
        """Remove task history entries older than *max_age_days*.

        When *max_age_days* is ``None`` (the default), the value is read
        from ``TASK_HISTORY_RETENTION_DAYS``.  When the effective value
        is ``0``, pruning is skipped entirely (keep indefinitely).

        Returns the number of entries removed (``0`` when skipped).
        """
        if max_age_days is None:
            max_age_days = get_settings().database.task_history_retention_days

        if max_age_days <= 0:
            return 0  # 0 = keep indefinitely

        cutoff = datetime.now(UTC).isoformat()
        # SQLite date arithmetic: entries with completed_at older than cutoff
        sql = """
            DELETE FROM task_history
            WHERE completed_at IS NOT NULL
              AND completed_at < datetime(?, '-' || ? || ' days')
        """
        try:
            with self._lock, self._conn:
                cursor = self._conn.execute(sql, (cutoff, max_age_days))
                removed = cursor.rowcount
            if removed:
                logger.info(
                    "Pruned %d task history entries older than %d days",
                    removed,
                    max_age_days,
                )
            return removed
        except self._db_error as e:
            raise DatabaseError(
                "Failed to prune task history",
                details=str(e),
            ) from e

    # ------------------------------------------------------------------
    # Segments (parent-context store)
    # ------------------------------------------------------------------

    def _insert_segments(
        self,
        accession_number: str,
        segments: list[Segment],
    ) -> None:
        """Insert all segments for a filing.

        Called from inside ``register_filing`` / ``register_filing_if_new``
        while the threading lock and SQLite transaction are already held —
        the caller is responsible for both. We use ``INSERT OR REPLACE`` so
        a stale segment row left behind by an aborted ingest never blocks
        a fresh registration on the same accession.
        """
        sql = """
            INSERT OR REPLACE INTO segments
                (accession_number, segment_index, path, content_type, content)
            VALUES (?, ?, ?, ?, ?)
        """
        rows = [
            (
                accession_number,
                seg.segment_index,
                seg.path,
                seg.content_type.value,
                seg.content,
            )
            for seg in segments
        ]
        self._conn.executemany(sql, rows)

    def get_parent_segments(
        self,
        accession_segment_pairs: list[tuple[str, int]],
    ) -> dict[tuple[str, int], str]:
        """Bulk-fetch parent segment text for a set of chunk locations.

        Used by the search engine to attach the broader paragraph to each
        ChromaDB result before returning it to the caller. The single SQL
        round-trip replaces what would otherwise be O(N) point lookups.

        Args:
            accession_segment_pairs: Tuples of (accession_number,
                segment_index) identifying the segments to fetch.

        Returns:
            Mapping from each requested ``(accession, segment_index)``
            pair to its segment content. Missing keys indicate a chunk
            whose parent could not be resolved (e.g. legacy data
            ingested before the segments table was populated) — callers
            should fall back to the chunk text in that case.

        Raises:
            DatabaseError: If the query fails.
        """
        if not accession_segment_pairs:
            return {}

        # Deduplicate so we never bind the same pair twice — saves work
        # when multiple top-k results share a parent segment.
        unique_pairs = list(set(accession_segment_pairs))

        result: dict[tuple[str, int], str] = {}
        chunk_size = 499  # 499 pairs * 2 params = 998, under SQLite's 999 limit
        for i in range(0, len(unique_pairs), chunk_size):
            batch = unique_pairs[i : i + chunk_size]
            placeholders = ", ".join("(?, ?)" for _ in batch)
            sql = (
                "SELECT accession_number, segment_index, content "
                f"FROM segments WHERE (accession_number, segment_index) IN ({placeholders})"
            )
            params: list = []
            for accession, seg_idx in batch:
                params.append(accession)
                params.append(seg_idx)
            try:
                with self._read_lock:
                    rows = self._read_conn.execute(sql, params).fetchall()
            except self._db_error as e:
                raise DatabaseError(
                    "Failed to fetch parent segments",
                    details=str(e),
                ) from e
            for row in rows:
                result[(row["accession_number"], row["segment_index"])] = row["content"]

        return result

    def count_segments(self, accession_number: str | None = None) -> int:
        """Count rows in the segments table (optionally filtered).

        Mainly for tests and diagnostics — the production search path uses
        ``get_parent_segments`` for batched lookups instead.
        """
        if accession_number is None:
            sql = "SELECT COUNT(*) FROM segments"
            params: tuple = ()
        else:
            sql = "SELECT COUNT(*) FROM segments WHERE accession_number = ?"
            params = (accession_number,)
        try:
            with self._read_lock:
                row = self._read_conn.execute(sql, params).fetchone()
            return row[0]
        except self._db_error as e:
            raise DatabaseError(
                "Failed to count segments",
                details=str(e),
            ) from e

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_record(row: Any) -> FilingRecord:
        """Convert a SQLite Row to a FilingRecord dataclass."""
        return FilingRecord(
            id=row["id"],
            ticker=row["ticker"],
            form_type=row["form_type"],
            filing_date=row["filing_date"],
            accession_number=row["accession_number"],
            chunk_count=row["chunk_count"],
            ingested_at=row["ingested_at"],
        )
