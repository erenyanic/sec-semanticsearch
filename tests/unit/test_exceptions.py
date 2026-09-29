"""
Tests for the custom exception hierarchy.

The exception classes carry structured data (message + details) and
custom formatting. We verify:
    - Base class message formatting (with and without details)
    - The em-dash separator in _format_message()
    - Inheritance chain (all exceptions are SECSemanticSearchError)
    - FilingLimitExceededError's custom __init__ and attributes
    - str() output matches _format_message()
"""

import pytest

from sec_semantic_search.core.exceptions import (
    ChunkingError,
    DatabaseError,
    EmbeddingError,
    FetchError,
    FilingLimitExceededError,
    ParseError,
    SearchError,
    SECSemanticSearchError,
)


class TestBaseException:
    """SECSemanticSearchError is the root of the hierarchy."""

    def test_message_only(self):
        exc = SECSemanticSearchError("Something went wrong")
        assert exc.message == "Something went wrong"
        assert exc.details is None
        assert str(exc) == "Something went wrong"

    def test_message_with_details(self):
        exc = SECSemanticSearchError("Failed", details="Connection refused")
        assert exc.message == "Failed"
        assert exc.details == "Connection refused"
        assert str(exc) == "Failed — Connection refused"

    def test_format_message_em_dash(self):
        """The separator is an em-dash (—), not a hyphen (-)."""
        exc = SECSemanticSearchError("A", details="B")
        assert "—" in str(exc)
        assert " — " in str(exc)

    def test_is_exception(self):
        """Must be catchable as a standard Exception."""
        with pytest.raises(SECSemanticSearchError):
            raise SECSemanticSearchError("test")


class TestSubclassInheritance:
    """All domain exceptions must inherit from SECSemanticSearchError."""

    @pytest.mark.parametrize(
        "exc_class",
        [
            FetchError,
            ParseError,
            ChunkingError,
            EmbeddingError,
            DatabaseError,
            SearchError,
            FilingLimitExceededError,
        ],
    )
    def test_inherits_from_base(self, exc_class):
        assert issubclass(exc_class, SECSemanticSearchError)

    def test_filing_limit_inherits_from_database_error(self):
        """FilingLimitExceededError is a special case of DatabaseError."""
        assert issubclass(FilingLimitExceededError, DatabaseError)

    @pytest.mark.parametrize(
        "exc_class",
        [FetchError, ParseError, ChunkingError, EmbeddingError, SearchError],
    )
    def test_subclass_preserves_message_and_details(self, exc_class):
        """Subclasses should inherit the message+details formatting."""
        exc = exc_class("msg", details="dtl")
        assert exc.message == "msg"
        assert exc.details == "dtl"
        assert str(exc) == "msg — dtl"


class TestFilingLimitExceededError:
    """This exception has a custom __init__ with count/limit attributes."""

    def test_attributes(self):
        exc = FilingLimitExceededError(current_count=15, max_filings=20)
        assert exc.current_count == 15
        assert exc.max_filings == 20

    def test_message_includes_counts(self):
        exc = FilingLimitExceededError(current_count=20, max_filings=20)
        assert "20/20" in exc.message

    def test_message_mentions_db_max_filings(self):
        """The message should hint at the config knob."""
        exc = FilingLimitExceededError(current_count=5, max_filings=5)
        assert "DB_MAX_FILINGS" in str(exc)

    def test_optional_details(self):
        exc = FilingLimitExceededError(5, 5, details="extra context")
        assert exc.details == "extra context"

    def test_catchable_as_database_error(self):
        """CLI catches DatabaseError, so this must be catchable that way."""
        with pytest.raises(DatabaseError):
            raise FilingLimitExceededError(10, 10)
