"""
Tests for the chunker's handling of segments without sentence boundaries (F-07).

Covers:
    - Tables (rows joined by line breaks) split at row boundaries
    - Terminator-less text without line breaks split into word windows
    - No chunk ever exceeds ``token_limit + tolerance``, overlap included
    - Chunks are exact substrings of their segment (original whitespace kept)
    - Chunks, taken in order, cover every word of the segment
    - Incoherent limit / tolerance / overlap combinations are rejected
"""

import random

import pytest
from pydantic import ValidationError

from sec_semantic_search.config.settings import ChunkingSettings
from sec_semantic_search.core.types import ContentType, Segment
from sec_semantic_search.pipeline.chunk import TextChunker

LIMIT, TOLERANCE, OVERLAP = 20, 5, 5
CAP = LIMIT + TOLERANCE


@pytest.fixture
def chunker() -> TextChunker:
    return TextChunker(token_limit=LIMIT, tolerance=TOLERANCE, overlap=OVERLAP)


def _segment(text: str, sample_filing_id, content_type=ContentType.TABLE) -> Segment:
    return Segment(
        path="Part II > Item 8", content_type=content_type, content=text, filing_id=sample_filing_id
    )


def _table(rows: int, cells: int = 6, start: int = 0) -> str:
    """Rows of numeric cells joined by newlines; no sentence terminators."""
    return "\n".join(
        " ".join(f"r{r}c{c}:{r * 1000 + c:,}" for c in range(cells))
        for r in range(start, start + rows)
    )


def _assert_invariants(chunks, text: str) -> None:
    """Cap, exactness and coverage — the properties every split must keep."""
    position = 0
    covered_until = 0
    for chunk in chunks:
        assert chunk.token_count <= CAP, f"{chunk.token_count} tokens > cap {CAP}"
        assert chunk.token_count == len(chunk.content.split())
        start = text.find(chunk.content, position)
        assert start >= 0, "chunk is not a verbatim substring of its segment"
        # Every word between the previous chunk's end and this chunk's start
        # would be lost from the index.
        assert text[covered_until:start].strip() == ""
        covered_until = max(covered_until, start + len(chunk.content))
        position = start
    assert text[covered_until:].strip() == ""


class TestTables:
    """Oversized table segments split at row boundaries."""

    def test_large_table_is_split_under_the_cap(self, chunker, sample_filing_id):
        text = _table(rows=60)  # 360 words, no terminators
        chunks = chunker.chunk_segment(_segment(text, sample_filing_id))
        assert len(chunks) > 1
        _assert_invariants(chunks, text)

    def test_rows_are_never_cut(self, chunker, sample_filing_id):
        text = _table(rows=60)
        rows = set(text.split("\n"))
        for chunk in chunker.chunk_segment(_segment(text, sample_filing_id)):
            for line in chunk.content.split("\n"):
                assert line in rows

    def test_rows_keep_their_line_breaks(self, chunker, sample_filing_id):
        text = _table(rows=60)
        chunks = chunker.chunk_segment(_segment(text, sample_filing_id))
        assert all("\n" in c.content for c in chunks[:-1])

    def test_next_chunk_repeats_trailing_rows_within_budget(self, sample_filing_id):
        chunker = TextChunker(token_limit=20, tolerance=5, overlap=6)
        text = _table(rows=40, cells=3)  # 3-word rows fit the 6-token budget
        chunks = chunker.chunk_segment(_segment(text, sample_filing_id))
        for prev, curr in zip(chunks[:-1], chunks[1:], strict=True):
            assert curr.content.split("\n")[0] in prev.content.split("\n")

    def test_2000_word_terminator_less_segment(self, sample_filing_id):
        """The audit's regression case, at production limits."""
        chunker = TextChunker(token_limit=500, tolerance=50, overlap=50)
        text = _table(rows=250, cells=8)
        chunks = chunker.chunk_segment(_segment(text, sample_filing_id))
        assert max(c.token_count for c in chunks) <= 550
        assert len(chunks) >= 4


class TestWordWindows:
    """A single line with no boundary at all falls back to word windows."""

    def test_line_without_breaks_is_split_into_windows(self, chunker, sample_filing_id):
        text = " ".join(f"w{i}" for i in range(137))
        chunks = chunker.chunk_segment(_segment(text, sample_filing_id, ContentType.TEXT))
        _assert_invariants(chunks, text)
        assert max(c.token_count for c in chunks) <= CAP

    def test_oversized_row_inside_a_table(self, chunker, sample_filing_id):
        long_row = " ".join(f"x{i}" for i in range(70))
        text = f"{_table(rows=3)}\n{long_row}\n{_table(rows=3, start=10)}"
        chunks = chunker.chunk_segment(_segment(text, sample_filing_id))
        _assert_invariants(chunks, text)


class TestOverlapCap:
    """Carried-over overlap is dropped rather than overflow the cap."""

    def test_whole_sentence_overlap_dropped_when_it_would_overflow(self, sample_filing_id):
        chunker = TextChunker(token_limit=20, tolerance=5, overlap=5)
        # Each 18-word sentence exceeds the overlap budget and would be reused
        # whole; with the next 18-word sentence that makes 36 > 25.
        sentences = [" ".join(f"s{n}w{i}" for i in range(17)) + " end." for n in range(4)]
        text = " ".join(sentences)
        chunks = chunker.chunk_segment(_segment(text, sample_filing_id, ContentType.TEXT))
        assert [c.token_count for c in chunks] == [18, 18, 18, 18]
        _assert_invariants(chunks, text)

    def test_overlap_kept_when_it_fits(self, sample_filing_id):
        chunker = TextChunker(token_limit=20, tolerance=5, overlap=5)
        text = " ".join(f"Sentence number {i} is short." for i in range(20))
        chunks = chunker.chunk_segment(_segment(text, sample_filing_id, ContentType.TEXT))
        for prev, curr in zip(chunks[:-1], chunks[1:], strict=True):
            last = TextChunker.SENTENCE_PATTERN.split(prev.content)[-1]
            assert curr.content.startswith(last)


class TestExactSubstrings:
    """Chunks keep the whitespace between sentences, so the UI can find them."""

    def test_newline_separated_sentences_stay_verbatim(self, chunker, sample_filing_id):
        text = "\n\n".join(f"Paragraph {i} opens here. It then closes here." for i in range(12))
        chunks = chunker.chunk_segment(_segment(text, sample_filing_id, ContentType.TEXT))
        assert len(chunks) > 1
        _assert_invariants(chunks, text)
        assert any("\n\n" in c.content for c in chunks)

    @pytest.mark.parametrize("seed", range(25))
    def test_random_mixed_segments_keep_every_invariant(self, chunker, sample_filing_id, seed):
        rng = random.Random(seed)
        word = iter(range(10**6))
        pieces = []
        for _ in range(rng.randint(5, 40)):
            n = rng.choice([1, 3, 8, 15, 30, 60])
            body = " ".join(f"t{next(word)}" for _ in range(n))
            pieces.append(body + rng.choice([".", "!", "?", "", ""]))
            pieces.append(rng.choice([" ", "  ", "\n", "\n\n", " \n "]))
        text = "".join(pieces).strip()
        chunks = chunker.chunk_segment(_segment(text, sample_filing_id))
        _assert_invariants(chunks, text)


class TestValidation:
    """Incoherent configurations are rejected at construction."""

    def test_tolerance_must_be_smaller_than_limit(self):
        with pytest.raises(ValueError, match="tolerance"):
            TextChunker(token_limit=20, tolerance=20, overlap=0)

    def test_negative_tolerance_rejected(self):
        with pytest.raises(ValueError, match="tolerance"):
            TextChunker(token_limit=20, tolerance=-1, overlap=0)

    def test_overlap_above_half_the_limit_rejected(self):
        with pytest.raises(ValueError, match="overlap"):
            TextChunker(token_limit=20, tolerance=5, overlap=11)

    def test_overlap_of_exactly_half_accepted(self):
        TextChunker(token_limit=20, tolerance=5, overlap=10)

    @pytest.mark.parametrize(
        ("env", "message"),
        [
            ({"CHUNKING_TOLERANCE": "500"}, "CHUNKING_TOLERANCE"),
            ({"CHUNKING_OVERLAP": "251"}, "CHUNKING_OVERLAP"),
            (
                {"CHUNKING_TOKEN_LIMIT": "0", "CHUNKING_TOLERANCE": "0", "CHUNKING_OVERLAP": "0"},
                "CHUNKING_TOKEN_LIMIT",
            ),
        ],
    )
    def test_settings_reject_incoherent_env(self, monkeypatch, env, message):
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        with pytest.raises(ValidationError, match=message):
            ChunkingSettings()

    def test_settings_defaults_are_valid(self, monkeypatch):
        for name in ("CHUNKING_TOKEN_LIMIT", "CHUNKING_TOLERANCE", "CHUNKING_OVERLAP"):
            monkeypatch.delenv(name, raising=False)
        s = ChunkingSettings()
        assert (s.token_limit, s.tolerance, s.overlap) == (500, 50, 50)
