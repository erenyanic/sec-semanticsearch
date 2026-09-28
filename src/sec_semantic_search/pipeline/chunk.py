"""
Text chunking for SEC filings.

This module splits long segments into smaller chunks suitable for embedding.
It uses sentence-boundary splitting to ensure chunks don't cut mid-sentence.

Usage:
    from sec_semantic_search.pipeline import TextChunker

    chunker = TextChunker()
    chunks = chunker.chunk_segments(segments)
"""

import re
from typing import NamedTuple

from sec_semantic_search.config import get_settings
from sec_semantic_search.core import Chunk, ChunkingError, Segment, get_logger

logger = get_logger(__name__)


class _Unit(NamedTuple):
    """A sentence (or fallback piece) and the whitespace that preceded it."""

    sep: str
    text: str
    tokens: int


class TextChunker:
    """
    Splits segments into embedding-ready chunks.

    This class implements sentence-boundary aware chunking to ensure
    that text is split at natural boundaries rather than mid-sentence.

    The chunking algorithm:
        1. If segment fits within token limit, keep as-is
        2. Otherwise, split on sentence boundaries (. ! ?). A "sentence" longer
           than ``token_limit + tolerance`` has no boundary to split on — in
           practice a table, whose rows are joined by line breaks — so it is
           split on line breaks instead, and any row still over the cap into
           windows of ``token_limit`` words.
        3. Accumulate sentences targeting ``token_limit ± tolerance`` — finalise
           early once the running chunk has reached ``token_limit - tolerance``,
           and never overflow past ``token_limit + tolerance``.
        4. Seed each subsequent chunk with trailing whole sentences from the
           previous one, up to the configured overlap budget. If even the last
           sentence exceeds the budget, that single sentence is reused whole
           (chunks never start mid-sentence) — unless it would push the new
           chunk past ``token_limit + tolerance``, in which case it is dropped.

    No chunk exceeds ``token_limit + tolerance``. Chunks keep the original
    whitespace between sentences, so each chunk is an exact substring of its
    segment and the UI can highlight it inside the parent text.

    Attributes:
        token_limit: Maximum tokens per chunk (from settings)
        tolerance: Acceptable ± deviation around ``token_limit`` (from settings)
        overlap: Token budget reused at the start of each subsequent chunk

    Example:
        >>> chunker = TextChunker()
        >>> chunks = chunker.chunk_segments(segments)
        >>> print(f"Created {len(chunks)} chunks")
    """

    # Sentence boundary pattern: split after . ! ? followed by whitespace
    SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")

    # The same boundaries captured, and the fallback boundaries for an
    # over-long sentence: line breaks (table rows), then words.
    _SENTENCE_SPLIT = re.compile(r"((?<=[.!?])\s+)")
    _LINE_SPLIT = re.compile(r"(\s*\n\s*)")
    _WORD = re.compile(r"\S+")

    def __init__(
        self,
        token_limit: int | None = None,
        tolerance: int | None = None,
        overlap: int | None = None,
    ) -> None:
        """
        Initialise the chunker with configurable limits.

        Args:
            token_limit: Max tokens per chunk. If None, uses settings.
            tolerance: Acceptable ± deviation around ``token_limit``. If None, uses settings.
            overlap: Token budget reused at the start of each subsequent chunk.
                If None, uses settings. Pass ``0`` to disable overlap.
        """
        settings = get_settings()
        self.token_limit = token_limit if token_limit is not None else settings.chunking.token_limit
        self.tolerance = tolerance if tolerance is not None else settings.chunking.tolerance
        self.overlap = overlap if overlap is not None else settings.chunking.overlap

        if not 0 <= self.tolerance < self.token_limit:
            raise ValueError(
                f"tolerance ({self.tolerance}) must be ≥ 0 and smaller than "
                f"token_limit ({self.token_limit})"
            )
        if self.overlap < 0:
            raise ValueError("overlap must be ≥ 0")
        # A larger overlap turns nearly every sentence into a chunk boundary
        # and multiplies embedding work and storage per filing.
        if self.overlap > self.token_limit // 2:
            raise ValueError(
                f"overlap ({self.overlap}) must be at most half of token_limit ({self.token_limit})"
            )

        logger.debug(
            "TextChunker initialised: limit=%d, tolerance=±%d, overlap=%d",
            self.token_limit,
            self.tolerance,
            self.overlap,
        )

    def _count_tokens(self, text: str) -> int:
        """
        Approximate token count using whitespace splitting.

        This is a simple heuristic that works well for English text.
        More accurate tokenisation would require the actual model's
        tokeniser, but whitespace splitting is sufficient for chunking.

        Args:
            text: Text to count tokens in.

        Returns:
            Approximate token count.
        """
        return len(text.split())

    def _chunk_text(self, text: str) -> list[tuple[str, int]]:
        """
        Split text into chunks respecting sentence boundaries.

        Targets a chunk size of ``token_limit ± tolerance``: finalise as soon as
        the running chunk has reached ``token_limit - tolerance`` (early-stop),
        and never let it grow past ``token_limit + tolerance`` (hard cap). Each
        subsequent chunk is seeded with trailing whole sentences from the
        previous one, up to ``self.overlap`` tokens, as long as the seed and the
        next sentence fit under the cap.

        Args:
            text: Text content to split.

        Returns:
            List of (chunk_text, token_count) tuples.
        """
        total_tokens = self._count_tokens(text)

        # If text already fits, return as single chunk
        if total_tokens <= self.token_limit:
            return [(text, total_tokens)]

        upper = self.token_limit + self.tolerance
        lower = self.token_limit - self.tolerance

        chunks: list[tuple[str, int]] = []
        current: list[_Unit] = []
        current_tokens = 0
        # Tracks the size of the overlap prefix carried over from the previous
        # chunk so the final flush can skip emitting an overlap-only tail.
        overlap_prefix_len = 0

        for unit in self._split_units(text):
            # Only finalise once the running chunk contains at least one
            # sentence beyond the overlap carried over from the previous chunk
            # — otherwise we'd emit a duplicate of the previous chunk's tail.
            has_new_content = len(current) > overlap_prefix_len
            # Hard cap: adding this sentence would overshoot ``limit + tolerance``.
            # Early-stop: running chunk already inside the ± band; finalise at
            # the previous sentence boundary rather than overshoot the target.
            if has_new_content and (
                current_tokens + unit.tokens > upper or current_tokens >= lower
            ):
                chunks.append((self._join(current), current_tokens))
                current, current_tokens = self._build_overlap(current)
                overlap_prefix_len = len(current)

            # The carried-over overlap never pushes a chunk past the cap: drop
            # its oldest sentences until this one fits. Only the overlap can be
            # dropped here — new content was finalised above if it did not fit.
            while overlap_prefix_len and current_tokens + unit.tokens > upper:
                current_tokens -= current.pop(0).tokens
                overlap_prefix_len -= 1

            current.append(unit)
            current_tokens += unit.tokens

        # Flush remaining sentences. Skip overlap-only tails — they would
        # duplicate the previous chunk entirely without contributing new text.
        if current and len(current) > overlap_prefix_len:
            chunks.append((self._join(current), current_tokens))

        return chunks

    def _split_units(self, text: str) -> list[_Unit]:
        """
        Split text into sentences, each no longer than ``token_limit + tolerance``.

        A sentence over the cap is split on line breaks (table rows), and a
        row still over the cap into windows of ``token_limit`` words. Each
        unit keeps the whitespace before it, so joining units reproduces the
        text exactly.
        """
        upper = self.token_limit + self.tolerance
        units: list[_Unit] = []
        for sep, sentence in self._split_keeping_separators(text, self._SENTENCE_SPLIT, ""):
            tokens = self._count_tokens(sentence)
            if tokens <= upper:
                units.append(_Unit(sep, sentence, tokens))
                continue
            for row_sep, row in self._split_keeping_separators(sentence, self._LINE_SPLIT, sep):
                row_tokens = self._count_tokens(row)
                if row_tokens <= upper:
                    units.append(_Unit(row_sep, row, row_tokens))
                else:
                    units.extend(self._word_windows(row_sep, row))
        return units

    @staticmethod
    def _split_keeping_separators(
        text: str,
        pattern: re.Pattern[str],
        lead: str,
    ) -> list[tuple[str, str]]:
        """
        Split ``text`` on ``pattern`` (one capturing group) into
        ``(separator, piece)`` pairs, ``lead`` being the separator before the
        first piece. Concatenating the pairs reproduces ``lead + text``, minus
        any trailing whitespace.
        """
        pairs: list[tuple[str, str]] = []
        pending = lead
        for i, part in enumerate(pattern.split(text)):
            if i % 2 or not part:
                pending += part
                continue
            pairs.append((pending, part))
            pending = ""
        return pairs

    def _word_windows(self, sep: str, row: str) -> list[_Unit]:
        """Split a row with no usable boundary into windows of ``token_limit`` words."""
        spans = [m.span() for m in self._WORD.finditer(row)]
        units: list[_Unit] = []
        previous_end = 0
        for i in range(0, len(spans), self.token_limit):
            window = spans[i : i + self.token_limit]
            start, end = window[0][0], window[-1][1]
            lead = sep if i == 0 else ""
            units.append(_Unit(lead + row[previous_end:start], row[start:end], len(window)))
            previous_end = end
        return units

    @staticmethod
    def _join(units: list[_Unit]) -> str:
        """Rejoin units with their original separators."""
        return units[0].text + "".join(u.sep + u.text for u in units[1:])

    def _build_overlap(self, units: list[_Unit]) -> tuple[list[_Unit], int]:
        """
        Select trailing whole sentences from a just-finalised chunk to seed the
        next one. Walks backwards accumulating sentences while the total stays
        within ``self.overlap``. If even the single last sentence exceeds the
        budget, it is still reused whole — sentence boundaries are never split.
        """
        if self.overlap == 0 or not units:
            return [], 0

        overlap_units: list[_Unit] = []
        overlap_tokens = 0
        for unit in reversed(units):
            if overlap_units and overlap_tokens + unit.tokens > self.overlap:
                break
            overlap_units.insert(0, unit)
            overlap_tokens += unit.tokens
            if overlap_tokens >= self.overlap:
                break

        return overlap_units, overlap_tokens

    def chunk_segment(self, segment: Segment, start_index: int = 0) -> list[Chunk]:
        """
        Split a single segment into chunks.

        Args:
            segment: Segment to chunk.
            start_index: Starting chunk index for this segment.

        Returns:
            List of Chunk objects with sequential indices.
        """
        text_chunks = self._chunk_text(segment.content)

        return [
            Chunk(
                content=text,
                path=segment.path,
                content_type=segment.content_type,
                filing_id=segment.filing_id,
                chunk_index=start_index + i,
                token_count=tokens,
                segment_index=segment.segment_index,
            )
            for i, (text, tokens) in enumerate(text_chunks)
        ]

    def chunk_segments(self, segments: list[Segment]) -> list[Chunk]:
        """
        Chunk all segments from a filing.

        This is the main entry point for chunking. It processes all
        segments and assigns sequential chunk indices across the
        entire filing.

        Args:
            segments: List of segments from FilingParser.

        Returns:
            List of Chunk objects ready for embedding.

        Raises:
            ChunkingError: If segments list is empty.

        Example:
            >>> chunks = chunker.chunk_segments(segments)
            >>> for chunk in chunks[:3]:
            ...     print(f"[{chunk.chunk_index}] {chunk.path[:50]}...")
        """
        if not segments:
            raise ChunkingError(
                "No segments to chunk",
                details="Received empty segments list.",
            )

        filing_id = segments[0].filing_id

        logger.info(
            "Chunking %d segments from %s %s",
            len(segments),
            filing_id.ticker,
            filing_id.form_type,
        )

        chunks: list[Chunk] = []
        current_index = 0

        for segment in segments:
            segment_chunks = self.chunk_segment(segment, start_index=current_index)
            chunks.extend(segment_chunks)
            current_index += len(segment_chunks)

        # Log statistics — token counts are retained from chunking, no recount
        token_counts = [c.token_count for c in chunks]
        min_tokens = min(token_counts)
        max_tokens = max(token_counts)
        avg_tokens = sum(token_counts) / len(token_counts)
        # Chunks between the limit and the cap are normal (the ± band); only
        # chunks past the cap would mean the fallback split failed.
        cap = self.token_limit + self.tolerance
        over_cap = sum(1 for t in token_counts if t > cap)

        logger.info(
            "Created %d chunks from %d segments (tokens: %d-%d, avg %.0f, %d over the %d-token cap)",
            len(chunks),
            len(segments),
            min_tokens,
            max_tokens,
            avg_tokens,
            over_cap,
            cap,
        )

        return chunks
