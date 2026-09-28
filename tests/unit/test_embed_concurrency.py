"""
Concurrency tests for ``EmbeddingGenerator`` (OPTIMIZATIONS.md F-01, F-13).

API searches now run in threadpool workers, so the generator can be used
from several threads at once. The lock must guarantee one model load, one
encode at a time, an unload that waits for in-flight encodes, a bounded
wait for callers that ask for one, and no unload from a stale idle timer.
"""

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sec_semantic_search.config.constants import EMBEDDING_DIMENSION
from sec_semantic_search.core.exceptions import EmbeddingBusyError, EmbeddingError
from sec_semantic_search.pipeline.embed import EmbeddingGenerator


def _vectors(texts, **kwargs):
    return np.zeros((len(texts), EMBEDDING_DIMENSION), dtype=np.float32)


@pytest.fixture
def generator():
    gen = EmbeddingGenerator()
    gen._model = MagicMock()
    gen._model.encode_query.side_effect = _vectors
    gen._model.encode_document.side_effect = _vectors
    return gen


def _hold_lock(gen: EmbeddingGenerator, release: threading.Event) -> threading.Thread:
    """Start a thread that holds the generator lock until *release* is set."""
    acquired = threading.Event()

    def _run():
        with gen._lock:
            acquired.set()
            release.wait(timeout=5)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    assert acquired.wait(timeout=5)
    return thread


class TestSingleLoad:
    def test_concurrent_first_access_loads_once(self):
        gen = EmbeddingGenerator()
        loads: list[int] = []

        def slow_load():
            loads.append(1)
            time.sleep(0.05)
            return MagicMock()

        with patch.object(EmbeddingGenerator, "_load_model", side_effect=slow_load):
            threads = [threading.Thread(target=lambda: gen.model) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)

        assert len(loads) == 1
        assert gen.is_loaded


class TestSerializedEncode:
    def test_encodes_never_overlap(self, generator):
        active = 0
        peak = 0
        counter_lock = threading.Lock()

        def tracking_encode(texts, **kwargs):
            nonlocal active, peak
            with counter_lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.01)
            with counter_lock:
                active -= 1
            return _vectors(texts)

        generator._model.encode_query.side_effect = tracking_encode
        generator._model.encode_document.side_effect = tracking_encode
        # Searches (queries) and an ingest (documents) share one lock.
        threads = [
            threading.Thread(target=generator.embed_query, args=(f"q{i}",)) for i in range(4)
        ] + [
            threading.Thread(target=generator.embed_texts, args=([f"d{i}"], False))
            for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert generator._model.encode_query.call_count == 4
        assert generator._model.encode_document.call_count == 4
        assert peak == 1

    def test_unload_waits_for_in_flight_encode(self, generator):
        encoding = threading.Event()
        release = threading.Event()

        def blocking_encode(texts, **kwargs):
            encoding.set()
            release.wait(timeout=5)
            return _vectors(texts)

        generator._model.encode_query.side_effect = blocking_encode
        encoder = threading.Thread(target=generator.embed_query, args=("q",))
        encoder.start()
        assert encoding.wait(timeout=5)

        unloader = threading.Thread(target=generator.unload)
        unloader.start()
        unloader.join(timeout=0.2)
        assert unloader.is_alive(), "unload() did not wait for the encode"
        assert generator.is_loaded

        release.set()
        encoder.join(timeout=5)
        unloader.join(timeout=5)
        assert not generator.is_loaded


class TestBoundedWait:
    def test_timeout_raises_busy_error(self, generator):
        release = threading.Event()
        holder = _hold_lock(generator, release)
        try:
            start = time.monotonic()
            with pytest.raises(EmbeddingBusyError):
                generator.embed_query("q", lock_timeout=0.05)
            assert time.monotonic() - start < 2
        finally:
            release.set()
            holder.join(timeout=5)

        generator._model.encode_query.assert_not_called()
        generator._model.encode_document.assert_not_called()

    def test_busy_error_is_an_embedding_error(self):
        assert issubclass(EmbeddingBusyError, EmbeddingError)

    def test_timeout_passes_through_chromadb_helper(self, generator):
        release = threading.Event()
        holder = _hold_lock(generator, release)
        try:
            with pytest.raises(EmbeddingBusyError):
                generator.embed_query_for_chromadb("q", lock_timeout=0.05)
        finally:
            release.set()
            holder.join(timeout=5)

    def test_default_waits_for_lock(self, generator):
        """Without a timeout (CLI, ingest) the caller waits rather than failing."""
        release = threading.Event()
        holder = _hold_lock(generator, release)
        result: list[np.ndarray] = []
        waiter = threading.Thread(target=lambda: result.append(generator.embed_query("q")))
        waiter.start()
        waiter.join(timeout=0.1)
        assert waiter.is_alive()

        release.set()
        holder.join(timeout=5)
        waiter.join(timeout=5)
        assert result and result[0].shape == (EMBEDDING_DIMENSION,)

    def test_lock_released_after_encode_failure(self, generator):
        generator._model.encode_query.side_effect = RuntimeError("CUDA OOM")
        with pytest.raises(EmbeddingError):
            generator.embed_query("q")

        generator._model.encode_query.side_effect = _vectors
        assert generator.embed_query("q", lock_timeout=0.05).shape == (EMBEDDING_DIMENSION,)


class TestIdleTimerGeneration:
    def test_stale_timer_does_not_unload(self, generator):
        """A timer that fired during an encode is superseded by the encode's re-arm."""
        generator._idle_generation = 2
        generator._on_idle_timeout(1)
        assert generator.is_loaded

    def test_current_timer_unloads(self, generator):
        generator._idle_generation = 2
        generator._on_idle_timeout(2)
        assert not generator.is_loaded

    def test_model_access_advances_generation(self, generator):
        generator._idle_timeout_seconds = 60.0
        try:
            before = generator._idle_generation
            _ = generator.model
            _ = generator.model
            assert generator._idle_generation == before + 2
        finally:
            generator._cancel_idle_timer()

    def test_no_timer_when_timeout_disabled(self, generator):
        generator._idle_timeout_seconds = 0
        _ = generator.model
        assert generator._idle_timer is None
        assert generator._idle_generation == 0
