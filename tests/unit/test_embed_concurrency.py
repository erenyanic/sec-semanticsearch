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


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class TestIdleTimer:
    """One timer per idle period, not one per model access."""

    def test_repeated_access_creates_one_timer(self, generator):
        generator._idle_timeout_seconds = 60.0
        created: list[threading.Timer] = []
        real_timer = threading.Timer

        def counting_timer(*args, **kwargs):
            timer = real_timer(*args, **kwargs)
            created.append(timer)
            return timer

        try:
            with patch("sec_semantic_search.pipeline.embed.threading.Timer", counting_timer):
                for _ in range(50):
                    _ = generator.model
                    generator.embed_query("q")
            assert len(created) == 1
        finally:
            generator._cancel_idle_timer()

    def test_unloads_after_timeout_without_use(self, generator):
        generator._idle_timeout_seconds = 0.1
        _ = generator.model
        assert _wait_until(lambda: not generator.is_loaded)
        assert generator._idle_timer is None

    def test_use_postpones_unload(self, generator):
        generator._idle_timeout_seconds = 0.3
        start = time.monotonic()
        _ = generator.model
        time.sleep(0.2)
        generator.embed_query("q")  # last use at ~0.2 s: idle until ~0.5 s
        assert _wait_until(lambda: not generator.is_loaded, timeout=3)
        assert time.monotonic() - start >= 0.45

    def test_idle_counts_from_end_of_a_long_encode(self, generator):
        """A timer that fires during an encode must not unload right after it."""
        generator._idle_timeout_seconds = 0.2

        def slow(texts, **kwargs):
            time.sleep(0.35)  # longer than the timeout
            return _vectors(texts)

        generator._model.encode_document.side_effect = slow
        generator.embed_texts(["t"], show_progress=False)
        ended = time.monotonic()
        # The timer fired mid-encode, waited for the lock, then re-armed.
        time.sleep(0.05)
        assert generator.is_loaded
        assert _wait_until(lambda: not generator.is_loaded, timeout=3)
        assert time.monotonic() - ended >= 0.15

    def test_stale_timer_does_not_unload(self, generator):
        """A callback from a timer that is no longer armed is ignored."""
        generator._idle_timeout_seconds = 60.0
        _ = generator.model
        armed = generator._idle_timer
        try:
            stale = threading.Thread(target=generator._on_idle_timeout)
            stale.start()
            stale.join(timeout=2)
            assert generator.is_loaded
            assert generator._idle_timer is armed
        finally:
            generator._cancel_idle_timer()

    def test_timer_cancelled_by_unload_is_ignored_after_reload(self, generator):
        generator._idle_timeout_seconds = 60.0
        _ = generator.model
        old = generator._idle_timer
        model = generator._model
        generator.unload()
        assert generator._idle_timer is None
        generator._model = model  # reload
        generator._touch()
        new = generator._idle_timer
        try:
            assert new is not None and new is not old
            # The old timer's callback, had it already fired, must not act.
            stale = threading.Thread(target=generator._on_idle_timeout)
            stale.start()
            stale.join(timeout=2)
            assert generator.is_loaded
            assert generator._idle_timer is new
        finally:
            generator._cancel_idle_timer()

    def test_no_timer_when_timeout_disabled(self, generator):
        generator._idle_timeout_seconds = 0
        _ = generator.model
        generator.embed_query("q")
        assert generator._idle_timer is None

    def test_failed_load_arms_no_timer(self):
        gen = EmbeddingGenerator()
        gen._idle_timeout_seconds = 60.0
        with patch.object(EmbeddingGenerator, "_load_model", side_effect=EmbeddingError("x")):
            with pytest.raises(EmbeddingError):
                gen.embed_query("q")
        assert gen._idle_timer is None
