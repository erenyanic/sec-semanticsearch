"""
Tests for the EmbeddingGenerator pipeline component.

Loading the real 300M-parameter model takes ~10s and requires CUDA.
We mock SentenceTransformer to test the generator's own logic:
device detection, lazy loading, input validation, error wrapping,
and the embed_query_for_chromadb() format conversion.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from sec_semantic_search.config.constants import EMBEDDING_DIMENSION
from sec_semantic_search.core.exceptions import EmbeddingError
from sec_semantic_search.pipeline.embed import EmbeddingGenerator


@pytest.fixture
def mock_model():
    """A mock SentenceTransformer that returns correctly shaped arrays."""
    model = MagicMock()

    def fake_encode(texts, **kwargs):
        return np.random.default_rng(42).random((len(texts), EMBEDDING_DIMENSION), dtype=np.float32)

    model.encode_document.side_effect = fake_encode
    model.encode_query.side_effect = fake_encode
    return model


@pytest.fixture
def generator(mock_model):
    """An EmbeddingGenerator with the real model replaced by a mock."""
    gen = EmbeddingGenerator()
    gen._model = mock_model
    return gen


# -----------------------------------------------------------------------
# Device detection
# -----------------------------------------------------------------------


class TestDeviceDetection:
    """The device property resolves 'auto' to 'cuda' or 'cpu'."""

    def test_auto_resolves_to_string(self):
        gen = EmbeddingGenerator()
        assert gen.device in ("cuda", "cpu")

    def test_explicit_cpu(self):
        gen = EmbeddingGenerator(device="cpu")
        assert gen.device == "cpu"

    def test_explicit_cuda(self):
        gen = EmbeddingGenerator(device="cuda")
        assert gen.device == "cuda"


# -----------------------------------------------------------------------
# BF16 quantisation
# -----------------------------------------------------------------------


class TestBF16Quantisation:
    """BF16 quantisation is applied automatically on CUDA devices."""

    def test_bf16_applied_on_cuda(self):
        """Verify torch_dtype=torch.bfloat16 is passed when device is CUDA."""
        gen = EmbeddingGenerator(device="cuda")

        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            _ = gen._load_model()

            # Verify SentenceTransformer was called with model_kwargs containing torch_dtype
            call_kwargs = mock_st.call_args[1]
            assert "model_kwargs" in call_kwargs
            assert "torch_dtype" in call_kwargs["model_kwargs"]
            assert call_kwargs["model_kwargs"]["torch_dtype"] == torch.bfloat16

    def test_bf16_not_applied_on_cpu(self):
        """Verify torch_dtype is NOT passed when device is CPU."""
        gen = EmbeddingGenerator(device="cpu")

        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            _ = gen._load_model()

            # Verify SentenceTransformer was called without torch_dtype on CPU
            call_kwargs = mock_st.call_args[1]
            assert "model_kwargs" in call_kwargs
            # model_kwargs should be empty dict when device is CPU
            assert call_kwargs["model_kwargs"] == {}

    def test_cuda_device_passed_correctly(self):
        """Verify device parameter is passed to SentenceTransformer."""
        gen = EmbeddingGenerator(device="cuda")

        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            _ = gen._load_model()

            call_kwargs = mock_st.call_args[1]
            assert call_kwargs["device"] == "cuda"

    def test_cpu_device_passed_correctly(self):
        """Verify device parameter is passed to SentenceTransformer."""
        gen = EmbeddingGenerator(device="cpu")

        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            _ = gen._load_model()

            call_kwargs = mock_st.call_args[1]
            assert call_kwargs["device"] == "cpu"


# -----------------------------------------------------------------------
# Lazy loading
# -----------------------------------------------------------------------


class TestLazyLoading:
    """The model should only be loaded on first access."""

    def test_model_none_initially(self):
        gen = EmbeddingGenerator()
        assert gen._model is None

    def test_model_property_triggers_load(self, mock_model):
        gen = EmbeddingGenerator()
        gen._model = None

        with patch(
            "sec_semantic_search.pipeline.embed.EmbeddingGenerator._load_model",
            return_value=mock_model,
        ):
            model = gen.model

        assert model is mock_model

    def test_model_cached_after_first_access(self, generator, mock_model):
        """Accessing .model twice should not reload."""
        _ = generator.model
        _ = generator.model
        # _model was set directly, so _load_model should never be called
        assert generator._model is mock_model


# -----------------------------------------------------------------------
# embed_texts
# -----------------------------------------------------------------------


class TestEmbedTexts:
    """embed_texts() is the core encoding method."""

    def test_returns_correct_shape(self, generator):
        texts = ["hello world", "test sentence"]
        result = generator.embed_texts(texts, show_progress=False)
        assert result.shape == (2, EMBEDDING_DIMENSION)

    def test_empty_texts_raises(self, generator):
        with pytest.raises(EmbeddingError, match="No texts to embed"):
            generator.embed_texts([], show_progress=False)

    def test_returns_numpy_array(self, generator):
        result = generator.embed_texts(["test"], show_progress=False)
        assert isinstance(result, np.ndarray)


# -----------------------------------------------------------------------
# embed_chunks
# -----------------------------------------------------------------------


class TestEmbedChunks:
    """embed_chunks() extracts text from Chunk objects."""

    def test_returns_correct_shape(self, generator, sample_chunks):
        result = generator.embed_chunks(sample_chunks, show_progress=False)
        assert result.shape == (len(sample_chunks), EMBEDDING_DIMENSION)

    def test_empty_chunks_raises(self, generator):
        with pytest.raises(EmbeddingError, match="No chunks to embed"):
            generator.embed_chunks([], show_progress=False)


# -----------------------------------------------------------------------
# embed_query and embed_query_for_chromadb
# -----------------------------------------------------------------------


class TestEmbedQuery:
    """embed_query() returns a 1D array for a single query."""

    def test_returns_1d_array(self, generator):
        result = generator.embed_query("test query")
        assert result.ndim == 1
        assert result.shape == (EMBEDDING_DIMENSION,)

    def test_empty_query_raises(self, generator):
        with pytest.raises(EmbeddingError, match="Empty query"):
            generator.embed_query("")

    def test_whitespace_query_raises(self, generator):
        with pytest.raises(EmbeddingError, match="Empty query"):
            generator.embed_query("   \t  ")


class TestEmbedQueryForChromadb:
    """embed_query_for_chromadb() returns the list[list[float]] format."""

    def test_returns_nested_list(self, generator):
        result = generator.embed_query_for_chromadb("test query")
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], list)
        assert len(result[0]) == EMBEDDING_DIMENSION

    def test_values_are_floats(self, generator):
        result = generator.embed_query_for_chromadb("test")
        assert all(isinstance(v, float) for v in result[0])


# -----------------------------------------------------------------------
# Error wrapping
# -----------------------------------------------------------------------


class TestErrorWrapping:
    """Model errors should be wrapped as EmbeddingError."""

    def test_load_model_failure(self):
        gen = EmbeddingGenerator()
        # SentenceTransformer is imported inside _load_model() (lazy),
        # so we patch it at its source module.
        with patch(
            "sentence_transformers.SentenceTransformer",
            side_effect=Exception("CUDA out of memory"),
        ):
            with pytest.raises(EmbeddingError, match="Failed to load model"):
                gen._load_model()

    def test_encode_failure(self, generator, mock_model):
        mock_model.encode_document.side_effect = RuntimeError("Encoding failed")
        with pytest.raises(EmbeddingError, match="Failed to generate embeddings"):
            generator.embed_texts(["test"], show_progress=False)
