"""
Tests for query and document task prompts in ``EmbeddingGenerator`` (F-08).

Covers:
    - Chunks are encoded with ``encode_document``; queries with ``encode_query``
    - A real ``SentenceTransformer`` prepends the model's configured prompts
      (the strings documented for ``google/embeddinggemma-300m``)
    - A model without prompts encodes text unchanged
    - Query text never reaches the embedder's log output
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from sentence_transformers import SentenceTransformer
from torch import nn

from sec_semantic_search.config.constants import EMBEDDING_DIMENSION
from sec_semantic_search.pipeline.embed import EmbeddingGenerator

# The prompts ``google/embeddinggemma-300m`` documents for retrieval.
GEMMA_PROMPTS = {
    "query": "task: search result | query: ",
    "document": "title: none | text: ",
}


def _vectors(texts, **kwargs):
    return np.zeros((len(texts), EMBEDDING_DIMENSION), dtype=np.float32)


@pytest.fixture
def mock_generator():
    gen = EmbeddingGenerator(batch_size=8)
    gen._model = MagicMock()
    gen._model.encode_document.side_effect = _vectors
    gen._model.encode_query.side_effect = _vectors
    return gen


class _Recorder(nn.Module):
    """A one-module SentenceTransformer body that records what it tokenizes."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[str] = []
        self.anchor = nn.Parameter(torch.zeros(1))

    def tokenize(self, texts, **kwargs):
        self.seen.extend(texts)
        return {"n": torch.arange(len(texts))}

    def forward(self, features):
        features["sentence_embedding"] = torch.ones(len(features["n"]), EMBEDDING_DIMENSION)
        return features

    def get_sentence_embedding_dimension(self) -> int:
        return EMBEDDING_DIMENSION


def _real_generator(prompts: dict[str, str] | None) -> tuple[EmbeddingGenerator, _Recorder]:
    recorder = _Recorder()
    model = SentenceTransformer(modules=[recorder], device="cpu", prompts=prompts)
    gen = EmbeddingGenerator(device="cpu")
    gen._model = model
    return gen, recorder


class TestMethodSelection:
    """Each path uses the method that applies its task prompt."""

    def test_chunks_are_encoded_as_documents(self, mock_generator, sample_chunks):
        mock_generator.embed_chunks(sample_chunks, show_progress=False)
        model = mock_generator._model
        model.encode_document.assert_called_once()
        model.encode_query.assert_not_called()
        model.encode.assert_not_called()
        kwargs = model.encode_document.call_args.kwargs
        assert kwargs == {"batch_size": 8, "show_progress_bar": False, "convert_to_numpy": True}

    def test_embed_texts_encodes_documents(self, mock_generator):
        mock_generator.embed_texts(["a", "b"], show_progress=False)
        mock_generator._model.encode_document.assert_called_once()
        mock_generator._model.encode_query.assert_not_called()

    def test_query_is_encoded_as_query(self, mock_generator):
        mock_generator.embed_query("risk factors")
        model = mock_generator._model
        model.encode_query.assert_called_once()
        assert model.encode_query.call_args.args[0] == ["risk factors"]
        model.encode_document.assert_not_called()
        model.encode.assert_not_called()

    def test_chromadb_helper_encodes_query(self, mock_generator):
        mock_generator.embed_query_for_chromadb("risk factors", lock_timeout=1.0)
        mock_generator._model.encode_query.assert_called_once()
        mock_generator._model.encode_document.assert_not_called()


class TestPromptsWithRealSentenceTransformer:
    """The installed sentence-transformers applies the model's prompts."""

    def test_document_prompt_prepended_to_chunks(self):
        gen, recorder = _real_generator(GEMMA_PROMPTS)
        gen.embed_texts(["Revenue rose 8%."], show_progress=False)
        assert "title: none | text: Revenue rose 8%." in recorder.seen
        assert not any(t.startswith("task: search result") for t in recorder.seen)

    def test_query_prompt_prepended_to_query(self):
        gen, recorder = _real_generator(GEMMA_PROMPTS)
        vector = gen.embed_query("what drove revenue")
        assert "task: search result | query: what drove revenue" in recorder.seen
        assert not any(t.startswith("title: none") for t in recorder.seen)
        assert vector.shape == (EMBEDDING_DIMENSION,)

    def test_model_without_prompts_encodes_text_unchanged(self):
        gen, recorder = _real_generator(None)
        gen.embed_texts(["Revenue rose 8%."], show_progress=False)
        gen.embed_query("what drove revenue")
        assert recorder.seen == ["Revenue rose 8%.", "what drove revenue"]


class TestQueryPrivacy:
    """Search queries are never persisted, including in logs (AD#29)."""

    def test_query_text_not_logged(self, mock_generator):
        query = "confidential acquisition target"
        with patch("sec_semantic_search.pipeline.embed.logger") as log:
            mock_generator.embed_query(query)
        for call in log.method_calls:
            assert not any(query in str(arg) for arg in call.args)
