"""Tests for gemini_embedder.py — all API calls mocked."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag_core.config import EMBEDDING_DIM


def _make_mock_response(n: int):
    """Create a mock embed_content response with n embeddings."""
    mock = MagicMock()
    embeddings = []
    for _ in range(n):
        emb = MagicMock()
        emb.values = np.random.randn(EMBEDDING_DIM).tolist()
        embeddings.append(emb)
    mock.embeddings = embeddings
    return mock


@pytest.fixture
def embedder(tmp_path: Path, monkeypatch):
    """Create GeminiEmbedder with mocked client and temp cache dir."""
    monkeypatch.setattr("rag_core.config.EMBEDDINGS_DIR", tmp_path)
    monkeypatch.setattr("rag_core.embeddings.gemini_embedder.EMBEDDINGS_DIR", tmp_path)

    with patch("rag_core.embeddings.gemini_embedder.genai") as mock_genai:
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        from rag_core.embeddings.gemini_embedder import GeminiEmbedder

        emb = GeminiEmbedder()
        emb._client = mock_client
        yield emb, mock_client


def test_embed_texts_basic(embedder):
    emb, mock_client = embedder
    mock_client.models.embed_content.return_value = _make_mock_response(3)

    texts = ["hello", "world", "test"]
    ids = ["t_001", "t_002", "t_003"]
    result = emb.embed_texts(texts, ids)

    assert len(result) == 3
    assert all(isinstance(v, np.ndarray) for v in result.values())
    assert all(v.shape == (EMBEDDING_DIM,) for v in result.values())
    mock_client.models.embed_content.assert_called_once()


def test_embed_texts_batching(embedder):
    """13 texts should produce 2 batches (10 + 3)."""
    emb, mock_client = embedder
    mock_client.models.embed_content.side_effect = [
        _make_mock_response(10),
        _make_mock_response(3),
    ]

    texts = [f"text_{i}" for i in range(13)]
    ids = [f"t_{i:03d}" for i in range(13)]
    result = emb.embed_texts(texts, ids)

    assert len(result) == 13
    assert mock_client.models.embed_content.call_count == 2


def test_embed_texts_cache_hit(embedder, tmp_path: Path):
    """Cached embeddings should be loaded from disk, not from API."""
    emb, mock_client = embedder

    # Pre-cache one embedding
    cached_vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    np.save(tmp_path / "t_001.npy", cached_vec)

    mock_client.models.embed_content.return_value = _make_mock_response(1)

    result = emb.embed_texts(["hello", "world"], ["t_001", "t_002"])

    assert len(result) == 2
    # Only one API call (for t_002), t_001 came from cache
    mock_client.models.embed_content.assert_called_once()
    np.testing.assert_array_almost_equal(result["t_001"], cached_vec)


def test_embed_texts_all_cached(embedder, tmp_path: Path):
    """If all embeddings are cached, no API call should be made."""
    emb, mock_client = embedder

    for i in range(3):
        np.save(tmp_path / f"t_{i:03d}.npy", np.random.randn(EMBEDDING_DIM))

    result = emb.embed_texts(["a", "b", "c"], ["t_000", "t_001", "t_002"])

    assert len(result) == 3
    mock_client.models.embed_content.assert_not_called()


def test_embed_images_batching(embedder, tmp_path: Path):
    """8 images should produce 2 batches (6 + 2)."""
    emb, mock_client = embedder
    mock_client.models.embed_content.side_effect = [
        _make_mock_response(6),
        _make_mock_response(2),
    ]

    # Create fake image files
    paths = []
    ids = []
    for i in range(8):
        p = tmp_path / f"img_{i:05d}.png"
        p.write_bytes(b"fake-png")
        paths.append(p)
        ids.append(f"img_{i:05d}")

    result = emb.embed_images(paths, ids)

    assert len(result) == 8
    assert mock_client.models.embed_content.call_count == 2


def test_embed_query(embedder):
    emb, mock_client = embedder
    mock_client.models.embed_content.return_value = _make_mock_response(1)

    vec = emb.embed_query("What is the revenue?")

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (EMBEDDING_DIM,)
