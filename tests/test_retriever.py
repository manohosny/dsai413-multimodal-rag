"""Tests for retriever.py — all API calls mocked."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from rag_core.config import EMBEDDING_DIM
from rag_core.retrieval.retriever import Retriever


def _make_retriever():
    embedder = MagicMock()
    embedder.embed_query.return_value = np.zeros(EMBEDDING_DIM)

    store = MagicMock()
    retriever = Retriever(embedder, store)
    return retriever, embedder, store


def _match(corpus_id, score, record_type="text_chunk", modality="text", chunk_text="sample"):
    return {
        "id": f"tc_{corpus_id:05d}_000",
        "score": score,
        "metadata": {
            "corpus_id": corpus_id,
            "record_type": record_type,
            "modality": modality,
            "chunk_text": chunk_text,
            "image_path": f"data/pages/{corpus_id:05d}.png" if record_type == "page_image" else "",
        },
    }


def test_retrieve_basic():
    retriever, embedder, store = _make_retriever()
    store.query.return_value = [
        _match(1, 0.9),
        _match(2, 0.8),
    ]

    results = retriever.retrieve("test query")
    assert len(results) == 2
    assert results[0].corpus_id == 1
    assert results[0].score == 0.9
    embedder.embed_query.assert_called_once_with("test query")


def test_retrieve_text_only_filter():
    retriever, _, store = _make_retriever()
    store.query.return_value = []

    retriever.retrieve("query", system="text_only")
    _, kwargs = store.query.call_args
    assert kwargs["filter"] == {"record_type": "text_chunk"}


def test_retrieve_image_only_filter():
    retriever, _, store = _make_retriever()
    store.query.return_value = []

    retriever.retrieve("query", system="image_only")
    _, kwargs = store.query.call_args
    assert kwargs["filter"] == {"record_type": "page_image"}


def test_retrieve_unified_no_filter():
    retriever, _, store = _make_retriever()
    store.query.return_value = []

    retriever.retrieve("query", system="unified")
    _, kwargs = store.query.call_args
    assert kwargs["filter"] is None


def test_retrieve_pages_groups_by_corpus_id():
    retriever, _, store = _make_retriever()
    store.query.return_value = [
        _match(1, 0.9, "text_chunk"),
        _match(1, 0.7, "page_image"),
        _match(2, 0.85, "text_chunk"),
        _match(3, 0.6, "text_chunk"),
    ]

    page_ids, results = retriever.retrieve_pages("test query", top_k_pages=2)

    # Page 1 has max score 0.9, page 2 has 0.85 → top-2
    assert page_ids == [1, 2]
    # Results for top-2 pages only
    result_cids = {r.corpus_id for r in results}
    assert result_cids == {1, 2}


def test_retrieve_pages_max_score_ranking():
    """Pages should be ranked by their highest individual match score."""
    retriever, _, store = _make_retriever()
    store.query.return_value = [
        _match(1, 0.5),
        _match(1, 0.4),
        _match(2, 0.8),  # Page 2 has higher max score
    ]

    page_ids, _ = retriever.retrieve_pages("query", top_k_pages=2)
    assert page_ids[0] == 2  # Page 2 should rank first
