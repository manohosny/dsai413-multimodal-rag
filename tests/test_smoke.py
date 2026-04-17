"""Smoke tests: verify all packages import cleanly."""

from __future__ import annotations


def test_import_rag_core():
    import rag_core  # noqa: F401


def test_import_config():
    from rag_core import config  # noqa: F401
    assert config.EMBEDDING_DIM == 3072
    assert config.INDEX_NAME == "multimodal-rag"


def test_import_models():
    from rag_core.models import (  # noqa: F401
        Segment,
        Chunk,
        PageRecord,
        RetrievalResult,
        GenerationResult,
    )


def test_import_hf_loader():
    from rag_core.ingest.hf_loader import (  # noqa: F401
        load_corpus,
        load_queries,
        load_qrels,
    )
