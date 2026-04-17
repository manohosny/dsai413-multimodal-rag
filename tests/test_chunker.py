"""Tests for chunker.py."""

from __future__ import annotations

from rag_core.ingest.chunker import chunk_segments
from rag_core.models import Segment


def _seg(text: str, modality: str = "text", corpus_id: int = 1) -> Segment:
    return Segment(
        text=text,
        modality=modality,  # type: ignore[arg-type]
        corpus_id=corpus_id,
        page_number=1,
        doc_id="test",
    )


def test_short_text_single_chunk():
    """Text under the token limit should produce exactly one chunk."""
    seg = _seg("This is a short sentence.")
    chunks = chunk_segments([seg])
    assert len(chunks) == 1
    assert chunks[0].text == "This is a short sentence."
    assert chunks[0].modality == "text"


def test_long_text_splits():
    """Long text should be split into multiple chunks."""
    # ~400 words ≈ ~520 tokens, should produce at least 2 chunks
    words = "Word " * 400
    seg = _seg(words.strip() + ".")
    chunks = chunk_segments([seg])
    assert len(chunks) >= 2


def test_table_stays_whole():
    """Small table should remain as a single chunk."""
    table = """\
| Metric | Value |
|--------|-------|
| Revenue | $4.2B |
| Profit | $1.1B |"""
    seg = _seg(table, modality="table")
    chunks = chunk_segments([seg])
    assert len(chunks) == 1
    assert "|" in chunks[0].text
    assert chunks[0].modality == "table"


def test_chart_metadata_stays_whole():
    seg = _seg("Figure 3: Revenue breakdown by segment", modality="chart_metadata")
    chunks = chunk_segments([seg])
    assert len(chunks) == 1
    assert chunks[0].modality == "chart_metadata"


def test_footnote_stays_whole():
    seg = _seg("[1] Excludes restructuring charges.", modality="footnote")
    chunks = chunk_segments([seg])
    assert len(chunks) == 1
    assert chunks[0].modality == "footnote"


def test_chunk_ids_format():
    """Chunk IDs should follow tc_XXXXX_YYY format."""
    seg = _seg("Some text content.", corpus_id=42)
    chunks = chunk_segments([seg])
    assert chunks[0].chunk_id == "tc_00042_000"


def test_chunk_ids_unique_per_page():
    """Multiple segments from same page get sequential IDs."""
    segs = [
        _seg("First paragraph.", corpus_id=1),
        _seg("Second paragraph.", corpus_id=1),
    ]
    chunks = chunk_segments(segs)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), f"Duplicate IDs: {ids}"


def test_chunk_ids_reset_per_page():
    """Different pages start chunk index from 0."""
    segs = [
        _seg("Page one text.", corpus_id=1),
        _seg("Page two text.", corpus_id=2),
    ]
    chunks = chunk_segments(segs)
    assert chunks[0].chunk_id == "tc_00001_000"
    assert chunks[1].chunk_id == "tc_00002_000"


def test_empty_segments():
    assert chunk_segments([]) == []
