"""Tests for multimodal_parser.py."""

from __future__ import annotations

from rag_core.ingest.multimodal_parser import parse_page
from tests.conftest import (
    SAMPLE_MARKDOWN,
    SAMPLE_MARKDOWN_NO_SPECIAL,
    SAMPLE_MARKDOWN_TABLES_ONLY,
)


def _parse(md: str) -> list:
    return parse_page(md, corpus_id=1, page_number=1, doc_id="test")


def test_parses_all_four_modalities():
    segments = _parse(SAMPLE_MARKDOWN)
    modalities = {s.modality for s in segments}
    assert "text" in modalities
    assert "table" in modalities
    assert "chart_metadata" in modalities
    assert "footnote" in modalities


def test_table_is_single_segment():
    """A markdown table should be one segment, never split across multiple."""
    segments = _parse(SAMPLE_MARKDOWN)
    tables = [s for s in segments if s.modality == "table"]
    assert len(tables) == 1
    # Table should contain all rows
    assert "Revenue" in tables[0].text
    assert "Net Income" in tables[0].text


def test_table_preserves_delimiters():
    """Table segment must keep | and --- delimiters intact."""
    segments = _parse(SAMPLE_MARKDOWN)
    table = [s for s in segments if s.modality == "table"][0]
    assert "|" in table.text
    assert "---" in table.text


def test_chart_metadata_detected():
    segments = _parse(SAMPLE_MARKDOWN)
    charts = [s for s in segments if s.modality == "chart_metadata"]
    assert len(charts) >= 1
    assert "Figure 3" in charts[0].text


def test_footnote_detected():
    segments = _parse(SAMPLE_MARKDOWN)
    footnotes = [s for s in segments if s.modality == "footnote"]
    assert len(footnotes) >= 1
    assert "200M" in footnotes[0].text


def test_plain_text_only():
    """Markdown with no special blocks should produce only text segments."""
    segments = _parse(SAMPLE_MARKDOWN_NO_SPECIAL)
    assert all(s.modality == "text" for s in segments)
    assert len(segments) >= 1


def test_tables_only():
    segments = _parse(SAMPLE_MARKDOWN_TABLES_ONLY)
    assert len(segments) == 1
    assert segments[0].modality == "table"


def test_empty_markdown():
    assert _parse("") == []
    assert _parse("   ") == []
    assert _parse("\n\n") == []


def test_segment_metadata():
    """Segments carry corpus_id, page_number, doc_id from caller."""
    segments = parse_page("Hello world.", corpus_id=42, page_number=7, doc_id="doc_x")
    assert segments[0].corpus_id == 42
    assert segments[0].page_number == 7
    assert segments[0].doc_id == "doc_x"


def test_consecutive_tables_stay_separate():
    """Two tables separated by a blank line should be two segments."""
    md = """\
| A | B |
|---|---|
| 1 | 2 |

| C | D |
|---|---|
| 3 | 4 |
"""
    segments = _parse(md)
    tables = [s for s in segments if s.modality == "table"]
    assert len(tables) == 2
