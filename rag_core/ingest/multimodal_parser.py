"""Parse a page's markdown into modality-tagged segments (text, table, chart_metadata, footnote)."""

from __future__ import annotations

import re

from rag_core.models import Modality, Segment

# --- Patterns ----------------------------------------------------------------

_TABLE_SEP_RE = re.compile(r"^\|[\s\-:]+\|", re.MULTILINE)
_TABLE_ROW_RE = re.compile(r"^\|.+\|", re.MULTILINE)

_CHART_RE = re.compile(
    r"(?:Figure|Chart|Graph|Exhibit)\s+\d+",
    re.IGNORECASE,
)
_IMAGE_ALT_RE = re.compile(r"!\[([^\]]*)\]\(")

_FOOTNOTE_SUPER_RE = re.compile(r"[¹²³⁴⁵⁶⁷⁸⁹⁰]+")
_FOOTNOTE_BRACKET_RE = re.compile(r"^\[\d+\]\s")
_FOOTNOTE_SLASH_RE = re.compile(r"^\d+[/.]\s")
_FOOTNOTE_STAR_RE = re.compile(r"^\*\s")


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|")


def _is_separator_line(line: str) -> bool:
    return bool(_TABLE_SEP_RE.match(line.strip()))


def _classify_block(text: str) -> Modality:
    """Determine the modality of a text block."""
    lines = text.strip().splitlines()

    # Table: contains pipe-delimited rows with a separator line
    table_lines = [l for l in lines if _is_table_line(l)]
    sep_lines = [l for l in lines if _is_separator_line(l)]
    if len(table_lines) >= 2 and len(sep_lines) >= 1:
        return "table"

    # Chart/figure metadata
    if _CHART_RE.search(text) or _IMAGE_ALT_RE.search(text):
        return "chart_metadata"

    # Footnote patterns
    first_line = lines[0].strip() if lines else ""
    if (
        _FOOTNOTE_BRACKET_RE.match(first_line)
        or _FOOTNOTE_SLASH_RE.match(first_line)
        or _FOOTNOTE_STAR_RE.match(first_line)
    ):
        return "footnote"

    # Check for leading superscript digits (standalone footnote lines)
    if first_line and _FOOTNOTE_SUPER_RE.match(first_line):
        return "footnote"

    return "text"


def _split_into_blocks(markdown: str) -> list[str]:
    """Split markdown into blocks, preserving tables as single units.

    Strategy: walk line by line. When inside a table (consecutive pipe-rows),
    accumulate into one block. Otherwise split on blank lines.
    """
    lines = markdown.splitlines()
    blocks: list[str] = []
    current: list[str] = []
    in_table = False

    for line in lines:
        is_tbl = _is_table_line(line) or _is_separator_line(line)

        if is_tbl:
            if not in_table and current:
                # Flush the non-table block before starting a table
                blocks.append("\n".join(current))
                current = []
            in_table = True
            current.append(line)
        else:
            if in_table:
                # End of table — flush
                blocks.append("\n".join(current))
                current = []
                in_table = False

            if line.strip() == "":
                if current:
                    blocks.append("\n".join(current))
                    current = []
            else:
                current.append(line)

    if current:
        blocks.append("\n".join(current))

    return [b for b in blocks if b.strip()]


def parse_page(
    markdown: str,
    corpus_id: int,
    page_number: int,
    doc_id: str,
) -> list[Segment]:
    """Parse a page's markdown into modality-tagged Segments.

    Args:
        markdown: Raw markdown text of the page.
        corpus_id: Unique page identifier from the corpus.
        page_number: Page number within the document.
        doc_id: Document identifier.

    Returns:
        List of Segments, each tagged with its modality.
    """
    if not markdown or not markdown.strip():
        return []

    blocks = _split_into_blocks(markdown)
    segments: list[Segment] = []

    for block in blocks:
        modality = _classify_block(block)
        segments.append(
            Segment(
                text=block.strip(),
                modality=modality,
                corpus_id=corpus_id,
                page_number=page_number,
                doc_id=doc_id,
            )
        )

    return segments
