"""Modality-aware chunking: split Segments into embedding-ready Chunks."""

from __future__ import annotations

import re

from rag_core.models import Chunk, Segment

# Approximate tokens: 1 word ~= 1.3 tokens
_WORDS_PER_TOKEN = 1.0 / 1.3

_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")

# --- Chunking parameters per modality ----------------------------------------

TEXT_CHUNK_TOKENS = 200
TEXT_OVERLAP_TOKENS = 50
TABLE_MAX_TOKENS = 500


def _estimate_tokens(text: str) -> int:
    """Rough token count from word count."""
    return int(len(text.split()) / _WORDS_PER_TOKEN)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, then break oversized sentences on word boundaries."""
    parts = _SENTENCE_END_RE.split(text)
    result: list[str] = []
    max_words = int(TEXT_CHUNK_TOKENS * _WORDS_PER_TOKEN)

    for p in parts:
        p = p.strip()
        if not p:
            continue
        words = p.split()
        if len(words) <= max_words:
            result.append(p)
        else:
            # Split oversized sentence into word-boundary chunks
            for i in range(0, len(words), max_words):
                result.append(" ".join(words[i : i + max_words]))

    return result


def _chunk_text(segment: Segment, chunk_idx_start: int) -> tuple[list[Chunk], int]:
    """Chunk a text segment by sentence boundaries with overlap."""
    sentences = _split_sentences(segment.text)
    if not sentences:
        return [], chunk_idx_start

    chunks: list[Chunk] = []
    idx = chunk_idx_start
    current_sents: list[str] = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = _estimate_tokens(sent)

        if current_tokens + sent_tokens > TEXT_CHUNK_TOKENS and current_sents:
            # Emit chunk
            chunks.append(
                Chunk(
                    chunk_id=f"tc_{segment.corpus_id:05d}_{idx:03d}",
                    text=" ".join(current_sents),
                    modality=segment.modality,
                    corpus_id=segment.corpus_id,
                    doc_id=segment.doc_id,
                    page_number=segment.page_number,
                )
            )
            idx += 1

            # Overlap: keep last N tokens worth of sentences
            overlap_sents: list[str] = []
            overlap_tokens = 0
            for s in reversed(current_sents):
                st = _estimate_tokens(s)
                if overlap_tokens + st > TEXT_OVERLAP_TOKENS:
                    break
                overlap_sents.insert(0, s)
                overlap_tokens += st

            current_sents = overlap_sents
            current_tokens = overlap_tokens

        current_sents.append(sent)
        current_tokens += sent_tokens

    # Final chunk
    if current_sents:
        chunks.append(
            Chunk(
                chunk_id=f"tc_{segment.corpus_id:05d}_{idx:03d}",
                text=" ".join(current_sents),
                modality=segment.modality,
                corpus_id=segment.corpus_id,
                doc_id=segment.doc_id,
                page_number=segment.page_number,
            )
        )
        idx += 1

    return chunks, idx


def _chunk_table(segment: Segment, chunk_idx_start: int) -> tuple[list[Chunk], int]:
    """Chunk a table segment. Keep whole if small; split at row boundaries if large."""
    tokens = _estimate_tokens(segment.text)
    idx = chunk_idx_start

    if tokens <= TABLE_MAX_TOKENS:
        chunk = Chunk(
            chunk_id=f"tc_{segment.corpus_id:05d}_{idx:03d}",
            text=segment.text,
            modality="table",
            corpus_id=segment.corpus_id,
            doc_id=segment.doc_id,
            page_number=segment.page_number,
        )
        return [chunk], idx + 1

    # Split at row boundaries, keeping header with each chunk
    lines = segment.text.splitlines()
    header_lines: list[str] = []
    data_lines: list[str] = []
    past_separator = False

    for line in lines:
        if not past_separator:
            header_lines.append(line)
            if re.match(r"^\|[\s\-:]+\|", line.strip()):
                past_separator = True
        else:
            data_lines.append(line)

    header = "\n".join(header_lines)
    header_tokens = _estimate_tokens(header)
    max_data_tokens = TABLE_MAX_TOKENS - header_tokens

    chunks: list[Chunk] = []
    current_rows: list[str] = []
    current_tokens = 0

    for row in data_lines:
        row_tokens = _estimate_tokens(row)
        if current_tokens + row_tokens > max_data_tokens and current_rows:
            text = header + "\n" + "\n".join(current_rows)
            chunks.append(
                Chunk(
                    chunk_id=f"tc_{segment.corpus_id:05d}_{idx:03d}",
                    text=text,
                    modality="table",
                    corpus_id=segment.corpus_id,
                    doc_id=segment.doc_id,
                    page_number=segment.page_number,
                )
            )
            idx += 1
            current_rows = []
            current_tokens = 0

        current_rows.append(row)
        current_tokens += row_tokens

    if current_rows:
        text = header + "\n" + "\n".join(current_rows)
        chunks.append(
            Chunk(
                chunk_id=f"tc_{segment.corpus_id:05d}_{idx:03d}",
                text=text,
                modality="table",
                corpus_id=segment.corpus_id,
                doc_id=segment.doc_id,
                page_number=segment.page_number,
            )
        )
        idx += 1

    return chunks, idx


def _chunk_passthrough(segment: Segment, chunk_idx_start: int) -> tuple[list[Chunk], int]:
    """Keep chart_metadata and footnote segments whole."""
    chunk = Chunk(
        chunk_id=f"tc_{segment.corpus_id:05d}_{chunk_idx_start:03d}",
        text=segment.text,
        modality=segment.modality,
        corpus_id=segment.corpus_id,
        doc_id=segment.doc_id,
        page_number=segment.page_number,
    )
    return [chunk], chunk_idx_start + 1


def chunk_segments(segments: list[Segment]) -> list[Chunk]:
    """Convert parsed Segments into embedding-ready Chunks.

    Chunking strategy per modality:
    - text: ~200 tokens with 50-token overlap at sentence boundaries.
    - table: whole table; split at row boundaries if >500 tokens.
    - chart_metadata: keep whole.
    - footnote: keep whole.
    """
    chunks: list[Chunk] = []
    # Track chunk index per corpus_id for unique IDs
    idx_by_page: dict[int, int] = {}

    for seg in segments:
        idx = idx_by_page.get(seg.corpus_id, 0)

        if seg.modality == "text":
            new_chunks, idx = _chunk_text(seg, idx)
        elif seg.modality == "table":
            new_chunks, idx = _chunk_table(seg, idx)
        else:
            new_chunks, idx = _chunk_passthrough(seg, idx)

        chunks.extend(new_chunks)
        idx_by_page[seg.corpus_id] = idx

    return chunks
