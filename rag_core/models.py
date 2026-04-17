"""Shared data structures used across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


Modality = Literal["text", "table", "chart_metadata", "footnote"]


@dataclass(frozen=True)
class Segment:
    """A parsed section of a page's markdown, tagged with its modality."""

    text: str
    modality: Modality
    corpus_id: int
    page_number: int
    doc_id: str


@dataclass(frozen=True)
class Chunk:
    """An embedding-ready piece of text derived from one or more Segments."""

    chunk_id: str
    text: str
    modality: Modality
    corpus_id: int
    doc_id: str
    page_number: int


@dataclass
class PageRecord:
    """One corpus page: its image path, raw markdown, and identifiers."""

    corpus_id: int
    doc_id: str
    page_number: int
    image_path: Path
    markdown: str


@dataclass
class RetrievalResult:
    """A single Pinecone match returned during retrieval."""

    corpus_id: int
    score: float
    record_type: str  # "page_image" | "text_chunk"
    modality: str
    chunk_text: str
    image_path: str


@dataclass
class GenerationResult:
    """The answer produced by the generation step."""

    answer: str
    sources: list[int] = field(default_factory=list)
