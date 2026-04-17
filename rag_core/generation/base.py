"""Abstract interface for answer generation."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from rag_core.models import GenerationResult


class Generator(Protocol):
    """Protocol for generation backends."""

    def generate(
        self,
        query: str,
        page_images: list[Path],
        text_chunks: list[str],
    ) -> GenerationResult: ...
