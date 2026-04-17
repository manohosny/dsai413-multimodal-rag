"""Gemini 2.0 Flash generation: accept page images + text chunks, produce an answer."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from google import genai
from google.genai import types

from rag_core.config import GEMINI_API_KEY, GENERATION_MODEL
from rag_core.generation.prompts import QA_PROMPT
from rag_core.models import GenerationResult

logger = logging.getLogger(__name__)


class GeminiGenerator:
    """Generate answers using Gemini 2.0 Flash with page images and text context."""

    def __init__(self) -> None:
        self._client = genai.Client(api_key=GEMINI_API_KEY)

    def generate(
        self,
        query: str,
        page_images: list[Path],
        text_chunks: list[str],
    ) -> GenerationResult:
        """Generate an answer from page images and text chunks.

        Args:
            query: The user's question.
            page_images: Paths to page PNG images (top-k retrieved pages).
            text_chunks: Matched text chunks with modality context.

        Returns:
            GenerationResult with answer text and source page IDs.
        """
        # Build text context
        chunks_text = "\n---\n".join(text_chunks) if text_chunks else "(none)"
        prompt_text = QA_PROMPT.format(text_chunks=chunks_text, question=query)

        # Build multimodal content: images first, then text prompt
        contents: list = []
        for img_path in page_images:
            if img_path.exists():
                img_bytes = img_path.read_bytes()
                contents.append(
                    types.Part.from_bytes(data=img_bytes, mime_type="image/png")
                )

        contents.append(prompt_text)

        # Call Gemini Flash
        response = self._client.models.generate_content(
            model=GENERATION_MODEL,
            contents=contents,
        )

        answer_text = response.text or ""

        # Extract source page IDs from "Sources: [...]" line
        sources = self._parse_sources(answer_text)

        return GenerationResult(answer=answer_text, sources=sources)

    @staticmethod
    def _parse_sources(text: str) -> list[int]:
        """Extract page IDs from 'Sources: [...]' at end of answer."""
        match = re.search(r"Sources:\s*\[([^\]]*)\]", text)
        if not match:
            return []
        raw = match.group(1)
        ids = []
        for part in raw.split(","):
            part = part.strip()
            # Extract digits from patterns like "page 42", "42", "img_00042"
            digits = re.findall(r"\d+", part)
            if digits:
                ids.append(int(digits[-1]))
        return ids
