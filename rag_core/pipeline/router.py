"""End-to-end RAG pipeline: query → retrieve → generate → answer."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rag_core.config import PAGES_DIR
from rag_core.embeddings.gemini_embedder import GeminiEmbedder
from rag_core.generation.gemini import GeminiGenerator
from rag_core.retrieval.pinecone_store import PineconeStore
from rag_core.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Orchestrates retrieval and generation for a given system configuration."""

    def __init__(self, system: str = "unified") -> None:
        self.system = system
        self._embedder = GeminiEmbedder()
        self._store = PineconeStore()
        self._retriever = Retriever(self._embedder, self._store)
        self._generator = GeminiGenerator()

    def answer(self, query: str) -> dict[str, Any]:
        """Run the full RAG pipeline on a query.

        Returns dict with:
            - answer: str
            - sources: list[int]
            - retrieval_results: list of RetrievalResult
            - page_corpus_ids: list[int]
        """
        # Retrieve top pages
        page_ids, results = self._retriever.retrieve_pages(
            query, top_k_pages=3, system=self.system
        )

        # Collect page images and text chunks
        page_images: list[Path] = []
        text_chunks: list[str] = []
        seen_images: set[str] = set()

        for r in results:
            if r.record_type == "page_image" and r.image_path:
                if r.image_path not in seen_images:
                    # Construct absolute path from corpus_id for reliability
                    img = PAGES_DIR / f"{r.corpus_id:05d}.png"
                    if img.exists():
                        page_images.append(img)
                    seen_images.add(r.image_path)
            elif r.record_type == "text_chunk" and r.chunk_text:
                label = f"[{r.modality}]" if r.modality else ""
                text_chunks.append(f"{label} {r.chunk_text}")

        # If no images were retrieved (e.g. text_only mode), still send page images
        # for the top pages so the VLM can use visual context for generation.
        if not page_images:
            for cid in page_ids:
                img = PAGES_DIR / f"{cid:05d}.png"
                if img.exists():
                    page_images.append(img)

        # Generate answer
        gen_result = self._generator.generate(query, page_images, text_chunks)

        logger.info(
            "Pipeline: %d pages, %d images, %d chunks → answer (%d chars)",
            len(page_ids),
            len(page_images),
            len(text_chunks),
            len(gen_result.answer),
        )

        return {
            "answer": gen_result.answer,
            "sources": gen_result.sources,
            "retrieval_results": results,
            "page_corpus_ids": page_ids,
        }
