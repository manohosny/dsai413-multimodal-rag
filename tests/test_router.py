"""Tests for pipeline/router.py — all components mocked."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from rag_core.models import GenerationResult, RetrievalResult


def test_pipeline_answer():
    """RAGPipeline.answer() should call retriever then generator and return structured result."""
    with (
        patch("rag_core.pipeline.router.GeminiEmbedder"),
        patch("rag_core.pipeline.router.PineconeStore"),
        patch("rag_core.pipeline.router.Retriever") as MockRetriever,
        patch("rag_core.pipeline.router.GeminiGenerator") as MockGenerator,
    ):
        # Setup retriever mock
        mock_retriever = MockRetriever.return_value
        mock_retriever.retrieve_pages.return_value = (
            [1, 2],
            [
                RetrievalResult(
                    corpus_id=1,
                    score=0.9,
                    record_type="page_image",
                    modality="",
                    chunk_text="",
                    image_path="/tmp/test/00001.png",
                ),
                RetrievalResult(
                    corpus_id=1,
                    score=0.85,
                    record_type="text_chunk",
                    modality="table",
                    chunk_text="Revenue: $4.2B",
                    image_path="",
                ),
                RetrievalResult(
                    corpus_id=2,
                    score=0.8,
                    record_type="page_image",
                    modality="",
                    chunk_text="",
                    image_path="/tmp/test/00002.png",
                ),
            ],
        )

        # Setup generator mock
        mock_generator = MockGenerator.return_value
        mock_generator.generate.return_value = GenerationResult(
            answer="Revenue was $4.2B. Sources: [1, 2]",
            sources=[1, 2],
        )

        from rag_core.pipeline.router import RAGPipeline

        pipe = RAGPipeline(system="unified")
        result = pipe.answer("What is the revenue?")

        assert result["answer"] == "Revenue was $4.2B. Sources: [1, 2]"
        assert result["sources"] == [1, 2]
        assert result["page_corpus_ids"] == [1, 2]
        assert len(result["retrieval_results"]) == 3

        # Verify retriever was called with correct system
        mock_retriever.retrieve_pages.assert_called_once_with(
            "What is the revenue?", top_k_pages=3, system="unified"
        )

        # Verify generator received images and chunks
        mock_generator.generate.assert_called_once()
        call_args = mock_generator.generate.call_args
        assert len(call_args[1].get("page_images", call_args[0][1])) == 2  # 2 images
        text_chunks = call_args[1].get("text_chunks", call_args[0][2])
        assert any("Revenue" in c for c in text_chunks)
