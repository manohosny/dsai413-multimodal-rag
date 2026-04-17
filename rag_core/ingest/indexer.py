"""End-to-end indexing pipeline: load → parse → chunk → embed → upsert to Pinecone."""

from __future__ import annotations

import argparse
import logging
import sys

from rag_core.config import require_keys
from rag_core.embeddings.gemini_embedder import GeminiEmbedder
from rag_core.ingest.chunker import chunk_segments
from rag_core.ingest.hf_loader import load_corpus
from rag_core.ingest.multimodal_parser import parse_page
from rag_core.models import Chunk
from rag_core.retrieval.pinecone_store import PineconeStore

logger = logging.getLogger(__name__)


def run_indexing(limit: int | None = None) -> None:
    """Run the full indexing pipeline.

    1. Load corpus from HuggingFace (with optional limit).
    2. Parse each page's markdown into modality-tagged segments.
    3. Chunk segments into embedding-ready pieces.
    4. Embed all text chunks and page images with Gemini Embedding 2.
    5. Upsert everything into the Pinecone index.
    """
    require_keys()

    # --- Load ---
    logger.info("Loading corpus (limit=%s)...", limit)
    pages = load_corpus(limit=limit)
    logger.info("Loaded %d pages", len(pages))

    # --- Parse + Chunk ---
    all_chunks: list[Chunk] = []
    for page in pages:
        segments = parse_page(
            page.markdown, page.corpus_id, page.page_number, page.doc_id
        )
        chunks = chunk_segments(segments)
        all_chunks.extend(chunks)

    logger.info(
        "Parsed %d pages → %d chunks",
        len(pages),
        len(all_chunks),
    )

    # --- Embed text chunks ---
    embedder = GeminiEmbedder()

    chunk_texts = [c.text for c in all_chunks]
    chunk_ids = [c.chunk_id for c in all_chunks]
    logger.info("Embedding %d text chunks...", len(chunk_texts))
    text_vecs = embedder.embed_texts(chunk_texts, chunk_ids)

    # --- Embed page images ---
    image_paths = [p.image_path for p in pages]
    image_ids = [f"img_{p.corpus_id:05d}" for p in pages]
    logger.info("Embedding %d page images...", len(image_paths))
    image_vecs = embedder.embed_images(image_paths, image_ids)

    # --- Build Pinecone records ---
    vectors = []

    # Page image records
    for page in pages:
        img_id = f"img_{page.corpus_id:05d}"
        if img_id not in image_vecs:
            continue
        vectors.append(
            {
                "id": img_id,
                "values": image_vecs[img_id].tolist(),
                "metadata": {
                    "corpus_id": page.corpus_id,
                    "record_type": "page_image",
                    "doc_id": page.doc_id,
                    "page_number": page.page_number,
                    "modality": "",
                    "chunk_text": "",
                    "image_path": str(page.image_path),
                },
            }
        )

    # Text chunk records
    for chunk in all_chunks:
        if chunk.chunk_id not in text_vecs:
            continue
        vectors.append(
            {
                "id": chunk.chunk_id,
                "values": text_vecs[chunk.chunk_id].tolist(),
                "metadata": {
                    "corpus_id": chunk.corpus_id,
                    "record_type": "text_chunk",
                    "doc_id": chunk.doc_id,
                    "page_number": chunk.page_number,
                    "modality": chunk.modality,
                    "chunk_text": chunk.text[:1000],  # Pinecone metadata limit
                    "image_path": "",
                },
            }
        )

    logger.info("Built %d Pinecone vectors (%d images + %d chunks)",
                len(vectors), len(image_vecs), len(text_vecs))

    # --- Upsert ---
    store = PineconeStore()
    store.ensure_index()
    store.upsert_batch(vectors)

    stats = store.describe()
    logger.info("Done. Index stats: %s", stats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Index corpus into Pinecone")
    parser.add_argument("--limit", type=int, default=None, help="Max pages to index")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    run_indexing(limit=args.limit)


if __name__ == "__main__":
    main()
