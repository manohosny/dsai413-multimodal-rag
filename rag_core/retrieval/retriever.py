"""Retrieve relevant pages: embed query → search Pinecone → group by page → rank."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

from rag_core.embeddings.gemini_embedder import GeminiEmbedder
from rag_core.models import RetrievalResult
from rag_core.retrieval.pinecone_store import PineconeStore

logger = logging.getLogger(__name__)

# Filter maps for eval system configurations
_SYSTEM_FILTERS: dict[str, dict | None] = {
    "unified": None,
    "text_only": {"record_type": "text_chunk"},
    "image_only": {"record_type": "page_image"},
}


class Retriever:
    """Embed a query, search Pinecone, and return results grouped by page."""

    def __init__(self, embedder: GeminiEmbedder, store: PineconeStore) -> None:
        self._embedder = embedder
        self._store = store

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        system: str = "unified",
    ) -> list[RetrievalResult]:
        """Embed query and search Pinecone. Returns raw matches."""
        query_vec = self._embedder.embed_query(query)
        pc_filter = _SYSTEM_FILTERS.get(system)

        matches = self._store.query(
            vector=query_vec.tolist(),
            top_k=top_k,
            filter=pc_filter,
        )

        results = []
        for m in matches:
            meta = m["metadata"]
            results.append(
                RetrievalResult(
                    corpus_id=int(meta.get("corpus_id", 0)),
                    score=m["score"],
                    record_type=meta.get("record_type", ""),
                    modality=meta.get("modality", ""),
                    chunk_text=meta.get("chunk_text", ""),
                    image_path=meta.get("image_path", ""),
                )
            )

        return results

    def retrieve_pages(
        self,
        query: str,
        top_k_pages: int = 3,
        system: str = "unified",
    ) -> tuple[list[int], list[RetrievalResult]]:
        """Retrieve and group results by page.

        Returns:
            - Top-k page corpus_ids, ranked by max score.
            - All RetrievalResult objects for those pages.
        """
        results = self.retrieve(query, top_k=20, system=system)

        # Group by corpus_id, track max score per page
        page_scores: dict[int, float] = defaultdict(float)
        page_results: dict[int, list[RetrievalResult]] = defaultdict(list)

        for r in results:
            page_scores[r.corpus_id] = max(page_scores[r.corpus_id], r.score)
            page_results[r.corpus_id].append(r)

        # Rank pages by combined score
        ranked_pages = sorted(page_scores.keys(), key=lambda cid: page_scores[cid], reverse=True)
        top_pages = ranked_pages[:top_k_pages]

        # Collect all results for top pages
        top_results = []
        for cid in top_pages:
            top_results.extend(page_results[cid])

        logger.info(
            "Retrieved %d results, grouped into %d pages, returning top %d",
            len(results),
            len(ranked_pages),
            len(top_pages),
        )

        return top_pages, top_results
