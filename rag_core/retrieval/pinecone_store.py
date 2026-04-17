"""Pinecone serverless index management: create, upsert, query."""

from __future__ import annotations

import logging
import time
from typing import Any

from pinecone import Pinecone, ServerlessSpec

from rag_core.config import (
    EMBEDDING_DIM,
    INDEX_NAME,
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_REGION,
    PINECONE_UPSERT_BATCH,
)

logger = logging.getLogger(__name__)


class PineconeStore:
    """Manages the Pinecone serverless index for multimodal embeddings."""

    def __init__(self) -> None:
        self._pc = Pinecone(api_key=PINECONE_API_KEY)
        self._index = None

    def _get_index(self):
        if self._index is None:
            self._index = self._pc.Index(INDEX_NAME)
        return self._index

    def ensure_index(self) -> None:
        """Create the index if it doesn't already exist, then connect."""
        existing = [idx.name for idx in self._pc.list_indexes()]
        if INDEX_NAME not in existing:
            logger.info(
                "Creating index '%s' (dim=%d, cosine, serverless)",
                INDEX_NAME,
                EMBEDDING_DIM,
            )
            self._pc.create_index(
                name=INDEX_NAME,
                dimension=EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
            )
            # Wait for index to be ready
            while not self._pc.describe_index(INDEX_NAME).status.get("ready"):
                time.sleep(2)
            logger.info("Index '%s' is ready", INDEX_NAME)
        else:
            logger.info("Index '%s' already exists", INDEX_NAME)

        self._index = self._pc.Index(INDEX_NAME)

    def upsert_batch(self, vectors: list[dict[str, Any]]) -> int:
        """Upsert vectors in batches of PINECONE_UPSERT_BATCH.

        Args:
            vectors: List of dicts with 'id', 'values', 'metadata'.

        Returns:
            Total number of vectors upserted.
        """
        index = self._get_index()
        total = 0

        for i in range(0, len(vectors), PINECONE_UPSERT_BATCH):
            batch = vectors[i : i + PINECONE_UPSERT_BATCH]
            index.upsert(vectors=batch)
            total += len(batch)
            logger.debug("Upserted batch %d-%d", i, i + len(batch))

        logger.info("Upserted %d vectors total", total)
        return total

    def query(
        self,
        vector: list[float],
        top_k: int = 20,
        filter: dict | None = None,
    ) -> list[dict]:
        """Query the index and return matches with metadata.

        Returns list of dicts with 'id', 'score', and 'metadata'.
        """
        index = self._get_index()
        result = index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            filter=filter,
        )
        return [
            {
                "id": m.id,
                "score": m.score,
                "metadata": dict(m.metadata) if m.metadata else {},
            }
            for m in result.matches
        ]

    def describe(self) -> dict:
        """Return index statistics."""
        index = self._get_index()
        stats = index.describe_index_stats()
        return {
            "total_vector_count": stats.total_vector_count,
            "dimension": stats.dimension,
        }
