"""Gemini Embedding 2 wrapper: embed text and images with batching, rate limiting, and caching."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from google import genai
from google.genai import types

from rag_core.config import (
    EMBEDDING_DIM,
    EMBEDDING_IMAGE_BATCH,
    EMBEDDING_MODEL,
    EMBEDDING_SLEEP_S,
    EMBEDDING_TEXT_BATCH,
    EMBEDDINGS_DIR,
    GEMINI_API_KEY,
)

logger = logging.getLogger(__name__)


class GeminiEmbedder:
    """Embed text and images using Gemini Embedding 2 with batching and disk cache."""

    def __init__(self) -> None:
        self._client = genai.Client(api_key=GEMINI_API_KEY)

    # --- Cache ---------------------------------------------------------------

    @staticmethod
    def _cache_path(record_id: str) -> Path:
        return EMBEDDINGS_DIR / f"{record_id}.npy"

    @staticmethod
    def _load_cache(record_id: str) -> np.ndarray | None:
        path = GeminiEmbedder._cache_path(record_id)
        if path.exists():
            return np.load(path)
        return None

    @staticmethod
    def _save_cache(record_id: str, vec: np.ndarray) -> None:
        np.save(GeminiEmbedder._cache_path(record_id), vec)

    # --- Text embedding ------------------------------------------------------

    def embed_texts(
        self, texts: list[str], ids: list[str]
    ) -> dict[str, np.ndarray]:
        """Embed text strings. Batches in groups of EMBEDDING_TEXT_BATCH with sleep.

        Returns dict mapping record_id -> 3072-dim vector.
        Skips API call for any id that already has a cached .npy file.
        """
        result: dict[str, np.ndarray] = {}
        to_embed: list[tuple[int, str, str]] = []  # (original_idx, id, text)

        # Check cache first
        for i, (text, rid) in enumerate(zip(texts, ids)):
            cached = self._load_cache(rid)
            if cached is not None:
                result[rid] = cached
            else:
                to_embed.append((i, rid, text))

        if not to_embed:
            logger.info("All %d text embeddings loaded from cache", len(ids))
            return result

        logger.info(
            "Embedding %d texts (%d cached, %d to embed)",
            len(ids),
            len(result),
            len(to_embed),
        )

        # Batch embed
        for batch_start in range(0, len(to_embed), EMBEDDING_TEXT_BATCH):
            batch = to_embed[batch_start : batch_start + EMBEDDING_TEXT_BATCH]
            batch_texts = [t for _, _, t in batch]
            batch_ids = [rid for _, rid, _ in batch]

            # Each text must be a separate Content object for true batch embedding.
            # Passing a list of strings produces a single concatenated embedding.
            contents = [
                types.Content(parts=[types.Part(text=t)]) for t in batch_texts
            ]
            response = self._client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=contents,
                config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM),
            )

            for rid, emb in zip(batch_ids, response.embeddings):
                vec = np.array(emb.values, dtype=np.float32)
                self._save_cache(rid, vec)
                result[rid] = vec

            logger.debug(
                "Text batch %d-%d done",
                batch_start,
                batch_start + len(batch),
            )

            if batch_start + EMBEDDING_TEXT_BATCH < len(to_embed):
                time.sleep(EMBEDDING_SLEEP_S)

        return result

    # --- Image embedding -----------------------------------------------------

    def embed_images(
        self, image_paths: list[Path], ids: list[str]
    ) -> dict[str, np.ndarray]:
        """Embed page images. Batches in groups of EMBEDDING_IMAGE_BATCH with sleep.

        Returns dict mapping record_id -> 3072-dim vector.
        Skips API call for any id that already has a cached .npy file.
        """
        result: dict[str, np.ndarray] = {}
        to_embed: list[tuple[Path, str]] = []

        for path, rid in zip(image_paths, ids):
            cached = self._load_cache(rid)
            if cached is not None:
                result[rid] = cached
            else:
                to_embed.append((path, rid))

        if not to_embed:
            logger.info("All %d image embeddings loaded from cache", len(ids))
            return result

        logger.info(
            "Embedding %d images (%d cached, %d to embed)",
            len(ids),
            len(result),
            len(to_embed),
        )

        for batch_start in range(0, len(to_embed), EMBEDDING_IMAGE_BATCH):
            batch = to_embed[batch_start : batch_start + EMBEDDING_IMAGE_BATCH]

            # Each image must be a separate Content object for true batch embedding.
            contents = []
            batch_ids = []
            for path, rid in batch:
                img_bytes = path.read_bytes()
                contents.append(
                    types.Content(
                        parts=[types.Part.from_bytes(data=img_bytes, mime_type="image/png")]
                    )
                )
                batch_ids.append(rid)

            response = self._client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=contents,
                config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM),
            )

            for rid, emb in zip(batch_ids, response.embeddings):
                vec = np.array(emb.values, dtype=np.float32)
                self._save_cache(rid, vec)
                result[rid] = vec

            logger.debug(
                "Image batch %d-%d done",
                batch_start,
                batch_start + len(batch),
            )

            if batch_start + EMBEDDING_IMAGE_BATCH < len(to_embed):
                time.sleep(EMBEDDING_SLEEP_S)

        return result

    # --- Query embedding (no cache) ------------------------------------------

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. No caching (queries are ephemeral)."""
        response = self._client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[query],
            config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM),
        )
        return np.array(response.embeddings[0].values, dtype=np.float32)
