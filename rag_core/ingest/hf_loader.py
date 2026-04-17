"""Load the ViDoRe V3 Finance dataset from HuggingFace and cache page images."""

from __future__ import annotations

import logging
from pathlib import Path

from datasets import load_dataset

from rag_core.config import HF_DATASET, PAGES_DIR
from rag_core.models import PageRecord

logger = logging.getLogger(__name__)


def load_corpus(limit: int | None = None) -> list[PageRecord]:
    """Load corpus split and cache page images as PNGs.

    Args:
        limit: If set, only load the first *limit* pages (for dev/testing).

    Returns:
        List of PageRecord with image_path pointing to the cached PNG.
    """
    ds = load_dataset(HF_DATASET, data_dir="corpus", split="test")

    records: list[PageRecord] = []
    total = min(limit, len(ds)) if limit else len(ds)

    for idx in range(total):
        row = ds[idx]
        corpus_id: int = row["corpus_id"]
        doc_id: str = row["doc_id"]
        page_number: int = row["page_number_in_doc"]
        markdown: str = row["markdown"] or ""
        image = row["image"]  # PIL Image

        # Cache page image to disk
        image_path = PAGES_DIR / f"{corpus_id:05d}.png"
        if not image_path.exists():
            image.save(image_path, format="PNG")
            logger.debug("Saved %s", image_path)

        records.append(
            PageRecord(
                corpus_id=corpus_id,
                doc_id=doc_id,
                page_number=page_number,
                image_path=image_path,
                markdown=markdown,
            )
        )

        if (idx + 1) % 500 == 0:
            logger.info("Loaded %d / %d pages", idx + 1, total)

    logger.info("Loaded %d pages total", len(records))
    return records


def load_queries() -> list[dict]:
    """Load the queries split. Returns list of dicts with query_id, query, answer, source_type."""
    ds = load_dataset(HF_DATASET, data_dir="queries", split="test")
    return [dict(row) for row in ds]


def load_qrels() -> list[dict]:
    """Load the qrels split. Returns list of dicts with query_id, corpus_id, score, content_type, bounding_boxes."""
    ds = load_dataset(HF_DATASET, data_dir="qrels", split="test")
    return [dict(row) for row in ds]
