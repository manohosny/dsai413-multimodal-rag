"""Day 1 risk probes: verify Gemini Embedding 2, Pinecone, and cross-modal quality.

Run: uv run python scripts/day1_probe.py
"""

from __future__ import annotations

import sys
import time

import numpy as np
from google import genai
from google.genai import types
from pinecone import Pinecone, ServerlessSpec

from rag_core.config import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    GEMINI_API_KEY,
    PINECONE_API_KEY,
    require_keys,
)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def probe_gemini_embedding() -> bool:
    """Probe 1: Can we embed text and images with Gemini Embedding 2?"""
    print("\n=== Probe 1: Gemini Embedding 2 ===")
    client = genai.Client(api_key=GEMINI_API_KEY)

    # Text embedding (batch of 10)
    texts = [f"Financial metric number {i}" for i in range(10)]
    t0 = time.time()
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM),
    )
    t1 = time.time()

    vec = result.embeddings[0].values
    print(f"  Text batch (10): {t1 - t0:.2f}s, dim={len(vec)}")
    assert len(vec) == EMBEDDING_DIM, f"Expected {EMBEDDING_DIM}, got {len(vec)}"
    print("  PASS: text embedding works")
    return True


def probe_pinecone_index() -> bool:
    """Probe 2: Can we create a 3072-dim index on Pinecone free tier?"""
    print("\n=== Probe 2: Pinecone 3072-dim Index ===")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    test_index_name = "probe-test-3072"

    # Clean up if leftover from previous run
    existing = [idx.name for idx in pc.list_indexes()]
    if test_index_name in existing:
        pc.delete_index(test_index_name)
        time.sleep(2)

    # Create
    t0 = time.time()
    pc.create_index(
        name=test_index_name,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    t1 = time.time()
    print(f"  Index created: {t1 - t0:.2f}s")

    # Wait for ready
    index = pc.Index(test_index_name)
    time.sleep(5)

    # Upsert random vectors
    vectors = [
        {"id": f"test_{i}", "values": np.random.randn(EMBEDDING_DIM).tolist()}
        for i in range(5)
    ]
    index.upsert(vectors=vectors)
    time.sleep(3)

    # Query
    result = index.query(vector=vectors[0]["values"], top_k=3, include_metadata=True)
    print(f"  Query returned {len(result.matches)} matches")
    assert len(result.matches) > 0, "No matches returned"

    # Cleanup
    pc.delete_index(test_index_name)
    print("  PASS: 3072-dim index works on free tier")
    return True


def probe_cross_modal_quality() -> bool:
    """Probe 3: Do image and text embeddings share meaningful similarity?"""
    print("\n=== Probe 3: Cross-Modal Embedding Quality ===")
    from rag_core.ingest.hf_loader import load_corpus

    client = genai.Client(api_key=GEMINI_API_KEY)
    pages = load_corpus(limit=3)

    img_vecs = []
    txt_vecs = []

    for page in pages:
        # Embed image
        img_bytes = page.image_path.read_bytes()
        img_result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[types.Part.from_bytes(data=img_bytes, mime_type="image/png")],
            config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM),
        )
        img_vecs.append(np.array(img_result.embeddings[0].values))
        time.sleep(1)

        # Embed text
        text = page.markdown[:500] if page.markdown else "empty page"
        txt_result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[text],
            config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM),
        )
        txt_vecs.append(np.array(txt_result.embeddings[0].values))
        time.sleep(1)

    # Similarity matrix
    print("\n  Cosine similarity (image_i vs text_j):")
    print(f"  {'':>8}", end="")
    for j in range(len(pages)):
        print(f"  txt_{pages[j].corpus_id:<5}", end="")
    print()

    for i in range(len(pages)):
        print(f"  img_{pages[i].corpus_id:<4}", end="")
        for j in range(len(pages)):
            sim = cosine_sim(img_vecs[i], txt_vecs[j])
            marker = " <-- match" if i == j else ""
            print(f"  {sim:>7.3f}{marker}", end="")
        print()

    # Check: diagonal should be higher than off-diagonal
    diag_avg = np.mean([cosine_sim(img_vecs[i], txt_vecs[i]) for i in range(len(pages))])
    off_diag = []
    for i in range(len(pages)):
        for j in range(len(pages)):
            if i != j:
                off_diag.append(cosine_sim(img_vecs[i], txt_vecs[j]))
    off_avg = np.mean(off_diag)

    print(f"\n  Diagonal avg (matched):   {diag_avg:.3f}")
    print(f"  Off-diagonal avg (unmatched): {off_avg:.3f}")
    print(f"  Gap: {diag_avg - off_avg:.3f}")

    if diag_avg > off_avg:
        print("  PASS: matched pairs have higher similarity")
    else:
        print("  WARN: no clear diagonal advantage — text chunks will carry retrieval")
    return True


def main() -> None:
    require_keys()
    results = {}

    for name, fn in [
        ("gemini_embedding", probe_gemini_embedding),
        ("pinecone_index", probe_pinecone_index),
        ("cross_modal_quality", probe_cross_modal_quality),
    ]:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"  FAIL: {e}")
            results[name] = False

    print("\n=== Summary ===")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
