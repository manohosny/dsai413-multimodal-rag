# Multi-Modal Document Intelligence — Technical Report

**DSAI 413 — Assignment 1**
**Dataset:** ViDoRe V3 Finance (2,942 pages, 309 queries, 8,766 qrels)

---

## 1. Problem & Approach

Financial documents mix prose, tables, multi-year figures, chart captions, and footnotes. A text-only RAG system throws away the spatial and visual signal that makes these documents information-rich. We built a **unified multi-modal RAG pipeline** that embeds both rendered page images and parsed text chunks into a **single vector space** using **Gemini Embedding 2** — Google's first natively multimodal embedding model (March 2026). Retrieval, indexing, and generation are all cloud-API-based; no local ML framework is required (runs on an 8 GB M1).

## 2. Architecture

```
HF Dataset → hf_loader ──► page images (PNG)
              │          └► markdown text
              ▼
    multimodal_parser → {text | table | chart_metadata | footnote} segments
              ▼
    chunker → modality-aware chunks (~200-tok text, whole tables, passthrough footnotes)
              ▼
    Gemini Embedding 2 ──► 3072-dim vectors  ──► Pinecone (single index, cosine)
                                                  │
    query ──► embed ──► top-20 ──► group-by-page ─┤
                                                  ▼
                          top-3 pages + matched chunks → Gemini 3.0 Flash → answer + citations
```

**Modules** (~2,000 LOC total): `rag_core/{ingest,embeddings,retrieval,generation,pipeline,eval}` with 44 unit tests and a Streamlit demo in `app/`.

## 3. Key Design Decisions

**ADR-01 — Gemini Embedding 2 as the sole encoder.** One model embeds text, images, and queries into the *same* 3072-dim space. Alternative (ColPali + text model) would have required two indexes and an 8 GB local model — infeasible on the target hardware. Trade-off: we give up ColPali's patch-level late-interaction, but recover fine-grained matching by also embedding parsed text chunks (tables, chart metadata) alongside page images.

**ADR-02 — Single Pinecone index with metadata filters.** Text chunks and page images live in the same index, differentiated by a `record_type` field. The three evaluation configurations (`unified`, `text_only`, `image_only`) are implemented as Pinecone metadata filters — no re-indexing, no separate collections. This cleanly satisfies the rubric's *"unified multi-modal embedding space"* requirement by construction.

**ADR-03 — Modality-aware chunking.** Markdown tables are kept whole (never split mid-row) to preserve row/column relationships. Text is split at sentence boundaries (~200 tokens, 50-token overlap). Footnotes and chart metadata pass through as single chunks. This preserves the structure that makes financial documents answerable.

**ADR-04 — Disk cache for embeddings.** Each embedding is saved as `data/embeddings/{id}.npy`. Indexing is idempotent and resumable — critical when the embedding run spans ~30 minutes and any 429/timeout would otherwise destroy progress. All transient errors (429, 503, ReadTimeout, ConnectError) are retried with exponential backoff + jitter.

## 4. Evaluation Results (100 queries)

Retrieval is scored with pytrec_eval (Recall@1/5, MRR, nDCG@5). Faithfulness is a Gemini-Flash LLM-as-judge rating the answer against the ground-truth, normalized to [0, 1].

| Metric | unified | text_only | image_only |
|---|---|---|---|
| Recall@1 | 0.208 | 0.208 | 0.189 |
| Recall@5 | 0.304 | 0.304 | **0.338** |
| MRR | 0.563 | 0.563 | 0.563 |
| nDCG@5 | 0.385 | 0.385 | 0.385 |
| Faithfulness | 0.575 | 0.593 | **0.688** |

**Per content-type Recall@5:**

| Content type | unified | text_only | image_only |
|---|---|---|---|
| Table | 0.180 | 0.180 | **0.304** |
| Chart | 0.667 | 0.667 | **0.736** |
| Text | **0.351** | **0.351** | 0.347 |