# DSAI 413 — Multi-Modal RAG (Gemini + Pinecone)

Unified multimodal RAG over financial document pages.
- Embedding: Gemini Embedding 2 (text + images → 3072-dim unified space, via API)
- Vector store: Pinecone serverless (single index, cosine)
- Generation: Gemini 2.0 Flash (via API)
- Dataset: vidore/vidore_v3_finance_en (HuggingFace)

Full architecture in @docs/DESIGN.md — read before planning changes.

## Hardware
- MacBook Air M1, 8 GB. NO local model inference.
- All embedding, retrieval, and generation runs via cloud APIs.
- No torch, no transformers, no local vector DB.

## Dependencies (minimal)
- `google-genai>=1.0` — Gemini Embedding 2 + Gemini Flash
- `pinecone>=5.0` — Pinecone serverless
- `datasets>=3.0` — HuggingFace
- `pillow>=10` — image handling
- `streamlit>=1.38` — UI
- `python-dotenv>=1.0` — env vars
- `pytrec-eval-terrier>=0.5` — eval metrics
- Use `uv` for env management.

## Commands
- `uv sync` — install deps
- `uv run make index` — full pipeline: load HF → parse → chunk → embed → upsert to Pinecone
- `uv run make demo` — Streamlit on :8501
- `uv run make eval SYSTEM=unified` — eval one system
- `uv run pytest tests/ -x -q`

## Dataset
- `vidore/vidore_v3_finance_en` on HuggingFace (CC BY 4.0)
- Load with `data_dir=`: corpus, queries, qrels, documents_metadata, pdfs
- corpus: `corpus_id`, `image` (PIL), `doc_id`, `markdown`, `page_number_in_doc`
- queries: `query_id`, `query`, `answer`, `source_type`
- qrels: `query_id`, `corpus_id`, `score` (1/2), `content_type`, `bounding_boxes`

## Vector store: Pinecone
- ONE index: `multimodal-rag`, dimension=3072, metric=cosine
- Contains BOTH page-image embeddings AND text-chunk embeddings
- Each record has metadata:
  `record_type`: "page_image" | "text_chunk"
  `modality`: "" (images) | "text" | "table" | "chart_metadata" | "footnote"
  `corpus_id`, `doc_id`, `page_number`, `chunk_text`, `image_path`
- For eval configs, use Pinecone metadata filter:
  text_only: filter={"record_type": "text_chunk"}
  image_only: filter={"record_type": "page_image"}
  unified: no filter

## Embedding: Gemini Embedding 2
- Model: `gemini-embedding-2-preview`
- Dimension: 3072 (full quality — do NOT reduce unless I say so)
- Text: pass string to embed_content
- Images: pass as `types.Part.from_bytes(data=bytes, mime_type="image/png")`
- Rate limit: batch embedding calls in groups of 10, sleep 1s between batches
- Cache all embeddings to `data/embeddings/` as .npy files for resume
- If cache exists, load from disk — do NOT re-call the API

## Multi-modal parsing
- Source: `markdown` field in corpus. No Tesseract/pdfplumber/OCR.
- Tables: lines with `|` and `---`. Never split mid-row. Keep delimiters intact.
- Chart metadata: "Figure X", "Chart X", "Graph X", captions, alt-text.
- Footnotes: superscript `¹²³`, `[1]`, `1/`, `*` and their text.
- Modalities: "text" | "table" | "chart_metadata" | "footnote"

## Code style
- Python 3.11+, type hints, `from __future__ import annotations`.
- Small modules (<200 lines). pathlib.Path. Never bare-except.
- Dataclasses or pydantic for cross-module structures.

## Workflow
- Commit after every subtask.
- Run module test before moving on.
- Confirm before full-corpus indexing (API calls use quota).

## Things to NEVER do
- Install torch, transformers, or any local ML framework.
- Call Gemini/Pinecone APIs from inside tests. Mock everything.
- Re-index full corpus for small changes. Use 50-page slice.
- Commit data/pages/ or data/embeddings/.

## Secrets (.env, gitignored)
- `GEMINI_API_KEY` — for embedding + generation
- `PINECONE_API_KEY` — for vector store
- Fail fast with clear message if either is missing.