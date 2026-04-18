# Multi-Modal Document QA (RAG)

Unified multimodal RAG system for financial document question answering.
Embeds both page images and parsed text chunks into a single vector space using **Gemini Embedding 2**, stores them in **Pinecone serverless**, and generates answers with **Gemini 3.0 Flash**.

**Dataset:** [vidore/vidore_v3_finance_en](https://huggingface.co/datasets/vidore/vidore_v3_finance_en) — 2,942 pages, 309 queries, 8,766 qrels.

## Architecture

```
HuggingFace Dataset
    ↓
[hf_loader] → page images + markdown text
    ↓
[multimodal_parser] → text / table / chart_metadata / footnote segments
    ↓
[chunker] → modality-aware chunks
    ↓
[gemini_embedder] → 3072-dim vectors (Gemini Embedding 2)
    ↓
[pinecone_store] → single Pinecone index (cosine)
    ↓
[retriever] → embed query → search → group by page → top-3
    ↓
[gemini_generator] → Gemini 3.0 Flash → answer + citations
```

## Quick Start

```bash
# 1. Clone and setup
git clone <repo-url> && cd dsai413-multimodal-rag
cp .env.example .env   # Add your GEMINI_API_KEY and PINECONE_API_KEY

# 2. Install dependencies
uv sync

# 3. Index the corpus (one-time, ~30-60 min)
uv run make index

# 4. Launch the demo
uv run make demo        # Opens Streamlit on :8501

# 5. Run evaluation
uv run make eval SYSTEM=unified
uv run make eval SYSTEM=text_only
uv run make eval SYSTEM=image_only
```

## Three Evaluation Systems

| System | Pinecone Filter | What it tests |
|--------|----------------|---------------|
| **text_only** | `record_type = text_chunk` | Text retrieval only |
| **image_only** | `record_type = page_image` | Visual retrieval only |
| **unified** | No filter | Full system (all records) |

## Evaluation Results (100 queries)

| Metric | unified | text_only | image_only |
|--------|---------|-----------|-----------|
| Recall@1 | 0.208 | 0.208 | 0.189 |
| Recall@5 | 0.304 | 0.304 | **0.338** |
| MRR | 0.563 | 0.563 | 0.563 |
| nDCG@5 | 0.385 | 0.385 | 0.385 |
| Faithfulness | 0.575 | 0.593 | **0.688** |

**Key finding:** `unified` and `text_only` produce identical retrieval metrics — cosine ranking over the shared space consistently puts text chunks above page images, so the "unified" top-20 is always text-only. `image_only` wins on tables (+69% Recall@5) and faithfulness (+19%). Full analysis in [report/technical_report.md](report/technical_report.md).

## Commands

| Command | Description |
|---------|-------------|
| `uv run make index` | Full indexing pipeline |
| `uv run make demo` | Streamlit UI on :8501 |
| `uv run make eval SYSTEM=unified` | Evaluate one system |
| `uv run make test` | Run all tests |
| `uv run make probe` | Day 1 risk probes |

## Dependencies

- `google-genai` — Gemini Embedding 2 + Gemini 3.0 Flash
- `pinecone` — Pinecone serverless
- `datasets` — HuggingFace
- `pillow` — Image handling
- `streamlit` — Demo UI
- `pytrec-eval-terrier` — Retrieval metrics

No torch. No transformers. Two API keys. Runs on 8 GB RAM.
