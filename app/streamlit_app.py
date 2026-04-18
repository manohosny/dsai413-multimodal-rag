"""Streamlit demo UI for the multimodal RAG system."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _safe_md(text: str) -> str:
    """Escape `$` so Streamlit doesn't interpret dollar amounts as LaTeX delimiters."""
    return text.replace("$", r"\$")


_SEP_CHARS = set("-:| \t")


def _parse_pipe_table(raw: str) -> pd.DataFrame | None:
    """Parse a pipe-delimited markdown-ish table into a DataFrame.

    Strips outer empty columns, drops GFM separator rows, pads uneven rows.
    Returns None if there are fewer than 2 rows or 2 columns of usable data.
    """
    rows: list[list[str]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        if set(line) <= _SEP_CHARS:
            continue
        cells = [c.strip() for c in line.split("|")]
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]
        if cells:
            rows.append(cells)

    if len(rows) < 2:
        return None
    width = max(len(r) for r in rows)
    if width < 2:
        return None

    padded = [r + [""] * (width - len(r)) for r in rows]
    seen: dict[str, int] = {}
    header: list[str] = []
    for i, h in enumerate(padded[0]):
        label = h or f"col_{i}"
        if label in seen:
            seen[label] += 1
            label = f"{label}_{seen[label]}"
        else:
            seen[label] = 0
        header.append(label)
    return pd.DataFrame(padded[1:], columns=header)


st.set_page_config(page_title="Multimodal RAG — Financial Docs", layout="wide")


@st.cache_resource
def get_pipeline(system: str):
    from rag_core.pipeline.router import RAGPipeline

    return RAGPipeline(system=system)


# --- Sidebar ---
st.sidebar.title("Settings")
system = st.sidebar.selectbox(
    "Retrieval system",
    ["unified", "text_only", "image_only"],
    index=0,
    help="unified = text + images, text_only = text chunks only, image_only = page images only",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Architecture:** Gemini Embedding 2 → Pinecone → Gemini 3.0 Flash"
)

# --- Main ---
st.title("Multi-Modal Document QA")
st.caption("Ask questions about financial documents. Powered by Gemini + Pinecone.")

query = st.text_input("Your question:", placeholder="e.g., What was the total revenue in 2023?")

if st.button("Ask", type="primary") and query:
    with st.spinner("Retrieving and generating..."):
        pipeline = get_pipeline(system)
        result = pipeline.answer(query)

    # --- Answer ---
    st.subheader("Answer")
    st.markdown(_safe_md(result["answer"]))

    # --- Retrieved pages ---
    st.subheader("Retrieved Pages")
    cols = st.columns(min(len(result["page_corpus_ids"]), 3))
    for i, cid in enumerate(result["page_corpus_ids"][:3]):
        with cols[i]:
            # Find image path from retrieval results
            img_path = None
            for r in result["retrieval_results"]:
                if r.corpus_id == cid and r.image_path:
                    img_path = _PROJECT_ROOT / r.image_path
                    break
            # Fallback: construct path from corpus_id
            if img_path is None or not img_path.exists():
                img_path = _PROJECT_ROOT / "data" / "pages" / f"{cid:05d}.png"
            if img_path.exists():
                st.image(str(img_path), caption=f"Page {cid}", use_container_width=True)
            else:
                st.info(f"Page {cid} (image not found)")

    # --- Text chunks ---
    st.subheader("Matched Text Chunks")
    text_results = [
        r for r in result["retrieval_results"] if r.record_type == "text_chunk" and r.chunk_text
    ]
    if text_results:
        for r in text_results[:5]:
            badge = f"`{r.modality}`" if r.modality else "`text`"
            st.markdown(f"**{badge}** (page {r.corpus_id}, score {r.score:.3f})")
            chunk = r.chunk_text.strip()
            looks_tabular = chunk.startswith("|") and chunk.count("|") > 10
            if r.modality == "table" or looks_tabular:
                df = _parse_pipe_table(chunk)
                if df is not None:
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.code(chunk, language="markdown")
            else:
                snippet = chunk[:500] + ("…" if len(chunk) > 500 else "")
                st.text(snippet)
            st.markdown("---")
    else:
        st.info("No text chunks retrieved for this query.")
