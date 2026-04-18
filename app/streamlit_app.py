"""Streamlit demo UI for the multimodal RAG system."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

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
    st.markdown(result["answer"])

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
            st.text(r.chunk_text[:500])
            st.markdown("---")
    else:
        st.info("No text chunks retrieved for this query.")
