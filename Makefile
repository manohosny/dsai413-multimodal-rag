.PHONY: index demo eval test probe

SYSTEM ?= unified

index:
	uv run python -m rag_core.ingest.indexer

demo:
	uv run streamlit run app/streamlit_app.py --server.port 8501

eval:
	uv run python -m rag_core.eval.vidore_eval --system $(SYSTEM)

test:
	uv run pytest tests/ -x -q

probe:
	uv run python scripts/day1_probe.py
