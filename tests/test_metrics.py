"""Tests for eval/metrics.py — pure computation, no API calls."""

from __future__ import annotations

from rag_core.eval.metrics import compute_metrics, per_content_type_metrics


def test_perfect_recall_at_1():
    """When top result is the only relevant doc, recall@1 = 1.0."""
    run = {"q1": {"doc1": 0.9, "doc2": 0.5}}
    qrels = {"q1": {"doc1": 1}}

    metrics = compute_metrics(run, qrels)
    assert metrics["recall_1"] == 1.0


def test_zero_recall_at_1():
    """When top result is not relevant, recall@1 = 0.0."""
    run = {"q1": {"doc2": 0.9, "doc1": 0.5}}
    qrels = {"q1": {"doc1": 1}}

    metrics = compute_metrics(run, qrels)
    assert metrics["recall_1"] == 0.0


def test_recall_at_5():
    """Relevant doc in top 5 should give recall@5 = 1.0."""
    run = {"q1": {"d1": 0.9, "d2": 0.8, "d3": 0.7, "d4": 0.6, "doc_rel": 0.5}}
    qrels = {"q1": {"doc_rel": 1}}

    metrics = compute_metrics(run, qrels)
    assert metrics["recall_5"] == 1.0


def test_mrr():
    """MRR = 1/rank of first relevant result."""
    # Relevant doc at position 2
    run = {"q1": {"d1": 0.9, "doc_rel": 0.8, "d3": 0.7}}
    qrels = {"q1": {"doc_rel": 1}}

    metrics = compute_metrics(run, qrels)
    assert metrics["recip_rank"] == 0.5  # 1/2


def test_multiple_queries_averaged():
    """Metrics should be averaged across queries."""
    run = {
        "q1": {"doc1": 0.9},  # recall@1 = 1.0
        "q2": {"doc3": 0.9, "doc2": 0.5},  # recall@1 = 0.0
    }
    qrels = {
        "q1": {"doc1": 1},
        "q2": {"doc2": 1},
    }

    metrics = compute_metrics(run, qrels)
    assert metrics["recall_1"] == 0.5  # (1.0 + 0.0) / 2


def test_empty_run():
    metrics = compute_metrics({}, {})
    assert all(v == 0.0 for v in metrics.values())


def test_per_content_type():
    run = {
        "q1": {"doc1": 0.9},
        "q2": {"doc2": 0.9},
    }
    qrels_raw = [
        {"query_id": "q1", "corpus_id": "doc1", "score": 1, "content_type": "table"},
        {"query_id": "q2", "corpus_id": "doc2", "score": 1, "content_type": "text"},
    ]

    result = per_content_type_metrics(run, qrels_raw)
    assert "table" in result
    assert "text" in result
    assert result["table"]["recall_1"] == 1.0
    assert result["text"]["recall_1"] == 1.0


def test_per_content_type_list():
    """content_type can be a list (e.g. from HuggingFace qrels)."""
    run = {"q1": {"doc1": 0.9}}
    qrels_raw = [
        {"query_id": "q1", "corpus_id": "doc1", "score": 1, "content_type": ["table", "text"]},
    ]

    result = per_content_type_metrics(run, qrels_raw)
    assert "table" in result
    assert "text" in result
    assert result["table"]["recall_1"] == 1.0
    assert result["text"]["recall_1"] == 1.0
