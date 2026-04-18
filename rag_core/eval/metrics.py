"""Retrieval metrics using pytrec_eval: Recall@k, MRR@10, nDCG@5."""

from __future__ import annotations

import pytrec_eval


MEASURES = {"recall_1", "recall_5", "recip_rank", "ndcg_cut_5"}


def compute_metrics(
    run_dict: dict[str, dict[str, float]],
    qrels_dict: dict[str, dict[str, int]],
) -> dict[str, float]:
    """Compute retrieval metrics across all queries.

    Args:
        run_dict: {query_id: {corpus_id_str: score}}
        qrels_dict: {query_id: {corpus_id_str: relevance_int}}

    Returns:
        Dict with averaged metrics: recall_1, recall_5, recip_rank, ndcg_cut_5.
    """
    evaluator = pytrec_eval.RelevanceEvaluator(qrels_dict, MEASURES)
    results = evaluator.evaluate(run_dict)

    # Average across queries
    agg: dict[str, float] = {}
    n = len(results)
    if n == 0:
        return {m: 0.0 for m in MEASURES}

    for metric in MEASURES:
        agg[metric] = sum(results[qid][metric] for qid in results) / n

    return agg


def per_content_type_metrics(
    run_dict: dict[str, dict[str, float]],
    qrels: list[dict],
) -> dict[str, dict[str, float]]:
    """Compute metrics broken down by content_type from qrels.

    Args:
        run_dict: {query_id: {corpus_id_str: score}}
        qrels: Raw qrels list of dicts with query_id, corpus_id, score, content_type.

    Returns:
        {content_type: {metric: value}} for each content_type in the data.
    """
    # Group qrels by content_type
    ct_qrels: dict[str, dict[str, dict[str, int]]] = {}
    ct_queries: dict[str, set[str]] = {}

    for row in qrels:
        raw_ct = row.get("content_type", "unknown")
        qid = str(row["query_id"])
        cid = str(row["corpus_id"])
        rel = int(row["score"])

        # content_type may be a list (e.g. ["table", "text"]) — expand to one entry per type
        if isinstance(raw_ct, list):
            types = raw_ct if raw_ct else ["unknown"]
        else:
            types = [raw_ct]

        for ct in types:
            if ct not in ct_qrels:
                ct_qrels[ct] = {}
                ct_queries[ct] = set()
            if qid not in ct_qrels[ct]:
                ct_qrels[ct][qid] = {}
            ct_qrels[ct][qid][cid] = rel
            ct_queries[ct].add(qid)

    # Compute metrics per content_type
    results: dict[str, dict[str, float]] = {}
    for ct, qrels_for_ct in ct_qrels.items():
        # Filter run_dict to only queries in this content_type
        run_for_ct = {qid: run_dict.get(qid, {}) for qid in ct_queries[ct] if qid in run_dict}
        if run_for_ct:
            results[ct] = compute_metrics(run_for_ct, qrels_for_ct)

    return results
