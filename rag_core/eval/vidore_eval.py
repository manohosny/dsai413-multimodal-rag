"""Benchmark runner: evaluate RAG pipeline over ViDoRe queries."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from rag_core.config import require_keys
from rag_core.eval.judge import LLMJudge
from rag_core.eval.metrics import compute_metrics, per_content_type_metrics
from rag_core.ingest.hf_loader import load_qrels, load_queries
from rag_core.pipeline.router import RAGPipeline

logger = logging.getLogger(__name__)


class VidoreEvaluator:
    """Run the RAG pipeline over benchmark queries and compute metrics."""

    def __init__(self, system: str, limit: int | None = None) -> None:
        self.system = system
        self.limit = limit
        self._pipeline = RAGPipeline(system=system)

    def run(self) -> dict:
        """Execute evaluation.

        Returns dict with:
            - system: str
            - run_dict: {query_id: {corpus_id: score}}
            - answers: {query_id: answer_text}
            - metrics: {metric_name: value}
            - per_content_type: {content_type: {metric: value}}
            - faithfulness: float (mean)
        """
        queries = load_queries()
        qrels_raw = load_qrels()

        if self.limit:
            queries = queries[: self.limit]

        # Build qrels dict for pytrec_eval
        qrels_dict: dict[str, dict[str, int]] = {}
        for row in qrels_raw:
            qid = str(row["query_id"])
            cid = str(row["corpus_id"])
            if qid not in qrels_dict:
                qrels_dict[qid] = {}
            qrels_dict[qid][cid] = int(row["score"])

        # Run pipeline for each query
        run_dict: dict[str, dict[str, float]] = {}
        answers: dict[str, str] = {}
        judge_items: list[dict] = []

        for i, q in enumerate(queries):
            qid = str(q["query_id"])
            query_text = q["query"]
            ground_truth = q.get("answer", "")

            logger.info("Query %d/%d [%s]: %s", i + 1, len(queries), qid, query_text[:80])

            result = self._pipeline.answer(query_text)
            answers[qid] = result["answer"]

            # Build run entry from retrieval scores
            run_dict[qid] = {}
            for r in result["retrieval_results"]:
                cid = str(r.corpus_id)
                # Keep max score per corpus_id
                if cid not in run_dict[qid] or r.score > run_dict[qid][cid]:
                    run_dict[qid][cid] = r.score

            if ground_truth:
                judge_items.append(
                    {
                        "query": query_text,
                        "answer": result["answer"],
                        "ground_truth": ground_truth,
                    }
                )

        # Compute metrics
        # Only evaluate queries that are in qrels
        eval_run = {qid: run_dict[qid] for qid in run_dict if qid in qrels_dict}
        eval_qrels = {qid: qrels_dict[qid] for qid in eval_run if qid in qrels_dict}

        metrics = compute_metrics(eval_run, eval_qrels) if eval_run else {}
        ct_metrics = per_content_type_metrics(eval_run, qrels_raw) if eval_run else {}

        # Judge faithfulness
        faithfulness = 0.0
        if judge_items:
            logger.info("Running LLM-as-judge on %d answers...", len(judge_items))
            judge = LLMJudge()
            faithfulness = judge.batch_judge(judge_items)

        return {
            "system": self.system,
            "num_queries": len(queries),
            "run_dict": run_dict,
            "answers": answers,
            "metrics": metrics,
            "per_content_type": ct_metrics,
            "faithfulness": faithfulness,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG system")
    parser.add_argument(
        "--system",
        choices=["text_only", "image_only", "unified"],
        default="unified",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    require_keys()
    evaluator = VidoreEvaluator(system=args.system, limit=args.limit)
    results = evaluator.run()

    # Print summary
    print(f"\n=== {args.system} ===")
    for k, v in results["metrics"].items():
        print(f"  {k}: {v:.4f}")
    print(f"  faithfulness: {results['faithfulness']:.4f}")

    if results["per_content_type"]:
        print("\nPer content type:")
        for ct, m in results["per_content_type"].items():
            print(f"  {ct}:")
            for k, v in m.items():
                print(f"    {k}: {v:.4f}")

    # Save to file
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Convert non-serializable items
        output = {
            "system": results["system"],
            "num_queries": results["num_queries"],
            "metrics": results["metrics"],
            "per_content_type": results["per_content_type"],
            "faithfulness": results["faithfulness"],
        }
        out_path.write_text(json.dumps(output, indent=2))
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
