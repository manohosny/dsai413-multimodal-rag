"""Compare evaluation results from multiple systems and generate a markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare_systems(result_files: list[Path]) -> str:
    """Load result JSON files and produce a markdown comparison table.

    Returns markdown string.
    """
    results = []
    for f in result_files:
        data = json.loads(f.read_text())
        results.append(data)

    if not results:
        return "No results to compare."

    # --- Main metrics table ---
    metrics_keys = ["recall_1", "recall_5", "recip_rank", "ndcg_cut_5", "faithfulness"]
    header = "| Metric | " + " | ".join(r["system"] for r in results) + " |"
    sep = "|---|" + "|".join("---" for _ in results) + "|"

    rows = [header, sep]
    for key in metrics_keys:
        row = f"| {key} |"
        for r in results:
            if key == "faithfulness":
                val = r.get("faithfulness", 0.0)
            else:
                val = r.get("metrics", {}).get(key, 0.0)
            row += f" {val:.4f} |"
        rows.append(row)

    md = "## System Comparison\n\n" + "\n".join(rows) + "\n"

    # --- Per content-type breakdown ---
    all_types: set[str] = set()
    for r in results:
        all_types.update(r.get("per_content_type", {}).keys())

    if all_types:
        md += "\n## Per Content-Type Breakdown\n"
        for ct in sorted(all_types):
            md += f"\n### {ct}\n\n"
            ct_header = "| Metric | " + " | ".join(r["system"] for r in results) + " |"
            ct_sep = "|---|" + "|".join("---" for _ in results) + "|"
            ct_rows = [ct_header, ct_sep]

            for key in ["recall_1", "recall_5", "recip_rank", "ndcg_cut_5"]:
                row = f"| {key} |"
                for r in results:
                    val = r.get("per_content_type", {}).get(ct, {}).get(key, 0.0)
                    row += f" {val:.4f} |"
                ct_rows.append(row)

            md += "\n".join(ct_rows) + "\n"

    return md


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare system evaluation results")
    parser.add_argument("files", nargs="+", type=Path, help="Result JSON files")
    parser.add_argument("--md", action="store_true", help="Output as markdown")
    args = parser.parse_args()

    md = compare_systems(args.files)
    print(md)


if __name__ == "__main__":
    main()
