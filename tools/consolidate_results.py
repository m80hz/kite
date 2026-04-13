#!/usr/bin/env python3
"""
Consolidate KITE evaluation results across tasks by question type.

Given a results_merged.json (keys like "InsertCylinder/Task identification"),
this script computes weighted averages per question type across all tasks,
weighting each entry by its num_qa.

Output contains, per question type:
- num_qa_total: total number of QA pairs aggregated
- metrics: weighted averages for metric-style entries (e.g., exact_match, token_f1, ...)
- score_overall: weighted average for normalized score entries

Usage:
  python tools/consolidate_results.py --input path/to/results_merged.json --format json
  python tools/consolidate_results.py -i results_merged.json -o consolidated.json
  python tools/consolidate_results.py -i results_merged.json --format table
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, Tuple


def _round_floats(obj: Any, ndigits: int | None) -> Any:
    """Recursively round floats in nested structures if ndigits is provided."""
    if ndigits is None:
        return obj
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, ndigits) for v in obj]
    return obj


def consolidate_results(data: Dict[str, Any]) -> Dict[str, Any]:
    """Consolidate metrics and scores by question type weighted by num_qa.

    Parameters
    ----------
    data : Dict[str, Any]
        Parsed JSON from results_merged.json.

    Returns
    -------
    Dict[str, Any]
        Consolidated dictionary keyed by question type with aggregated values.
    """
    # Aggregators
    num_qa_total: Dict[str, int] = defaultdict(int)
    # For metrics fields: per question-type, per metric -> weighted sum and denom
    metrics_weighted_sum: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    metrics_weighted_den: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    # For score_overall fields
    score_weighted_sum: Dict[str, float] = defaultdict(float)
    score_weighted_den: Dict[str, int] = defaultdict(int)

    for key, entry in data.items():
        if not isinstance(entry, dict):
            continue
        if "/" not in key:
            # Unexpected; skip
            continue
        try:
            _, question_type = key.split("/", 1)
        except ValueError:
            continue

        n = int(entry.get("num_qa", 0) or 0)
        if n <= 0:
            # Skip entries without questions
            continue

        num_qa_total[question_type] += n

        # Metrics style
        metrics = entry.get("metrics")
        if isinstance(metrics, dict):
            for mname, mval in metrics.items():
                try:
                    fv = float(mval)
                except Exception:
                    continue
                metrics_weighted_sum[question_type][mname] += fv * n
                metrics_weighted_den[question_type][mname] += n

        # score_overall style
        if "score_overall" in entry:
            try:
                score = float(entry["score_overall"])
            except Exception:
                score = None
            if score is not None:
                score_weighted_sum[question_type] += score * n
                score_weighted_den[question_type] += n

    # Build consolidated output
    consolidated: Dict[str, Any] = {}
    for qtype in sorted(set(num_qa_total.keys()) | set(score_weighted_den.keys()) | set(metrics_weighted_den.keys())):
        out: Dict[str, Any] = {"num_qa_total": int(num_qa_total.get(qtype, 0))}

        # Metrics averages
        if qtype in metrics_weighted_den:
            metrics_avg: Dict[str, float] = {}
            for mname, denom in metrics_weighted_den[qtype].items():
                if denom > 0:
                    metrics_avg[mname] = metrics_weighted_sum[qtype][mname] / denom
            if metrics_avg:
                # Sort metrics for stable output
                out["metrics"] = {k: metrics_avg[k] for k in sorted(metrics_avg.keys())}

        # Score overall average
        denom_score = score_weighted_den.get(qtype, 0)
        if denom_score > 0:
            out["score_overall"] = score_weighted_sum[qtype] / denom_score

        consolidated[qtype] = out

    return consolidated


def render_table(consolidated: Dict[str, Any], round_ndigits: int | None = 4) -> str:
    """Render a human-readable table string for consolidated results.

    Produces two sections: score_overall and metrics.
    """
    lines: list[str] = []

    # Score-only or score+metrics lines
    score_rows: list[Tuple[str, int, float]] = []
    metrics_blocks: list[Tuple[str, int, Dict[str, float]]] = []

    for qtype, obj in consolidated.items():
        n = int(obj.get("num_qa_total", 0))
        if "score_overall" in obj:
            val = obj["score_overall"]
            if round_ndigits is not None:
                val = round(float(val), round_ndigits)
            score_rows.append((qtype, n, float(val)))
        if "metrics" in obj:
            metrics = obj["metrics"]
            if isinstance(metrics, dict):
                mproc = {k: (round(float(v), round_ndigits) if round_ndigits is not None else float(v)) for k, v in metrics.items()}
                metrics_blocks.append((qtype, n, mproc))

    if score_rows:
        lines.append("Score-overall (weighted):")
        # compute column widths
        name_w = max((len(nm) for nm, _, _ in score_rows), default=4)
        header = f"  {'Question Type'.ljust(name_w)}  |  num_qa  |  score_overall"
        lines.append(header)
        lines.append("  " + ("-" * (len(header) - 2)))
        for nm, n, v in sorted(score_rows, key=lambda x: x[0]):
            lines.append(f"  {nm.ljust(name_w)}  |  {str(n).rjust(6)}  |  {v:>13}")
        lines.append("")

    if metrics_blocks:
        lines.append("Metrics (weighted):")
        for nm, n, metrics in sorted(metrics_blocks, key=lambda x: x[0]):
            lines.append(f"  {nm} (num_qa={n}):")
            for mk, mv in metrics.items():
                lines.append(f"    - {mk}: {mv}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Consolidate KITE results_merged.json by question type (weighted by num_qa)")
    p.add_argument("--input", "-i", required=True, help="Path to results_merged.json (use '-' for stdin)")
    p.add_argument("--output", "-o", help="Optional path to write consolidated results (JSON). If omitted, prints to stdout.")
    p.add_argument("--format", "-f", choices=["json", "table"], default="json", help="Output format when printing to stdout (default: json)")
    p.add_argument("--round", type=int, default=6, help="Round floats to N digits (use -1 to disable)")

    args = p.parse_args(list(argv) if argv is not None else None)

    # Load input
    if args.input == "-":
        loaded = json.load(sys.stdin)
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            loaded = json.load(f)

    consolidated = consolidate_results(loaded)

    round_ndigits = None if args.round is not None and args.round < 0 else args.round

    if args.output:
        to_write = _round_floats(consolidated, round_ndigits)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(to_write, f, ensure_ascii=False, indent=2)
        return 0

    # Print to stdout
    if args.format == "json":
        print(json.dumps(_round_floats(consolidated, round_ndigits), ensure_ascii=False, indent=2))
    else:
        print(render_table(consolidated, round_ndigits=round_ndigits if isinstance(round_ndigits, int) else 4), end="")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
