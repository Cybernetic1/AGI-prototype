#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def _safe_ratio(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def evaluate(worked_examples: dict[str, Any], rule_alloc: dict[str, Any]) -> dict[str, Any]:
    examples = list(worked_examples.get("examples", []) or [])
    rule_shapes = list(rule_alloc.get("rule_shapes", []) or [])
    available = {str(s.get("shape")) for s in rule_shapes if int(s.get("count", 0)) > 0}

    category_stats: dict[str, dict[str, int]] = {}
    completed = dead_end = contradictions = 0
    for ex in examples:
        category = str(ex.get("category", "unknown"))
        stats = category_stats.setdefault(
            category,
            {"total": 0, "completed": 0, "dead_end": 0, "contradiction": 0, "shape_covered": 0},
        )
        stats["total"] += 1

        required = {str(s) for s in (ex.get("required_shapes") or [])}
        trace = list(ex.get("trace") or [])
        target = str(ex.get("target") or "")
        negatives = {str(n) for n in (ex.get("semantic_ar_tuple", {}).get("hard_negatives") or [])}

        shape_ok = required.issubset(available)
        trace_ok = len(trace) >= 3 and any(str(step.get("step")) == "infer" for step in trace if isinstance(step, dict))
        contradiction = target in negatives
        chain_ok = bool(shape_ok and trace_ok and not contradiction)

        if shape_ok:
            stats["shape_covered"] += 1
        if not trace_ok:
            dead_end += 1
            stats["dead_end"] += 1
        if contradiction:
            contradictions += 1
            stats["contradiction"] += 1
        if chain_ok:
            completed += 1
            stats["completed"] += 1

    per_category = {}
    for cat, s in sorted(category_stats.items()):
        per_category[cat] = {
            "total": s["total"],
            "chain_success_rate": _safe_ratio(s["completed"], s["total"]),
            "dead_end_ratio": _safe_ratio(s["dead_end"], s["total"]),
            "contradiction_ratio": _safe_ratio(s["contradiction"], s["total"]),
            "shape_coverage_rate": _safe_ratio(s["shape_covered"], s["total"]),
        }

    total = len(examples)
    chain_success = _safe_ratio(completed, total)
    dead_end_ratio = _safe_ratio(dead_end, total)
    contradiction_ratio = _safe_ratio(contradictions, total)
    coverage = _safe_ratio(sum(s["shape_covered"] for s in category_stats.values()), total)

    thresholds = {
        "min_chain_success_rate": 0.85,
        "max_dead_end_ratio": 0.10,
        "max_contradiction_ratio": 0.05,
        "min_shape_coverage_rate": 0.95,
    }
    gates = {
        "chain_success": chain_success >= thresholds["min_chain_success_rate"],
        "dead_end": dead_end_ratio <= thresholds["max_dead_end_ratio"],
        "contradiction": contradiction_ratio <= thresholds["max_contradiction_ratio"],
        "shape_coverage": coverage >= thresholds["min_shape_coverage_rate"],
    }

    return {
        "artifact_id": "chain-completeness-eval-v1",
        "generated_at": _utc_now(),
        "sources": {
            "worked_examples_id": worked_examples.get("artifact_id"),
            "rule_allocation_id": rule_alloc.get("artifact_id"),
        },
        "summary": {
            "total_examples": total,
            "chain_success_rate": chain_success,
            "dead_end_ratio": dead_end_ratio,
            "contradiction_ratio": contradiction_ratio,
            "shape_coverage_rate": coverage,
        },
        "thresholds": thresholds,
        "gates": gates,
        "all_gates_green": all(gates.values()),
        "per_category": per_category,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate chain completeness for semantic-AR worked examples.")
    parser.add_argument("--worked-examples", default="rules/reports/semantic_ar_worked_examples_v1.json")
    parser.add_argument("--rule-allocation", default="rules/reports/rule_shape_allocation_300m_v1.json")
    parser.add_argument("--output", default="rules/reports/chain_completeness_eval_v1.json")
    args = parser.parse_args()

    payload = evaluate(
        _load_json(Path(args.worked_examples)),
        _load_json(Path(args.rule_allocation)),
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved chain completeness evaluation to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
