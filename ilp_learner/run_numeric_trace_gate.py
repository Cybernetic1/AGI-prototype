from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from cognitive_loop import benchmark_single_problem
from forward_engine.phase3_synthetic_test import ontology_classify_details
from ilp_learner.lt_numeric_bridge import extract_numeric_bridge


IN_PATH = Path("logic_transformer/data/gsm8k_train_lt.jsonl")
OUT_PATH = Path("ilp_learner/gsm8k_numeric_trace_gate.json")
SCAN_LIMIT = 220
EVAL_LIMIT = 80
MAX_CYCLES = 2
MIN_CLASS_CONFIDENCE = 0.70


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _closest_belief_error(expected, beliefs):
    exp = _to_float(expected)
    if exp is None:
        return None, None
    nums = [_to_float(v) for v in (beliefs or [])]
    nums = [v for v in nums if v is not None]
    if not nums:
        return None, None
    best = min(nums, key=lambda v: abs(v - exp))
    return abs(best - exp), best


def _infer_expected_operator(text: str, cls_label: str) -> str:
    lower = (text or "").lower()
    if "how much" in lower and "each" in lower:
        return "divide"
    if "how many more" in lower:
        return "subtract"
    if "left" in lower or "remain" in lower or "remaining" in lower:
        return "subtract"
    if "per hour" in lower or "per minute" in lower:
        return "multiply"
    if cls_label in {"transfer", "loss"}:
        return "subtract"
    return "unknown"


def _classify_failure(row):
    if row["solved"]:
        return {
            "category": "solved",
            "score": 0,
            "confidence": 1.0,
            "expected_operator": _infer_expected_operator(row.get("question", ""), row.get("class", "")),
            "closest_belief": row.get("expected"),
            "min_abs_error": 0.0,
            "reason": "Solved",
        }

    missing = list(row.get("bridge_missing") or [])
    signals = row.get("bridge_signals") or {}
    beliefs = row.get("belief_values") or []
    min_err, closest = _closest_belief_error(row.get("expected"), beliefs)
    near_threshold = 1.0
    exp = _to_float(row.get("expected"))
    if exp is not None:
        near_threshold = max(1.0, abs(exp) * 0.1)
    near_miss = min_err is not None and min_err <= near_threshold

    expected_op = _infer_expected_operator(row.get("question", ""), row.get("class", ""))
    culprit = "semantic_mismatch"
    score = 55
    confidence = 0.65
    reason = "Beliefs were produced but do not match expected target"

    if missing:
        culprit = "grounding_failure"
        score = 75 + 5 * len(missing)
        if "quantity" in missing:
            score += 10
        if "query_event" in missing:
            score += 8
        confidence = 0.95
        reason = f"Missing primitives: {', '.join(missing)}"
    elif not beliefs:
        culprit = "empty_reasoning_path"
        score = 85
        confidence = 0.9
        reason = "No belief values were produced"
    elif near_miss:
        culprit = "near_miss_arithmetic"
        score = 35
        confidence = 0.7
        reason = "Closest belief is near expected value"
    elif signals.get("has_two_quantities") and row.get("class") in {"comparison", "transfer", "loss"}:
        culprit = "binding_scope_failure"
        score = 70
        confidence = 0.78
        reason = "Multi-quantity case likely mixed owner/item or event scope"
    elif expected_op in {"divide", "subtract", "multiply"}:
        culprit = "operator_target_mismatch"
        score = 68
        confidence = 0.8
        reason = f"Question implies {expected_op} but produced target appears off-path"

    return {
        "category": culprit,
        "score": int(score),
        "confidence": float(confidence),
        "expected_operator": expected_op,
        "closest_belief": closest,
        "min_abs_error": min_err,
        "reason": reason,
    }


def main() -> int:
    selected = []
    scanned = 0
    skip_neutral = 0
    skip_low_conf = 0

    with IN_PATH.open() as f:
        for i, line in enumerate(f):
            if scanned >= SCAN_LIMIT or len(selected) >= EVAL_LIMIT:
                break
            scanned += 1
            obj = json.loads(line)
            text = obj.get("text") or obj.get("question")
            answer = obj.get("answer")
            props = obj.get("input_props") or []
            if not text or answer is None or not props:
                continue

            q_events = [p for p in props if p.get("pred") == "query"]
            if q_events:
                q_ev = str((q_events[0].get("args") or [None])[0])
            else:
                evs = [p for p in props if p.get("pred") == "event"]
                if not evs:
                    continue
                q_ev = str((evs[0].get("args") or [None])[0])

            verbs = [p for p in props if p.get("pred") == "predicate" and str((p.get("args") or [None])[0]) == q_ev]
            if not verbs:
                continue
            verb_arg = (verbs[0].get("args") or [None, None])[1]
            if isinstance(verb_arg, str) and verb_arg.startswith("?"):
                continue

            cls = ontology_classify_details(str(verb_arg))
            if cls["label"] == "neutral":
                skip_neutral += 1
                continue
            if float(cls["confidence"]) < MIN_CLASS_CONFIDENCE:
                skip_low_conf += 1
                continue

            selected.append(
                {
                    "index": i,
                    "id": obj.get("id"),
                    "text": text,
                    "answer": answer,
                    "input_props": props,
                    "verb": str(verb_arg),
                    "class": str(cls["label"]),
                    "class_confidence": float(cls["confidence"]),
                    "class_reason": str(cls["reason"]),
                }
            )

    start = time.time()
    results = []
    for row in selected:
        bridge = extract_numeric_bridge(input_props=row["input_props"], class_label=row["class"])
        bench = benchmark_single_problem(
            row["text"],
            row["answer"],
            max_cycles=MAX_CYCLES,
            quiet=True,
            input_props=row["input_props"],
            class_label=row["class"],
            use_ltm_recall=False,
            auto_accept_recall=False,
            write_ltm_episode=False,
        )
        results.append(
            {
                "index": row["index"],
                "id": row["id"],
                "question": row["text"],
                "verb": row["verb"],
                "class": row["class"],
                "class_confidence": row["class_confidence"],
                "bridge_missing": bridge["missing_primitives"],
                "bridge_signals": bridge["signals"],
                "bridge_fact_count": len(bridge["bridge_facts"]),
                "expected": bench.get("expected"),
                "belief_values": bench.get("belief_values"),
                "solved": bool(bench.get("solved")),
            }
        )

    for row in results:
        diag = _classify_failure(row)
        row["culprit_category"] = diag["category"]
        row["culprit_score"] = diag["score"]
        row["culprit_confidence"] = diag["confidence"]
        row["expected_operator"] = diag["expected_operator"]
        row["closest_belief"] = diag["closest_belief"]
        row["min_abs_error"] = diag["min_abs_error"]
        row["culprit_reason"] = diag["reason"]

    duration = time.time() - start
    total = len(results)
    solved = sum(1 for r in results if r["solved"])
    nonempty = sum(1 for r in results if r["belief_values"])
    gate_pass = [r for r in results if not r["bridge_missing"]]
    gate_fail = [r for r in results if r["bridge_missing"]]
    missing_counter = Counter(m for r in gate_fail for m in r["bridge_missing"])
    culprit_counter = Counter(r["culprit_category"] for r in results if not r["solved"])
    top_culprits = sorted(
        [r for r in results if not r["solved"]],
        key=lambda r: (r["culprit_score"], r["culprit_confidence"]),
        reverse=True,
    )[:15]

    summary = {
        "dataset": str(IN_PATH),
        "scan_limit": SCAN_LIMIT,
        "eval_limit": EVAL_LIMIT,
        "max_cycles": MAX_CYCLES,
        "min_class_confidence": MIN_CLASS_CONFIDENCE,
        "scanned": scanned,
        "selected": len(selected),
        "skip_neutral": skip_neutral,
        "skip_low_confidence": skip_low_conf,
        "numeric_solved": solved,
        "numeric_solved_rate": (solved / total) if total else 0.0,
        "nonempty_belief_count": nonempty,
        "nonempty_belief_rate": (nonempty / total) if total else 0.0,
        "gate_pass_count": len(gate_pass),
        "gate_fail_count": len(gate_fail),
        "gate_pass_nonempty_belief_rate": (
            sum(1 for r in gate_pass if r["belief_values"]) / len(gate_pass) if gate_pass else 0.0
        ),
        "gate_fail_nonempty_belief_rate": (
            sum(1 for r in gate_fail if r["belief_values"]) / len(gate_fail) if gate_fail else 0.0
        ),
        "top_missing_primitives": missing_counter.most_common(12),
        "culprit_breakdown": culprit_counter.most_common(),
        "top_culprit_problems": [
            {
                "id": r["id"],
                "index": r["index"],
                "verb": r["verb"],
                "class": r["class"],
                "culprit_category": r["culprit_category"],
                "culprit_score": r["culprit_score"],
                "culprit_confidence": r["culprit_confidence"],
                "expected_operator": r["expected_operator"],
                "expected": r["expected"],
                "closest_belief": r["closest_belief"],
                "min_abs_error": r["min_abs_error"],
                "bridge_missing": r["bridge_missing"],
            }
            for r in top_culprits
        ],
        "duration_sec": duration,
    }

    OUT_PATH.write_text(json.dumps({"summary": summary, "results": results}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
