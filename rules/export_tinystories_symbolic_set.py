#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_tinystories_rows() -> list[dict[str, Any]]:
    src = REPO_ROOT / "cognitive_loop.py"
    tree = ast.parse(src.read_text(encoding="utf-8"), filename=str(src))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "TINYSTORIES_STATE_TRACKING_FAMILIES":
                value = ast.literal_eval(node.value)
                if isinstance(value, list):
                    return [row for row in value if isinstance(row, dict)]
    raise ValueError("TINYSTORIES_STATE_TRACKING_FAMILIES not found in cognitive_loop.py")


def _to_float(text: str, default: float = 0.0) -> float:
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    if not m:
        return default
    try:
        return float(m.group(0))
    except Exception:
        return default


def _symbolic_targets(row: dict[str, Any]) -> dict[str, Any]:
    family = str(row.get("family") or "")
    question = str(row.get("question") or "")
    expected = float(row.get("expected") or 0.0)

    if family == "tinystory_basket_lunch_dinner":
        start = _to_float(question, 20.0)
        first_eat = _to_float(question.split("First,")[1], 5.0) if "First," in question else 5.0
        rem_after_first = start - first_eat
        m_pct = re.search(r"Then\s+(\d+(?:\.\d+)?)%\s+of\s+the\s+remaining", question, re.IGNORECASE)
        pct = (float(m_pct.group(1)) / 100.0) if m_pct else 0.25
        return {
            "owner": "basket",
            "item": "apples",
            "states": [start, rem_after_first, expected],
            "operations": [
                {"op": "state_set", "args": ["basket", "apples", start]},
                {"op": "state_add_delta", "args": ["basket", "apples", -first_eat]},
                {"op": "state_apply_percent_reduction", "args": ["basket", "apples", pct]},
            ],
            "query": {"goal": "remaining_quantity", "args": ["basket", "apples", "?ans"], "expected": expected},
        }

    if family == "tinystory_basket_after_school":
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", question)]
        start = nums[0] if len(nums) >= 1 else 30.0
        eaten = nums[1] if len(nums) >= 2 else 6.0
        pct = (nums[2] / 100.0) if len(nums) >= 3 else 0.2
        rem_after_first = start - eaten
        return {
            "owner": "basket",
            "item": "apples",
            "states": [start, rem_after_first, expected],
            "operations": [
                {"op": "state_set", "args": ["basket", "apples", start]},
                {"op": "state_add_delta", "args": ["basket", "apples", -eaten]},
                {"op": "state_apply_percent_reduction", "args": ["basket", "apples", pct]},
            ],
            "query": {"goal": "remaining_quantity", "args": ["basket", "apples", "?ans"], "expected": expected},
        }

    if family == "tinystory_ratio_market_day":
        # Encodes the current ratio/finality template used in GSM + TinyStories blend.
        return {
            "owner": "A",
            "ref_owner": "B",
            "item": "apples",
            "states": [30.0, 15.0, 12.0, 20.0, expected],
            "operations": [
                {"op": "ratio_compose", "args": ["A", "B", "apples", 3.0]},
                {"op": "state_set", "args": ["A", "apples", 30.0]},
                {"op": "state_add_delta", "args": ["A", "apples", -15.0]},
                {"op": "state_apply_percent_reduction", "args": ["A", "apples", 0.2]},
                {"op": "state_add_delta", "args": ["A", "apples", 8.0]},
                {"op": "state_add_delta", "args": ["A", "apples", -5.0]},
            ],
            "query": {"goal": "remaining_quantity", "args": ["A", "apples", "?ans"], "expected": expected},
        }

    return {
        "owner": "unknown",
        "item": "unknown",
        "states": [expected],
        "operations": [],
        "query": {"goal": "remaining_quantity", "args": ["unknown", "unknown", "?ans"], "expected": expected},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export compact TinyStories symbolic curation set.")
    parser.add_argument("--output", default="rules/reports/tinystories_symbolic_compact_v1.jsonl")
    args = parser.parse_args()

    rows = _load_tinystories_rows()
    out_rows = []
    for row in rows:
        out_rows.append(
            {
                "dataset_id": "tinystories-symbolic-compact-v1",
                "family": row.get("family"),
                "question": row.get("question"),
                "expected": row.get("expected"),
                "symbolic_targets": _symbolic_targets(row),
                "split": "train",
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for item in out_rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Exported {len(out_rows)} TinyStories symbolic rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
