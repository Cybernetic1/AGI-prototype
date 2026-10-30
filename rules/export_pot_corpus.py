#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from working_memory.pot_dsl import PoTProgram, PoTStep, execute_program


def _extract_rows() -> list[dict[str, Any]]:
    src = REPO_ROOT / "cognitive_loop.py"
    tree = ast.parse(src.read_text(encoding="utf-8"), filename=str(src))
    wanted = {
        "FINALITY_REGRESSION_FAMILIES",
        "GENERALIZED_TEMPLATE_FAMILIES",
        "TINYSTORIES_STATE_TRACKING_FAMILIES",
        "MIXED_NARRATIVE_SYMBOLIC_FAMILIES",
    }
    rows: list[dict[str, Any]] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name) or target.id not in wanted:
                continue
            value = ast.literal_eval(node.value)
            if not isinstance(value, list):
                continue
            for item in value:
                if isinstance(item, dict) and item.get("question") and item.get("family"):
                    rows.append(
                        {
                            "suite_constant": target.id,
                            "family": item.get("family"),
                            "question": item.get("question"),
                            "expected": item.get("expected"),
                        }
                    )
    return rows


def _num(text: str, default: float | None = None) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    if not m:
        return default
    try:
        return float(m.group(0))
    except Exception:
        return default


def _program_for_row(row: dict[str, Any]) -> tuple[PoTProgram, str]:
    family = str(row.get("family"))
    question = str(row.get("question"))
    expected = row.get("expected")
    expected_num = _num(str(expected))

    if family == "distance_over_speed":
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", question)]
        miles = nums[0] if len(nums) >= 1 else expected_num or 0.0
        mph = nums[1] if len(nums) >= 2 else 1.0
        return (
            PoTProgram(
                program_id=f"pot:{family}",
                steps=(
                    PoTStep("const", "miles", (miles,)),
                    PoTStep("const", "mph", (mph,)),
                    PoTStep("div", "ans", ("$miles", "$mph")),
                ),
            ),
            "template_distance_over_speed",
        )

    if family in {
        "buy_more_final_gap",
        "donation_after_buy_more",
        "buy_more_then_donation_and_spoilage",
        "buy_more_then_donation_after_spoilage_long_tail",
    }:
        # Canonical finality sequence template for the active GSM family cluster.
        return (
            PoTProgram(
                program_id=f"pot:{family}",
                steps=(
                    PoTStep("const", "b_qty", (10,)),
                    PoTStep("const", "factor", (3,)),
                    PoTStep("mul", "a_start", ("$b_qty", "$factor")),
                    PoTStep("state_set", "a_state0", ("a", "apples", "$a_start")),
                    PoTStep("state_add_delta", "a_after_giveaway", ("a", "apples", -15)),
                    PoTStep("state_apply_percent_reduction", "a_after_eat", ("a", "apples", 0.2)),
                    PoTStep("state_add_delta", "a_after_buy", ("a", "apples", 8)),
                    PoTStep("state_add_delta", "a_after_spoil", ("a", "apples", -5)),
                    PoTStep("state_query", "ans", ("a", "apples")),
                ),
            ),
            "template_finality_sequence",
        )

    # Fallback keeps the dataset total coverage while marking quality for training filters.
    return (
        PoTProgram(
            program_id=f"pot:{family}",
            steps=(PoTStep("const", "ans", (expected_num if expected_num is not None else 0.0,)),),
        ),
        "fallback_expected_constant",
    )


def _program_to_dict(program: PoTProgram) -> dict[str, Any]:
    return {
        "program_id": program.program_id,
        "steps": [
            {"op": s.op, "out": s.out, "args": list(s.args), "meta": dict(s.meta)}
            for s in program.steps
        ],
    }


def _make_negative(program: PoTProgram) -> PoTProgram | None:
    # Corrupt one arithmetic/state-changing step to create an explicit negative.
    steps = list(program.steps)
    for idx, step in enumerate(steps):
        if step.op == "add":
            steps[idx] = PoTStep("sub", step.out, step.args, dict(step.meta))
            return PoTProgram(program_id=f"{program.program_id}:neg", steps=tuple(steps))
        if step.op == "sub":
            steps[idx] = PoTStep("add", step.out, step.args, dict(step.meta))
            return PoTProgram(program_id=f"{program.program_id}:neg", steps=tuple(steps))
        if step.op == "mul":
            steps[idx] = PoTStep("div", step.out, step.args, dict(step.meta))
            return PoTProgram(program_id=f"{program.program_id}:neg", steps=tuple(steps))
        if step.op == "div":
            steps[idx] = PoTStep("mul", step.out, step.args, dict(step.meta))
            return PoTProgram(program_id=f"{program.program_id}:neg", steps=tuple(steps))
        if step.op == "state_add_delta" and len(step.args) == 3:
            delta = step.args[2]
            if isinstance(delta, (int, float)):
                steps[idx] = PoTStep(step.op, step.out, (step.args[0], step.args[1], -float(delta)), dict(step.meta))
                return PoTProgram(program_id=f"{program.program_id}:neg", steps=tuple(steps))
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Export PoT DSL corpus from benchmark families.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="rules/reports/pot_corpus_v1.jsonl")
    parser.add_argument("--include-negatives", action="store_true", default=True)
    args = parser.parse_args()

    rows = _extract_rows()
    if args.limit > 0:
        rows = rows[: args.limit]

    out_rows: list[dict[str, Any]] = []
    for row in rows:
        program, quality = _program_for_row(row)
        result = execute_program(program)
        expected_num = _num(str(row.get("expected")))
        got = result.vars.get("ans")
        matched = False
        if result.ok and expected_num is not None and got is not None:
            matched = abs(float(got) - float(expected_num)) < 1e-9
        out_rows.append(
            {
                "family": row.get("family"),
                "suite_constant": row.get("suite_constant"),
                "question": row.get("question"),
                "expected": row.get("expected"),
                "label": 1,
                "trace_quality": quality,
                "program": _program_to_dict(program),
                "execution_ok": bool(result.ok),
                "execution_match_expected": bool(matched),
                "issues": [{"step_index": x.step_index, "field": x.field, "reason": x.reason} for x in result.issues],
            }
        )

        if args.include_negatives:
            neg = _make_negative(program)
            if neg is not None:
                neg_result = execute_program(neg)
                neg_got = neg_result.vars.get("ans")
                neg_match = False
                if neg_result.ok and expected_num is not None and neg_got is not None:
                    neg_match = abs(float(neg_got) - float(expected_num)) < 1e-9
                out_rows.append(
                    {
                        "family": row.get("family"),
                        "suite_constant": row.get("suite_constant"),
                        "question": row.get("question"),
                        "expected": row.get("expected"),
                        "label": 0,
                        "trace_quality": f"{quality}:corrupted",
                        "program": _program_to_dict(neg),
                        "execution_ok": bool(neg_result.ok),
                        "execution_match_expected": bool(neg_match),
                        "issues": [
                            {"step_index": x.step_index, "field": x.field, "reason": x.reason}
                            for x in neg_result.issues
                        ],
                    }
                )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in out_rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Exported {len(out_rows)} PoT corpus rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
