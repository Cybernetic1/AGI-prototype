#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "logic_transformer") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "logic_transformer"))

from semantic_ar import score_semantic_ar


def _suite_rows(name: str) -> list[dict[str, Any]]:
    cognitive_loop_path = REPO_ROOT / "cognitive_loop.py"
    tree = ast.parse(cognitive_loop_path.read_text(encoding="utf-8"), filename=str(cognitive_loop_path))
    constants: dict[str, list[dict[str, Any]]] = {}
    wanted = {
        "FINALITY_REGRESSION_FAMILIES",
        "GENERALIZED_TEMPLATE_FAMILIES",
        "TINYSTORIES_STATE_TRACKING_FAMILIES",
        "MIXED_NARRATIVE_SYMBOLIC_FAMILIES",
    }
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in wanted:
                value = ast.literal_eval(node.value)
                if isinstance(value, list):
                    constants[target.id] = value

    finality = list(constants.get("FINALITY_REGRESSION_FAMILIES", []))
    template = list(constants.get("GENERALIZED_TEMPLATE_FAMILIES", []))
    tinystory = list(constants.get("TINYSTORIES_STATE_TRACKING_FAMILIES", []))
    mixed = list(constants.get("MIXED_NARRATIVE_SYMBOLIC_FAMILIES", []))

    if name == "finality_regression":
        return finality
    if name == "template_generalization":
        return template
    if name == "tinystories_state_tracking":
        return tinystory
    if name == "mixed_narrative_symbolic":
        return mixed
    if name == "combined":
        return finality + template
    if name == "combined_widening":
        return finality + template + tinystory + mixed
    raise ValueError(f"unsupported suite: {name}")


def _find_candidates(question: str) -> list[tuple[str, list[Any]]]:
    out: list[tuple[str, list[Any]]] = []
    for owner, qty, item in re.findall(r"\b([A-Z][a-z]*)\s+(?:has|had)\s+(\d+(?:\.\d+)?)\s+([A-Za-z]+)\b", question):
        out.append(("Quantity", [owner, item, float(qty)]))
        out.append(("Quantity", [owner, item, "many"]))  # hard negative
        out.append(("Quantity", ["unknown", item, float(qty)]))  # lexical negative
    for owner, pct, item in re.findall(r"\b([A-Z][a-z]*)\b.*?(\d+(?:\.\d+)?)%\s+(?:of|remaining)\s+([A-Za-z]+)\b", question):
        out.append(("Percent", [owner, item, float(pct) / 100.0]))
        out.append(("Percent", [owner, item, "unknown"]))  # hard negative
    for owner, factor, item, ref_owner in re.findall(
        r"\b([A-Z][a-z]*)\s+has\s+(\d+(?:\.\d+)?)\s+times\s+as\s+many\s+([A-Za-z]+)\s+as\s+([A-Z][a-z]*)\b",
        question,
    ):
        out.append(("Ratio", [owner, ref_owner, item, float(factor)]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Export SAR training examples from benchmark family text.")
    parser.add_argument(
        "--suite",
        default="combined_widening",
        choices=[
            "finality_regression",
            "template_generalization",
            "tinystories_state_tracking",
            "mixed_narrative_symbolic",
            "combined",
            "combined_widening",
        ],
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--base-score", type=float, default=0.6)
    parser.add_argument("--output", default="rules/reports/sar_training_data_v1.jsonl")
    args = parser.parse_args()

    random.seed(int(args.seed))
    rows = _suite_rows(args.suite)
    if args.limit > 0:
        rows = rows[: args.limit]

    dataset: list[dict[str, Any]] = []
    for row in rows:
        question = str(row.get("question") or "")
        for concept, cand_args in _find_candidates(question):
            sar = score_semantic_ar(
                question=question,
                concept=concept,
                args=list(cand_args),
                base_score=float(args.base_score),
                config={"enabled": False},
            )
            label = 1 if sar.symbolic_gate_pass is True and sar.lexical_alignment >= 0.34 else 0
            dataset.append(
                {
                    "family": row.get("family"),
                    "question": question,
                    "concept": concept,
                    "args": cand_args,
                    "base_score": float(args.base_score),
                    "label": label,
                    "features": {
                        "lexical_alignment": sar.lexical_alignment,
                        "role_coverage": sar.role_coverage,
                        "concept_prior": sar.concept_prior,
                        "symbolic_gate_pass": sar.symbolic_gate_pass,
                    },
                }
            )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Exported {len(dataset)} SAR examples to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
