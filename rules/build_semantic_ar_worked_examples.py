#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class CategorySpec:
    name: str
    required_shapes: tuple[str, ...]
    build_prompt: Callable[[int], str]
    build_trace: Callable[[int], list[dict]]
    target_text: Callable[[int], str]
    negatives: tuple[str, ...]
    seed_rules: tuple[str, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _specs() -> list[CategorySpec]:
    return [
        CategorySpec(
            name="physical_inventory",
            required_shapes=("binary_relation", "transition_state", "arithmetic_operator", "control_query"),
            build_prompt=lambda i: f"Alex had {18 + i % 7} apples, gave away {4 + i % 3}, then bought {2 + i % 4}. How many now?",
            build_trace=lambda i: [
                {"step": "observe", "proposition": f"Has(alex, apples, {18 + i % 7})"},
                {"step": "transition", "proposition": f"StateTransition(alex, apples, delta=-{4 + i % 3})"},
                {"step": "transition", "proposition": f"StateTransition(alex, apples, delta=+{2 + i % 4})"},
                {"step": "infer", "proposition": "Remaining(alex, apples, ?q)"},
            ],
            target_text=lambda i: f"Remaining(alex, apples, {(18 + i % 7) - (4 + i % 3) + (2 + i % 4)})",
            negatives=("Total(group, apples, ?q)", "Ratio(alex, bob, apples, ?k)"),
            seed_rules=(
                "Has(x,item,q0) & StateTransition(x,item,delta=d1) -> Has(x,item,q0+d1)",
                "Has(x,item,q1) & StateTransition(x,item,delta=d2) -> Remaining(x,item,q1+d2)",
            ),
        ),
        CategorySpec(
            name="quantitative_arithmetic",
            required_shapes=("binary_relation", "arithmetic_operator", "control_query"),
            build_prompt=lambda i: f"A box has {3 + i % 6} packs with {5 + i % 4} pens each. Total pens?",
            build_trace=lambda i: [
                {"step": "observe", "proposition": f"Quantity(box, packs, {3 + i % 6})"},
                {"step": "observe", "proposition": f"Rate(pens, per_pack, {5 + i % 4})"},
                {"step": "infer", "proposition": "Multiply(packs, per_pack, total_pens)"},
                {"step": "infer", "proposition": "Total(box, pens, total_pens)"},
            ],
            target_text=lambda i: f"Total(box, pens, {(3 + i % 6) * (5 + i % 4)})",
            negatives=("Remaining(box, pens, ?q)", "Percent(box, pens, ?p)"),
            seed_rules=(
                "Quantity(x,item,n) & Rate(item,per_unit,r) -> Multiply(n,r,total)",
                "Multiply(n,r,total) -> Total(x,item,total)",
            ),
        ),
        CategorySpec(
            name="temporal_event",
            required_shapes=("transition_state", "temporal_causal", "control_query"),
            build_prompt=lambda i: f"At t1 Mia had {9 + i % 5} books; at t2 she lost {2 + i % 2}; at t3 she got {3 + i % 3}. Final books?",
            build_trace=lambda i: [
                {"step": "observe", "proposition": "Timepoint(t1)"},
                {"step": "observe", "proposition": f"Has(mia, books, {9 + i % 5})"},
                {"step": "transition", "proposition": f"Event(t2, lose, mia, books, {2 + i % 2})"},
                {"step": "transition", "proposition": f"Event(t3, gain, mia, books, {3 + i % 3})"},
                {"step": "infer", "proposition": "FinalState(mia, books, ?q)"},
            ],
            target_text=lambda i: f"FinalState(mia, books, {(9 + i % 5) - (2 + i % 2) + (3 + i % 3)})",
            negatives=("Event(t3, lose, mia, books, ?q)", "Contradiction(final_state, mia_books)"),
            seed_rules=(
                "Event(t,lose,x,item,d) -> StateTransition(x,item,delta=-d,event_order=t)",
                "Event(t,gain,x,item,d) -> StateTransition(x,item,delta=+d,event_order=t)",
                "OrderedTransitions(x,item,*) -> FinalState(x,item,q_final)",
            ),
        ),
        CategorySpec(
            name="causal_intervention",
            required_shapes=("binary_relation", "temporal_causal", "frame_role", "control_query"),
            build_prompt=lambda i: f"If pump speed is {2 + i % 3}x, output rises by {10 + i % 5}%. Base output 100. New output?",
            build_trace=lambda i: [
                {"step": "observe", "proposition": f"Intervention(pump_speed, factor, {2 + i % 3})"},
                {"step": "observe", "proposition": f"Percent(output, rise, {10 + i % 5}%)"},
                {"step": "infer", "proposition": "Cause(intervention, output_change)"},
                {"step": "infer", "proposition": "FinalState(system, output, ?q)"},
            ],
            target_text=lambda i: f"FinalState(system, output, {100 * (1 + (10 + i % 5) / 100):.2f})",
            negatives=("FinalState(system, output, 100)", "Cause(output_change, intervention)"),
            seed_rules=(
                "Intervention(v,factor,k) & Percent(o,rise,p) -> Cause(v,o)",
                "Cause(v,o) & Quantity(system,o,q0) & Percent(o,rise,p) -> FinalState(system,o,q0*(1+p))",
            ),
        ),
        CategorySpec(
            name="social_intent",
            required_shapes=("frame_role", "binary_relation", "transition_state", "control_query"),
            build_prompt=lambda i: f"Rin intends to gift {3 + i % 3} oranges from stock {11 + i % 4} to Kai. Remaining stock?",
            build_trace=lambda i: [
                {"step": "observe", "proposition": f"Has(rin, oranges, {11 + i % 4})"},
                {"step": "observe", "proposition": f"Intent(rin, gift, kai, oranges, {3 + i % 3})"},
                {"step": "infer", "proposition": "Event(transfer, rin, kai, oranges, qty)"},
                {"step": "infer", "proposition": "Remaining(rin, oranges, ?q)"},
            ],
            target_text=lambda i: f"Remaining(rin, oranges, {(11 + i % 4) - (3 + i % 3)})",
            negatives=("Remaining(kai, oranges, ?q)", "Intent(kai, gift, rin, oranges, ?q)"),
            seed_rules=(
                "Intent(a,gift,b,item,q) & Has(a,item,q0) -> Event(transfer,a,b,item,q)",
                "Event(transfer,a,b,item,q) & Has(a,item,q0) -> Remaining(a,item,q0-q)",
            ),
        ),
        CategorySpec(
            name="epistemic_defeasible",
            required_shapes=("control_query", "temporal_causal", "unary_attribute", "frame_role"),
            build_prompt=lambda i: f"Sensor says tank level {40 + i % 8}. Later manual check says {36 + i % 6}. Which belief retained?",
            build_trace=lambda i: [
                {"step": "observe", "proposition": f"Belief(sensor, tank_level, {40 + i % 8}, confidence=0.62)"},
                {"step": "observe", "proposition": f"Belief(manual_check, tank_level, {36 + i % 6}, confidence=0.89)"},
                {"step": "infer", "proposition": "Conflict(tank_level)"},
                {"step": "infer", "proposition": "RevisedBelief(tank_level, ?q)"},
            ],
            target_text=lambda i: f"RevisedBelief(tank_level, {36 + i % 6})",
            negatives=("RevisedBelief(tank_level, sensor_value)", "NoRevision(tank_level)"),
            seed_rules=(
                "Belief(a,fact,v1,c1) & Belief(b,fact,v2,c2) & c2>c1 -> Conflict(fact)",
                "Conflict(fact) & HigherConfidenceSource(s) -> RevisedBelief(fact,v_s)",
            ),
        ),
    ]


def build_examples(per_category: int, seed: int) -> dict:
    specs = _specs()
    examples: list[dict] = []
    for spec in specs:
        for i in range(per_category):
            trace = spec.build_trace(i)
            trace = sorted(trace, key=lambda row: {"observe": 0, "transition": 1, "infer": 2}.get(row["step"], 9))
            anchor = trace[0]["proposition"]
            positives = [row["proposition"] for row in trace[1:]]
            examples.append(
                {
                    "example_id": f"sar_{spec.name}_{i+1:03d}",
                    "category": spec.name,
                    "prompt": spec.build_prompt(i),
                    "required_shapes": list(spec.required_shapes),
                    "trace": trace,
                    "target": spec.target_text(i),
                    "semantic_ar_tuple": {
                        "anchor": anchor,
                        "positive_continuations": positives,
                        "hard_negatives": list(spec.negatives),
                        "retrieval_cues": [spec.name, "semantic-ar", "reasoning-chain"],
                    },
                    "seed_rules": list(spec.seed_rules),
                }
            )
    return {
        "artifact_id": "semantic-ar-worked-examples-v1",
        "generated_at": _utc_now(),
        "config": {
            "per_category": per_category,
            "seed": seed,
            "categories": [s.name for s in specs],
            "gsm_weighting": "diagnostic_only",
        },
        "summary": {
            "total_examples": len(examples),
            "categories": len(specs),
        },
        "examples": examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build worked semantic-AR examples and seed-rule templates.")
    parser.add_argument("--per-category", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="rules/reports/semantic_ar_worked_examples_v1.json")
    args = parser.parse_args()

    payload = build_examples(max(1, int(args.per_category)), int(args.seed))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved semantic-AR worked examples to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
