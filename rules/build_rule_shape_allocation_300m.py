#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _base_shape_specs(total_rules: int, d_rule: int) -> list[dict[str, Any]]:
    specs = [
        ("unary_attribute", 704, 2, "entity/property typing and priors"),
        ("binary_relation", 1024, 3, "owner-item and agent-patient style relations"),
        ("transition_state", 832, 5, "before/delta/after transitions"),
        ("arithmetic_operator", 640, 4, "add/sub/mul/div/percent transforms"),
        ("temporal_causal", 384, 4, "event ordering and causal links"),
        ("frame_role", 320, 6, "frame-semantic role bindings"),
        ("control_query", 192, 3, "goal routing, query focus, and verification"),
    ]
    current = sum(s[1] for s in specs)
    if total_rules != current:
        scale = total_rules / float(current)
        scaled = []
        for name, count, slots, rationale in specs:
            scaled.append((name, max(8, int(round(count * scale))), slots, rationale))
        specs = scaled
        drift = total_rules - sum(s[1] for s in specs)
        specs[1] = (specs[1][0], specs[1][1] + drift, specs[1][2], specs[1][3])

    return [
        {
            "shape": name,
            "count": count,
            "slots": slots,
            "tensor_shape": [count, slots, d_rule],
            "rationale": rationale,
            "memory_bindings": {
                "episodic_trigger": name in {"transition_state", "frame_role", "control_query"},
                "semantic_abstraction": name in {"unary_attribute", "binary_relation", "temporal_causal"},
                "rule_lifecycle_tracking": True,
            },
        }
        for name, count, slots, rationale in specs
    ]


def build_payload(total_rules: int, d_rule: int, gpu: str, corpus_cap_gb: int) -> dict[str, Any]:
    shapes = _base_shape_specs(total_rules=total_rules, d_rule=d_rule)
    total_tensors = sum(shape["count"] * shape["slots"] * d_rule for shape in shapes)
    return {
        "artifact_id": "rule-shape-allocation-300m-v1",
        "generated_at": _utc_now(),
        "model_profile": {
            "parameter_budget": "300M-class",
            "rule_embedding_dim": d_rule,
            "rule_bank_total": total_rules,
            "target_gpu": gpu,
            "corpus_cap_gb": corpus_cap_gb,
        },
        "rule_shapes": shapes,
        "allocation_summary": {
            "shape_count": len(shapes),
            "total_rules": total_rules,
            "total_rule_tensor_values": total_tensors,
        },
        "readiness_checks": {
            "requires_chain_completeness_eval": True,
            "min_chain_success_rate": 0.85,
            "max_dead_end_ratio": 0.10,
            "max_contradiction_ratio": 0.05,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize rule-shape allocation for a 300M LT configuration.")
    parser.add_argument("--total-rules", type=int, default=4096)
    parser.add_argument("--d-rule", type=int, default=256)
    parser.add_argument("--gpu-target", default="rtx_5090_32gb")
    parser.add_argument("--corpus-cap-gb", type=int, default=300)
    parser.add_argument("--output", default="rules/reports/rule_shape_allocation_300m_v1.json")
    args = parser.parse_args()

    payload = build_payload(
        total_rules=max(128, int(args.total_rules)),
        d_rule=max(32, int(args.d_rule)),
        gpu=str(args.gpu_target),
        corpus_cap_gb=max(1, int(args.corpus_cap_gb)),
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved rule-shape allocation to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
