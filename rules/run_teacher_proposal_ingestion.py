#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from working_memory.working_memory import WorkingMemory


def _load_proposals(path: str | None) -> Any:
    if not path:
        return {}
    src = Path(path)
    if not src.is_absolute():
        src = REPO_ROOT / src
    if not src.exists():
        raise FileNotFoundError(f"proposal file not found: {src}")
    payload = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(payload, (dict, list)):
        raise ValueError("proposal payload must be a JSON object or list")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest teacher-LLM proposals with strict proposal-only gating."
    )
    parser.add_argument("--input", default="", help="JSON file containing rule/fact/counterfactual candidates.")
    parser.add_argument("--enable", action="store_true", help="Enable ingestion. Disabled by default.")
    parser.add_argument("--ab-arm", choices=["control", "treatment"], default="control")
    parser.add_argument("--low-volume", dest="low_volume", action="store_true", default=True, help="Enable strict low-volume caps.")
    parser.add_argument("--no-low-volume", dest="low_volume", action="store_false", help="Disable strict low-volume caps.")
    parser.add_argument("--teacher-id", default="bootstrap-teacher-llm")
    parser.add_argument("--teacher-model", default="unknown")
    parser.add_argument("--max-total", type=int, default=3)
    parser.add_argument("--max-rules", type=int, default=1)
    parser.add_argument("--max-facts", type=int, default=1)
    parser.add_argument("--max-counterfactuals", type=int, default=1)
    parser.add_argument("--max-rule-chars", type=int, default=280)
    parser.add_argument("--output", default="", help="Optional path to write JSON report.")
    args = parser.parse_args()

    proposals = _load_proposals(args.input or None)
    wm = WorkingMemory(default_ttl=3)
    report = wm.ingest_teacher_proposals(
        proposals,
        source="teacher_llm",
        teacher_id=str(args.teacher_id),
        teacher_model=str(args.teacher_model),
        ab_arm=str(args.ab_arm),
        enable_ingestion=bool(args.enable),
        low_volume_mode=bool(args.low_volume),
        max_total=int(args.max_total),
        max_rules=int(args.max_rules),
        max_facts=int(args.max_facts),
        max_counterfactuals=int(args.max_counterfactuals),
        max_rule_chars=int(args.max_rule_chars),
    )

    content = json.dumps(report, indent=2)
    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = REPO_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
        print(f"Saved teacher-ingestion report to {out_path}")
    else:
        print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
