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

from ilp_learner.metagol_nds import MetaRuleGrammar


def _serialize_rule(rule: Any) -> dict[str, Any]:
    return {
        "family": getattr(rule, "family", ""),
        "name": getattr(rule, "name", ""),
        "head": list(getattr(rule, "head", ("", []))),
        "body": [list(item) for item in (getattr(rule, "body", []) or [])],
        "calc": getattr(rule, "calc", ""),
        "target": getattr(rule, "target", ""),
        "description": getattr(rule, "description", ""),
        "constraints": getattr(rule, "constraints", {}) or {},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export all seeded metarules for inspection.")
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "rules" / "reports" / "metarules_snapshot_v1.json"),
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--agi-transfer-families",
        action="store_true",
        help="Include AGI-transfer families (part-whole, temporal-causal, unit-compatibility).",
    )
    args = parser.parse_args()

    rules = MetaRuleGrammar.seeded_rules(include_agi_transfer_families=bool(args.agi_transfer_families))
    payload = {
        "count": len(rules),
        "include_agi_transfer_families": bool(args.agi_transfer_families),
        "rules": [_serialize_rule(rule) for rule in rules],
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {payload['count']} metarules to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
