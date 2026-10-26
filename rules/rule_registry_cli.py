#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from working_memory.rule_registry import RuleRegistry


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage NL rule lifecycle statuses.")
    parser.add_argument("rule_id", help="Rule ID to transition.")
    parser.add_argument("status", choices=["approved", "active", "retracted"], help="Target status.")
    parser.add_argument("--reviewer", default="human", help="Reviewer identity.")
    parser.add_argument("--reason", default="", help="Reason for status transition.")
    args = parser.parse_args()

    registry = RuleRegistry.default()
    entry = registry.transition(args.rule_id, args.status, reviewer=args.reviewer, reason=args.reason)
    print(
        f"updated {entry.get('rule_id')} -> status={entry.get('status')} "
        f"version={entry.get('version')} updated_at={entry.get('updated_at')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
