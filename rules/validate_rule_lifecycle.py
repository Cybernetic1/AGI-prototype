from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from working_memory.working_memory import WorkingMemory


def main() -> None:
    registry_path = REPO_ROOT / "rules" / "nl_rule_registry.yaml"
    original = registry_path.read_text(encoding="utf-8") if registry_path.exists() else "rules: []\n"
    wm = WorkingMemory(default_ttl=3)

    try:
        raw_rule = "If a later state exists for the same owner and item, prefer the later state."
        result = wm.inject_nl_rule(raw_rule, source="validator")
        assert result["accepted"] is True
        assert result["status"] == "proposed"
        rule_id = result["rule_id"]

        try:
            wm.transition_nl_rule(rule_id, "active", reviewer="validator", reason="should fail before approval")
            raise AssertionError("Expected invalid transition proposed->active to fail")
        except ValueError:
            pass

        approved = wm.transition_nl_rule(rule_id, "approved", reviewer="validator", reason="looks good")
        assert approved["status"] == "approved"
        active = wm.transition_nl_rule(rule_id, "active", reviewer="validator", reason="activate for runtime")
        assert active["status"] == "active"

        preferred_rules = [f for f in wm.facts if f.concept == "PreferredStateRule" and f.to_dict().get("rule_id") == rule_id]
        assert preferred_rules, "Expected active rule to materialize runtime PreferredStateRule fact"

        retracted = wm.transition_nl_rule(rule_id, "retracted", reviewer="validator", reason="test rollback")
        assert retracted["status"] == "retracted"
    finally:
        registry_path.write_text(original, encoding="utf-8")

    print("Rule lifecycle validation passed.")


if __name__ == "__main__":
    main()
