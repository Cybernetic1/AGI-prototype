from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from working_memory.working_memory import WorkingMemory


def _sample_proposals() -> dict:
    return {
        "rules": [
            {"rule_id": "teacher-prop-prefer-latest", "raw_rule": "If a later state exists for the same owner and item, prefer the later state."},
            {"rule_id": "teacher-prop-bind", "raw_rule": "If same owner and same item are mentioned, bind them to the same identity."},
        ],
        "facts": [
            {
                "fact": {
                    "kind": "state",
                    "id": "state_bob_apples_1",
                    "owner": "Bob",
                    "item": "apples",
                    "qty": 10,
                    "phase": "before",
                }
            }
        ],
        "counterfactuals": [
            {
                "hypothesis": "If Bob gives away apples, remaining apples should decrease.",
                "if": {
                    "kind": "state",
                    "id": "state_cf_before",
                    "owner": "Bob",
                    "item": "apples",
                    "qty": 10,
                    "phase": "before",
                },
                "then": {
                    "kind": "state",
                    "id": "state_cf_after",
                    "owner": "Bob",
                    "item": "apples",
                    "qty": 7,
                    "phase": "after",
                },
            }
        ],
    }


def main() -> None:
    registry_path = REPO_ROOT / "rules" / "nl_rule_registry.yaml"
    original = registry_path.read_text(encoding="utf-8") if registry_path.exists() else "rules: []\nteacher_proposals: []\n"
    wm = WorkingMemory(default_ttl=3)
    proposals = _sample_proposals()

    try:
        report = wm.ingest_teacher_proposals(
            proposals,
            source="teacher_llm",
            teacher_id="validator",
            teacher_model="validator-model",
            ab_arm="treatment",
            enable_ingestion=True,
            low_volume_mode=True,
            max_total=3,
            max_rules=1,
            max_facts=1,
            max_counterfactuals=1,
        )
        assert report["accepted"] is True
        assert report["counts"]["submitted"] == 4
        assert report["counts"]["accepted"] == 3
        assert report["counts"]["dropped_by_volume"] == 1
        assert report["accepted_by_kind"] == {"rules": 1, "facts": 1, "counterfactuals": 1}

        rule_events = [e for e in report["events"] if e.get("proposal_kind") == "rule" and e.get("decision") == "accepted"]
        assert rule_events, "Expected at least one accepted teacher rule proposal"
        rule_id = str(rule_events[0].get("rule_id"))
        entry = wm.rule_registry.get(rule_id) or {}
        assert entry.get("status") == "proposed"
        assert entry.get("runtime_authority") == "proposal_only"
        assert str((entry.get("audit") or {}).get("ab_arm")) == "treatment"

        wm.transition_nl_rule(rule_id, "approved", reviewer="validator", reason="smoke gate")
        try:
            wm.transition_nl_rule(rule_id, "active", reviewer="validator", reason="should stay proposal-only")
            raise AssertionError("Expected teacher proposal-only rule activation to fail")
        except ValueError as exc:
            assert "proposal-only" in str(exc).lower() or "runtime-active" in str(exc).lower()

        control = wm.ingest_teacher_proposals(
            proposals,
            source="teacher_llm",
            teacher_id="validator",
            teacher_model="validator-model",
            ab_arm="control",
            enable_ingestion=True,
            low_volume_mode=True,
        )
        assert control["accepted"] is False
        assert control.get("reason") == "control arm: ingestion skipped"

        logged = wm.rule_registry.list_teacher_proposals()
        assert logged, "Expected teacher proposal audit log entries"
        assert all("ab_arm" in item for item in logged), "Every teacher proposal event should keep A/B provenance"
    finally:
        registry_path.write_text(original, encoding="utf-8")

    print("Teacher proposal ingestion validation passed.")


if __name__ == "__main__":
    main()
