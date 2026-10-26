from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from working_memory.working_memory import WMNode, WorkingMemory


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def main() -> None:
    registry_path = REPO_ROOT / "rules" / "nl_rule_registry.yaml"
    ltm_path = REPO_ROOT / "rules" / "ltm_store.yaml"
    registry_before = _read(registry_path)
    ltm_before = _read(ltm_path)

    try:
        wm = WorkingMemory(default_ttl=3)
        # Register one teach-injected rule and force it active for testing.
        inject = wm.inject_nl_rule(
            "If a later state exists for the same owner and item, prefer the later state.",
            source="teacher",
            target_status="proposed",
            rule_id="r_test_rule",
        )
        assert inject["accepted"] is True
        wm.transition_nl_rule("r_test_rule", "approved", reviewer="teacher", reason="validation setup")
        wm.transition_nl_rule("r_test_rule", "active", reviewer="teacher", reason="validation setup")

        # Simulate defeasible accounting events.
        wm.update(
            {
                WMNode.create("DefeasibleMeta", layer="meta", source="defeasible-system2", label="rule_applied", arg0="r_test_rule", arg1="belief"),
                WMNode.create("DefeasibleMeta", layer="meta", source="defeasible-system2", label="overrides", arg0="r_test_rule", arg1="r_other"),
            }
        )
        wm.ltm_write_episode(reward=1.0, outcome="accounting")

        entry = wm.rule_registry.get("r_test_rule")
        stats = (entry or {}).get("accounting") or {}
        assert int(stats.get("applied", 0)) >= 1
        assert int(stats.get("wins", 0)) >= 1

        # Seed LTM with recallable propositions tied to the same rule_id.
        wm2 = WorkingMemory(default_ttl=3)
        wm2.update(
            {
                WMNode.create("Rule", layer="meta", source="teacher", rule_id="r_test_rule", family="prefer_latest_state", status="active", owner="Bob", item="apples"),
                WMNode.create("Query", layer="semantic", source="test", goal="remaining_quantity", args=("Bob", "apples")),
            }
        )
        wm2.ltm_write_episode(reward=0.5, outcome="recall-seed")

        wm3 = WorkingMemory(default_ttl=3)
        wm3.update({WMNode.create("Query", layer="semantic", source="test", goal="remaining_quantity", args=("Bob", "apples"))})
        recalled = wm3.ltm_associative_recall(top_k=3, min_score=0.01, inject=False)
        recall_id = recalled.get("recall_id")
        candidates = wm3.list_recall_candidates(recall_id)
        assert candidates, "Expected staged recall candidates"

        rule_candidates = [c for c in candidates if c.get("rule_id") == "r_test_rule"]
        assert rule_candidates, "Expected at least one candidate linked to r_test_rule"
        first = int(rule_candidates[0]["candidate_id"])
        wm3.resolve_recall_candidate(recall_id, first, accept=False, reason="test-reject")

        if len(rule_candidates) > 1:
            second = int(rule_candidates[1]["candidate_id"])
            wm3.resolve_recall_candidate(recall_id, second, accept=True, reason="test-accept")
        else:
            # Force an accept path on any other candidate and attach rule_id for accounting.
            any_id = int(candidates[0]["candidate_id"])
            wm3.pending_recall[recall_id][0]["rule_id"] = "r_test_rule"
            wm3.resolve_recall_candidate(recall_id, any_id, accept=True, reason="test-accept")

        entry2 = wm3.rule_registry.get("r_test_rule")
        stats2 = (entry2 or {}).get("accounting") or {}
        assert int(stats2.get("recall_rejected", 0)) >= 1
        assert int(stats2.get("recall_accepted", 0)) >= 1
    finally:
        if registry_before == "":
            if registry_path.exists():
                registry_path.unlink()
        else:
            registry_path.write_text(registry_before, encoding="utf-8")
        if ltm_before == "":
            if ltm_path.exists():
                ltm_path.unlink()
        else:
            ltm_path.write_text(ltm_before, encoding="utf-8")

    print("Rule accounting + recall accept/reject validation passed.")


if __name__ == "__main__":
    main()
