from __future__ import annotations

import sys
from pathlib import Path
import uuid
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from working_memory.working_memory import WMNode, WorkingMemory
from working_memory.rule_registry import RuleRegistry
from working_memory.ltm import LongTermMemory


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="agi_rule_accounting_") as tmp:
        tmp_dir = Path(tmp)
        reg_path = tmp_dir / "registry.yaml"
        ltm_path = tmp_dir / "ltm.yaml"

        wm = WorkingMemory(default_ttl=3)
        wm.rule_registry = RuleRegistry(reg_path)
        wm.ltm = LongTermMemory(ltm_path)
        # Register one teach-injected rule and force it active for testing.
        rule_id = f"r_test_rule_{uuid.uuid4().hex[:8]}"
        inject = wm.inject_nl_rule(
            "If a later state exists for the same owner and item, prefer the later state.",
            source="teacher",
            target_status="proposed",
            rule_id=rule_id,
        )
        assert inject["accepted"] is True
        wm.transition_nl_rule(rule_id, "approved", reviewer="teacher", reason="validation setup")
        wm.transition_nl_rule(rule_id, "active", reviewer="teacher", reason="validation setup")

        # Simulate defeasible accounting events.
        wm.update(
            {
                WMNode.create("DefeasibleMeta", layer="meta", source="defeasible-system2", label="rule_applied", arg0=rule_id, arg1="belief"),
                WMNode.create("DefeasibleMeta", layer="meta", source="defeasible-system2", label="overrides", arg0=rule_id, arg1="r_other"),
            }
        )
        wm.ltm_write_episode(reward=1.0, outcome="accounting")

        entry = wm.rule_registry.get(rule_id)
        stats = (entry or {}).get("accounting") or {}
        assert int(stats.get("applied", 0)) >= 1
        assert int(stats.get("wins", 0)) >= 1

        # Seed LTM with recallable propositions tied to the same rule_id.
        wm2 = WorkingMemory(default_ttl=3)
        wm2.rule_registry = RuleRegistry(reg_path)
        wm2.ltm = LongTermMemory(ltm_path)
        wm2.update(
            {
                WMNode.create("Rule", layer="meta", source="teacher", rule_id=rule_id, family="prefer_latest_state", status="active", owner="Bob", item="apples"),
                WMNode.create("Query", layer="semantic", source="test", goal="remaining_quantity", args=("Bob", "apples")),
            }
        )
        wm2.ltm_write_episode(reward=0.5, outcome="recall-seed")

        wm3 = WorkingMemory(default_ttl=3)
        wm3.rule_registry = RuleRegistry(reg_path)
        wm3.ltm = LongTermMemory(ltm_path)
        wm3.update({WMNode.create("Query", layer="semantic", source="test", goal="remaining_quantity", args=("Bob", "apples"))})
        recalled = wm3.ltm_associative_recall(top_k=3, min_score=0.01, inject=False)
        recall_id = recalled.get("recall_id")
        candidates = wm3.list_recall_candidates(recall_id)
        assert candidates, "Expected staged recall candidates"

        rule_candidates = [c for c in candidates if c.get("rule_id") == rule_id]
        assert rule_candidates, f"Expected at least one candidate linked to {rule_id}"
        first = int(rule_candidates[0]["candidate_id"])
        wm3.resolve_recall_candidate(recall_id, first, accept=False, reason="test-reject")

        # Run a fresh recall to test an accept path independently from the rejected candidate.
        recalled2 = wm3.ltm_associative_recall(top_k=3, min_score=0.01, inject=False)
        recall_id_2 = recalled2.get("recall_id")
        candidates2 = wm3.list_recall_candidates(recall_id_2)
        rule_candidates2 = [c for c in candidates2 if c.get("rule_id") == rule_id]
        if not rule_candidates2 and candidates2:
            wm3.pending_recall[recall_id_2][0]["rule_id"] = rule_id
            rule_candidates2 = [wm3.pending_recall[recall_id_2][0]]
        assert rule_candidates2, "Expected a candidate for accept-path accounting"
        second = int(rule_candidates2[0]["candidate_id"])
        wm3.resolve_recall_candidate(recall_id_2, second, accept=True, reason="test-accept")

        entry2 = wm3.rule_registry.get(rule_id)
        stats2 = (entry2 or {}).get("accounting") or {}
        assert int(stats2.get("recall_rejected", 0)) >= 1
        assert int(stats2.get("recall_accepted", 0)) >= 1
    print("Rule accounting + recall accept/reject validation passed.")


if __name__ == "__main__":
    main()
