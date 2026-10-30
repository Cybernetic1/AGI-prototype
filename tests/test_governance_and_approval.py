import tempfile
import unittest
from pathlib import Path

from working_memory.ltm import LongTermMemory
from working_memory.rule_registry import RuleRegistry


class GovernanceAndApprovalTest(unittest.TestCase):
    def test_rule_approval_metadata_tracks_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = RuleRegistry(Path(tmpdir) / "rules.yaml")
            entry = registry.register_proposal(
                rule_id="rule-approval-test",
                raw_rule="If the later state is preferred, keep it.",
                source="teacher",
                compiled_payload={"family": "finality", "condition": "later_state", "effect": "prefer_later", "priority": 10},
            )
            self.assertEqual(entry["approval"]["status"], "proposed")

            updated = registry.transition(
                "rule-approval-test",
                "approved",
                reviewer="reviewer-1",
                reason="quality gate pass",
            )
            self.assertEqual(updated["approval"]["status"], "approved")
            self.assertEqual(updated["approval"]["reviewer"], "reviewer-1")

    def test_ltm_quality_drift_and_rollback_anchor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ltm = LongTermMemory(Path(tmpdir) / "ltm.yaml")
            ltm.record_quality_snapshot(suite="gsm", metrics={"symbolic_validity": 0.96, "solve_rate": 0.77})
            ltm.record_quality_snapshot(suite="gsm", metrics={"symbolic_validity": 0.88, "solve_rate": 0.72})

            drift = ltm.check_quality_drift(
                suite="gsm",
                current_metrics={"symbolic_validity": 0.88},
                quality_metric="symbolic_validity",
                drift_threshold=0.05,
            )
            self.assertTrue(drift["drift_detected"])

            rollback = ltm.rollback_if_drift(
                suite="gsm",
                current_metrics={"symbolic_validity": 0.88},
                quality_metric="symbolic_validity",
                drift_threshold=0.05,
            )
            self.assertTrue(rollback["rolled_back"])
            self.assertIn("rollback_anchor", ltm._load())


if __name__ == "__main__":
    unittest.main()
