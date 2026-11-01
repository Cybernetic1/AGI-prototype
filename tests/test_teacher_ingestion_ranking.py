import tempfile
import unittest
from pathlib import Path

from working_memory.rule_registry import RuleRegistry
from working_memory.working_memory import WorkingMemory


class TeacherIngestionRankingTest(unittest.TestCase):
    def test_greedy_ranking_prefers_higher_score_rule(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wm = WorkingMemory(default_ttl=3)
            wm.rule_registry = RuleRegistry(Path(tmpdir) / "rules.yaml")
            proposals = {
                "rules": [
                    {
                        "rule_id": "rule-low",
                        "raw_rule": "If same owner and same item are mentioned, bind them to the same identity.",
                        "score": 0.1,
                    },
                    {
                        "rule_id": "rule-high",
                        "raw_rule": "If a later state exists for the same owner and item, prefer the later state.",
                        "score": 0.9,
                    },
                ]
            }
            report = wm.ingest_teacher_proposals(
                proposals,
                source="teacher_llm",
                ab_arm="treatment",
                enable_ingestion=True,
                low_volume_mode=False,
                max_total=1,
                max_rules=1,
                max_facts=0,
                max_counterfactuals=0,
                greedy_by_score=True,
                min_score=0.0,
            )
            accepted_rules = [
                e.get("rule_id")
                for e in report.get("events", [])
                if e.get("proposal_kind") == "rule" and e.get("decision") == "accepted"
            ]
            self.assertEqual(accepted_rules, ["rule-high"])

    def test_min_score_rejects_low_scored_rules(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wm = WorkingMemory(default_ttl=3)
            wm.rule_registry = RuleRegistry(Path(tmpdir) / "rules.yaml")
            proposals = {
                "rules": [
                    {
                        "rule_id": "rule-low",
                        "raw_rule": "If same owner and same item are mentioned, bind them to the same identity.",
                        "score": 0.05,
                    }
                ]
            }
            report = wm.ingest_teacher_proposals(
                proposals,
                source="teacher_llm",
                ab_arm="treatment",
                enable_ingestion=True,
                low_volume_mode=False,
                max_total=1,
                max_rules=1,
                max_facts=0,
                max_counterfactuals=0,
                greedy_by_score=True,
                min_score=0.1,
            )
            self.assertEqual(report["counts"]["accepted"], 0)
            self.assertEqual(report["counts"]["rejected"], 1)


if __name__ == "__main__":
    unittest.main()
