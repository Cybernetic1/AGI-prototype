import tempfile
import unittest
from pathlib import Path

from working_memory.rule_registry import RuleRegistry


class ModalRuleValidationTest(unittest.TestCase):
    def test_modal_rule_fails_without_context_anchors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = RuleRegistry(Path(tmpdir) / "rules.yaml")
            registry.register_proposal(
                rule_id="rule-bad-modal",
                raw_rule="bad modal",
                source="human",
                compiled_payload={
                    "rule_id": "rule-bad-modal",
                    "family": "belief_transfer",
                    "condition": "belief exists",
                    "effect": "copy it",
                    "priority": 10,
                },
                activation_policy="nl_compiler_validator",
            )
            registry.transition("rule-bad-modal", "approved", reviewer="test", reason="setup")
            with self.assertRaises(ValueError):
                registry.transition("rule-bad-modal", "active", reviewer="test", reason="should fail semantic anchors")

    def test_modal_rule_activates_with_required_anchors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = RuleRegistry(Path(tmpdir) / "rules.yaml")
            entry = registry.register_proposal(
                rule_id="rule-good-modal",
                raw_rule="belief transfer from context to context",
                source="human",
                compiled_payload={
                    "rule_id": "rule-good-modal",
                    "family": "belief_transfer",
                    "condition": "from_ctx to_ctx belief modality allows transfer",
                    "effect": "transfer prop_id to target_ctx",
                    "priority": 10,
                },
                activation_policy="nl_compiler_validator",
            )
            self.assertEqual(entry.get("status"), "proposed")
            registry.transition("rule-good-modal", "approved", reviewer="test", reason="setup")
            activated = registry.transition("rule-good-modal", "active", reviewer="test", reason="semantic anchors valid")
            self.assertEqual(activated.get("status"), "active")


if __name__ == "__main__":
    unittest.main()
