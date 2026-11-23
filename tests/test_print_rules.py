import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "rules" / "print_rules.py"
SPEC = importlib.util.spec_from_file_location("print_rules_module", MODULE_PATH)
print_rules = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(print_rules)


class PrintRulesTest(unittest.TestCase):
    def test_print_rule_snapshot_lists_concrete_rule_body(self):
        payload = {
            "rules": [{
                "family": "ratio_compose",
                "name": "ratio_compose",
                "head": ["quantity", ["?owner", "?item", "?ans"]],
                "body": [
                    ["ratio", ["?owner", "?ref_owner", "?item", "?factor"]],
                    ["quantity", ["?ref_owner", "?item", "?base"]],
                ],
                "calc": "base * factor",
                "target": "?ans",
                "description": "same-item ratio composition from a reference owner",
            }]
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "metarules_snapshot_v1.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            output = io.StringIO()

            with contextlib.redirect_stdout(output):
                print_rules.print_rule_snapshot(path)

        text = output.getvalue()
        self.assertIn("ratio_compose", text)
        self.assertIn("base * factor", text)
        self.assertIn("same-item ratio composition", text)

    def test_print_nl_rule_registry_keeps_raw_and_compiled_rule_text(self):
        payload = {
            "rules": [{
                "rule_id": "teacher-prop-prefer-latest",
                "status": "proposed",
                "raw_rule": "If a later state exists for the same owner and item, prefer the later state.",
                "compiled": {
                    "family": "prefer_latest_state",
                    "condition": "later_state_exists_for_same_owner_and_item",
                    "effect": "prefer_later_state_over_earlier_quantity",
                    "rule_text": "later_state(s1, s2) and same_owner_item(X, Y) -> prefer(s2, X, Y)",
                },
            }]
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "nl_rule_registry.yaml"
            path.write_text(json.dumps(payload), encoding="utf-8")
            output = io.StringIO()

            with contextlib.redirect_stdout(output):
                print_rules.print_nl_rule_registry(path)

        text = output.getvalue()
        self.assertIn("If a later state exists", text)
        self.assertIn("later_state(s1, s2)", text)
        self.assertIn("prefer_later_state_over_earlier_quantity", text)

    def test_print_rule_inventory_summary_reports_total_materialized_count(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            print_rules.print_rule_inventory_summary()
        text = output.getvalue()
        self.assertIn("System 2 file-backed rules: 4", text)
        self.assertIn("Defeasible rules: 3", text)
        self.assertIn("NL registry compiled rules: 1", text)
        self.assertIn("Metarule snapshot entries: 18", text)
        self.assertIn("Total materialized rule/metarule count: 34", text)

    def test_print_sys2_experta_rules_lists_code_defined_rules(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            print_rules.print_sys2_experta_rules(print_rules.SYS2_EXPERTA_FILE)
        text = output.getvalue()
        self.assertIn("System 2 Experta rules (code-defined)", text)
        self.assertIn("count: 36", text)
        self.assertIn("execute_transfer", text)
        self.assertIn("action:", text)
        self.assertIn("self.modify(", text)


if __name__ == "__main__":
    unittest.main()
