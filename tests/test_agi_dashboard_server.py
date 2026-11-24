import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "agi_dashboard_server.py"
SPEC = importlib.util.spec_from_file_location("agi_dashboard_server_module", MODULE_PATH)
dashboard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dashboard)


class DashboardServerTest(unittest.TestCase):
    def test_lt_launch_command_uses_allowlisted_trainer(self):
        dataset = REPO_ROOT / "rules" / "reports" / "tinystories_symbolic_compact_v1.jsonl"
        command = dashboard.build_launch_command(
            {
                "mode": "lt_train",
                "dataset": str(dataset),
                "epochs": 2,
                "holdout": 0.2,
                "threads": 2,
                "resume": True,
                "checkpoint": "checkpoints/dashboard-smoke.pt",
            }
        )
        self.assertEqual(Path(command[1]), REPO_ROOT / "logic_transformer" / "train.py")
        self.assertIn("--resume", command)
        self.assertNotIn("shell=True", command)

    def test_launch_paths_cannot_escape_repository(self):
        with self.assertRaises(ValueError):
            dashboard.build_launch_command(
                {
                    "mode": "lt_train",
                    "dataset": "/tmp/outside.jsonl",
                    "checkpoint": "checkpoints/test.pt",
                }
            )

    def test_unknown_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "mode must be"):
            dashboard.build_launch_command({"mode": "arbitrary_shell"})

    def test_rule_payload_includes_code_and_file_rules(self):
        payload = dashboard._rules_payload()
        self.assertEqual(payload["counts"]["experta"], 36)
        self.assertEqual(payload["counts"]["canonical_sys2"], 4)
        self.assertEqual(payload["counts"]["defeasible"], 3)
        self.assertTrue(payload["experta"])

    def test_dataset_payload_only_lists_repository_jsonl_files(self):
        payload = dashboard._dataset_payload()
        self.assertTrue(payload["datasets"])
        self.assertTrue(all(Path(row["path"]).suffix == ".jsonl" for row in payload["datasets"]))


if __name__ == "__main__":
    unittest.main()
