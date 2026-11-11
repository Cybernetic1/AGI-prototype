import tempfile
import unittest
from pathlib import Path

from working_memory.ltm import LongTermMemory
from working_memory.working_memory import WMNode


class LTMSemanticMemoryRecallTest(unittest.TestCase):
    def test_semantic_memory_contributes_when_episodes_archived(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ltm = LongTermMemory(Path(tmpdir) / "ltm.yaml")
            fact = WMNode.create("Remaining", owner="bob", item="apples", qty=7)
            ltm.capture_episode([fact], reward=1.0, outcome="solved", extra_cues=["owner:bob", "item:apples"])
            ltm.capture_episode([fact], reward=0.8, outcome="solved", extra_cues=["owner:bob", "item:apples"])
            report = ltm.consolidate(min_support=2)
            self.assertGreaterEqual(report["semantic_count"], 1)

            payload = ltm._load()
            for ep in payload.get("episodes", []):
                ep["archived"] = True
            ltm._save(payload)

            recalled = ltm.associative_recall(
                ["owner:bob", "item:apples"],
                top_k=2,
                min_score=0.01,
                include_semantic_memory=True,
            )
            props = recalled.get("propositions", [])
            self.assertTrue(props)
            self.assertGreaterEqual(int(recalled.get("semantic_hits", 0)), 1)
            self.assertTrue(any((p.get("attrs") or {}).get("owner") == "bob" for p in props))


if __name__ == "__main__":
    unittest.main()
