import tempfile
import unittest
from pathlib import Path

from working_memory.ltm import LongTermMemory


class LTMOntologyTriplesTest(unittest.TestCase):
    def test_upsert_and_recall_ontology_triples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ltm = LongTermMemory(Path(tmpdir) / "ltm.yaml")
            report = ltm.upsert_ontology_triples(
                triples=[
                    {"head": "give", "relation": "event_label", "tail": "transfer", "confidence": 0.9},
                    {"head": "give", "relation": "same_lemma", "tail": "donate", "confidence": 0.7},
                    {"head": "buy", "relation": "event_label", "tail": "creation", "confidence": 0.8},
                ],
                overwrite=True,
            )
            self.assertTrue(report["updated"])
            self.assertEqual(report["count"], 3)

            give_edges = ltm.recall_ontology_triples(lemma="give", max_edges=10, min_confidence=0.0)
            self.assertEqual(len(give_edges), 2)
            self.assertTrue(all(edge["head"] == "give" for edge in give_edges))
            self.assertGreaterEqual(float(give_edges[0]["confidence"]), float(give_edges[1]["confidence"]))


if __name__ == "__main__":
    unittest.main()
