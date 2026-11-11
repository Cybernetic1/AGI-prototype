import unittest
from unittest.mock import patch

from working_memory.ontology_unifier import build_unified_ontology, export_embedding_edges


class OntologyUnifierTest(unittest.TestCase):
    def test_build_unified_ontology_aligns_sources(self):
        with (
            patch("working_memory.ontology_unifier.import_wordnet", return_value=[
                {"source": "wordnet", "lemma": "give", "sense_id": "give.v.01", "event_label": "transfer"},
                {"source": "wordnet", "lemma": "buy", "sense_id": "buy.v.01", "event_label": "creation"},
            ]),
            patch("working_memory.ontology_unifier.import_verbnet", return_value=[
                {"source": "verbnet", "lemma": "give", "sense_id": "give-13.1", "event_label": "transfer"},
            ]),
            patch("working_memory.ontology_unifier.import_framenet", return_value=[
                {"source": "framenet", "lemma": "give", "sense_id": "Giving", "event_label": "transfer"},
            ]),
        ):
            payload = build_unified_ontology()

        self.assertEqual(payload["record_count"], 4)
        self.assertIn("give", payload["lemma_index"])
        give = payload["lemma_index"]["give"]
        self.assertEqual(give["event_label"], "transfer")
        self.assertGreaterEqual(give["event_confidence"], 0.9)
        self.assertTrue(any(edge["relation"] == "same_lemma" for edge in payload["alignment_edges"]))

    def test_export_embedding_edges_contains_event_label_edges(self):
        payload = {
            "lemma_index": {
                "give": {
                    "sources": ["wordnet", "verbnet"],
                    "event_label": "transfer",
                    "event_confidence": 0.8,
                }
            },
            "alignment_edges": [
                {"from": "lemma:give:wordnet", "to": "lemma:give:verbnet", "relation": "same_lemma", "confidence": 0.9}
            ],
        }
        rows = export_embedding_edges(payload)
        self.assertTrue(any(r[0] == "lemma:give" and r[1] == "event_label:transfer" for r in rows))
        self.assertTrue(any(r[2] == "same_lemma" for r in rows))


if __name__ == "__main__":
    unittest.main()
