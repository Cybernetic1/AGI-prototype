import json
import tempfile
import unittest
from pathlib import Path

from working_memory.ontology_seed_alignment import align_semantic_ar_examples


class OntologySeedAlignmentTest(unittest.TestCase):
    def test_align_semantic_ar_examples_matches_ontology_symbols(self):
        payload = {
            "examples": [{
                "example_id": "demo_001",
                "category": "physical_inventory",
                "trace": [{"step": "observe", "proposition": "Has(alex, apples, 18)"}],
                "target": "Remaining(alex, apples, 16)",
                "seed_rules": [
                    "Has(x,item,q0) & StateTransition(x,item,delta=d1) -> Has(x,item,q0+d1)",
                    "Has(x,item,q1) & StateTransition(x,item,delta=d2) -> Remaining(x,item,q1+d2)",
                ],
            }]
        }
        ontology = {
            "lemma_index": {
                "have": {"event_label": "transfer", "event_confidence": 0.9, "sources": ["wordnet"]},
                "remain": {"event_label": "comparison", "event_confidence": 0.8, "sources": ["verbnet"]},
                "state_transition": {"event_label": "comparison", "event_confidence": 0.7, "sources": ["framenet"]},
            }
        }

        aligned = align_semantic_ar_examples(payload, ontology)
        self.assertEqual(aligned["examples"][0]["ontology_alignment"]["target"]["chosen_concept"], "comparison")
        self.assertFalse(aligned["examples"][0]["ontology_alignment"]["seed_rules"][0]["needs_human_review"])

    def test_align_semantic_ar_examples_flags_unknown_symbol_for_review(self):
        payload = {"examples": [{
            "example_id": "demo_002",
            "category": "social_intent",
            "trace": [{"step": "observe", "proposition": "Intend(alex, share_food)"}],
            "target": "Reason(alex, share_food)",
            "seed_rules": ["Intend(x, action) -> Reason(x, action)"],
        }]}
        ontology = {"lemma_index": {}}

        aligned = align_semantic_ar_examples(payload, ontology)
        self.assertTrue(aligned["examples"][0]["ontology_alignment"]["seed_rules"][0]["needs_human_review"])
        self.assertGreaterEqual(aligned["ontology_alignment_summary"]["review_required_examples"], 1)


if __name__ == "__main__":
    unittest.main()
