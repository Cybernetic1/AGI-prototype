import unittest

from working_memory.ontology_hyperbolic_prior import (
    build_hyperbolic_prior_from_unified,
    lemma_prior_score,
)


class OntologyHyperbolicPriorTest(unittest.TestCase):
    def test_build_prior_from_unified(self):
        payload = {
            "lemma_index": {
                "give": {"event_label": "transfer", "event_confidence": 0.9, "sources": ["wordnet", "verbnet"]},
                "buy": {"event_label": "creation", "event_confidence": 0.8, "sources": ["wordnet"]},
            }
        }
        prior = build_hyperbolic_prior_from_unified(payload, dim=12)
        self.assertEqual(prior["lemma_count"], 2)
        self.assertIn("give", prior["lemmas"])
        self.assertEqual(len(prior["lemmas"]["give"]["vector"]), 12)

    def test_lemma_prior_score(self):
        prior = {
            "lemmas": {
                "give": {"label": "transfer", "confidence": 0.8},
            }
        }
        score, label, conf = lemma_prior_score(prior, "give")
        self.assertGreater(score, 0.8)
        self.assertEqual(label, "transfer")
        self.assertAlmostEqual(conf, 0.8, places=6)


if __name__ == "__main__":
    unittest.main()
