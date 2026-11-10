import unittest
import tempfile
from pathlib import Path

from working_memory.basic_english_grounding import (
    bootstrap_grounding,
    curate_noun_knowledge,
    curate_verb_knowledge,
    extract_unknown_terms,
    generate_pos_grounding_candidate,
    ground_term,
    ingest_pos_lexicon,
    propose_verb_frames,
    propose_noun_facts,
)
from working_memory.ltm import LongTermMemory


class BasicEnglishGroundingTest(unittest.TestCase):
    def test_extract_unknown_terms_finds_unseen_words(self):
        unknown = extract_unknown_terms("A student bought a curious gadget and had a remainder after lunch.")
        self.assertIn("student", unknown)
        self.assertIn("curious", unknown)
        self.assertIn("gadget", unknown)
        self.assertNotIn("bought", unknown)
        self.assertNotIn("after", unknown)

    def test_ground_term_maps_basic_story_words_to_canonical_concepts(self):
        term = ground_term("remaining")
        self.assertEqual(term["concept"], "remaining_quantity")
        self.assertIn("remain", term["gloss"].lower())

    def test_bootstrap_grounding_creates_inspectable_semantic_graph(self):
        graph = bootstrap_grounding("A student bought a curious gadget and some apples remained after lunch.")
        self.assertIn("unknown_terms", graph)
        self.assertTrue(graph["nodes"])
        self.assertIn("student", graph["unknown_terms"])
        self.assertIn("schema", graph)
        self.assertTrue(any(item.get("concept") == "remaining_quantity" for item in graph["grounded_terms"]))

    def test_pos_grounding_candidate_maps_verb_semantics(self):
        cand = generate_pos_grounding_candidate(
            word="distribute",
            pos="verb",
            gloss="to give things out to several people",
        )
        self.assertEqual(cand["kind"], "state_transition")
        self.assertEqual(cand["concept"], "transfer")
        self.assertGreaterEqual(cand["confidence"], 0.8)

    def test_ingest_pos_lexicon_registers_candidates(self):
        result = ingest_pos_lexicon(
            [
                {"word": "distribute", "pos": "verb", "gloss": "to give out shares"},
                {"word": "sum", "pos": "noun", "gloss": "a total amount"},
                {"word": "", "pos": "noun", "gloss": "invalid"},
            ],
            register=True,
            overwrite=True,
        )
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["registered"], 2)
        self.assertEqual(result["skipped"], 1)
        self.assertEqual(ground_term("distribute")["concept"], "transfer")
        self.assertEqual(ground_term("sum")["concept"], "sum_total")

    def test_propose_noun_facts_returns_five_structured_candidates(self):
        facts = propose_noun_facts(noun="apple", gloss="a sweet red fruit")
        self.assertLessEqual(len(facts), 5)
        self.assertGreaterEqual(len(facts), 3)
        attrs = {f["attribute"] for f in facts}
        self.assertIn("isa", attrs)
        self.assertTrue(any(f["value"] == "fruit" for f in facts if f["attribute"] == "isa"))

    def test_curate_noun_knowledge_persists_vetted_facts_in_ltm(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ltm = LongTermMemory(Path(tmpdir) / "ltm.yaml")

            def reviewer(noun, candidate):
                if candidate["attribute"] == "color":
                    return False
                return True

            report = curate_noun_knowledge(
                noun="apple",
                gloss="a sweet red fruit",
                decision_callback=reviewer,
                ltm=ltm,
                max_facts=5,
                overwrite=True,
            )
            self.assertTrue(report["store"]["updated"])
            recalled = ltm.recall_noun_facts("apple", max_facts=5)
            self.assertTrue(recalled)
            self.assertFalse(any(f["attribute"] == "color" for f in recalled))
            self.assertTrue(any(f["attribute"] == "isa" and f["value"] == "fruit" for f in recalled))

    def test_propose_verb_frames_maps_transfer_paradigm(self):
        frames = propose_verb_frames(verb="give", gloss="to transfer something to another person", max_frames=4)
        self.assertTrue(frames)
        self.assertTrue(
            any(
                f.get("paradigm") == "transfer_possession_change" and f.get("frame") == "transfer"
                for f in frames
            )
        )

    def test_curate_verb_knowledge_persists_vetted_frames_in_ltm(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ltm = LongTermMemory(Path(tmpdir) / "ltm.yaml")

            def reviewer(verb, candidate):
                return candidate["paradigm"] != "existence_state"

            report = curate_verb_knowledge(
                verb="give",
                gloss="to transfer possession to someone",
                decision_callback=reviewer,
                ltm=ltm,
                max_frames=5,
                overwrite=True,
            )
            self.assertTrue(report["store"]["updated"])
            recalled = ltm.recall_verb_frames("give", max_frames=5)
            self.assertTrue(recalled)
            self.assertFalse(any(f["paradigm"] == "existence_state" for f in recalled))
            self.assertTrue(any(f["paradigm"] == "transfer_possession_change" for f in recalled))


if __name__ == "__main__":
    unittest.main()
