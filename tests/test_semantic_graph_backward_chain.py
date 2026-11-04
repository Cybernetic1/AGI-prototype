import unittest

from cognitive_loop import _lift_semantic_candidates, score_wme_fact
from working_memory.semantic_schema import SemanticGraphSchema
from working_memory.working_memory import WMNode, WorkingMemory


class SemanticGraphBackwardChainTest(unittest.TestCase):
    def test_canonical_graph_can_backchain_generic_value_relationships(self):
        graph = SemanticGraphSchema.make_semantic_graph(
            nodes=[
                {"id": "entity:alice", "kind": "entity", "name": "alice"},
                {"id": "qty:alice:apples", "kind": "quantity", "entity_id": "entity:alice", "value": 5, "unit": "apple"},
            ],
            edges=[
                {"source": "entity:alice", "target": "qty:alice:apples", "relation": "has_value"},
                {"source": "qty:alice:apples", "target": "entity:alice", "relation": "value_of"},
            ],
        )

        matches = SemanticGraphSchema.backward_chain(
            graph,
            target="entity:alice",
            relation="has_value",
            max_depth=2,
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["source"], "entity:alice")
        self.assertEqual(matches[0]["target"], "qty:alice:apples")
        self.assertEqual(matches[0]["relation"], "has_value")

    def test_working_memory_exposes_same_layer_for_querying(self):
        graph = SemanticGraphSchema.make_semantic_graph(
            nodes=[
                {"id": "entity:sum", "kind": "entity", "name": "sum"},
                {"id": "qty:sum:total", "kind": "quantity", "entity_id": "entity:sum", "value": 8, "unit": "item"},
            ],
            edges=[
                {"source": "entity:sum", "target": "qty:sum:total", "relation": "has_value"},
            ],
        )

        wm = WorkingMemory()
        wm.add_semantic_graph(graph)
        matches = wm.query_semantic_graph(target="entity:sum", relation="has_value", max_depth=2)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["relation"], "has_value")
        self.assertEqual(matches[0]["target"], "qty:sum:total")

    def test_percent_remaining_phrase_is_decomposed_into_finer_primitives(self):
        facts = _lift_semantic_candidates("John had 40 apples. He gave away 20% of the remaining apples.")
        concepts = {fact.concept for fact in facts}

        self.assertIn("Quantity", concepts)
        self.assertIn("Percent", concepts)
        self.assertIn("Remaining", concepts)
        self.assertIn("StateTransition", concepts)

        percent_fact = next(f for f in facts if getattr(f, "concept", None) == "Percent")
        self.assertAlmostEqual(percent_fact.to_dict()["pct"], 0.2)

        transition_fact = next(f for f in facts if getattr(f, "concept", None) == "StateTransition")
        self.assertEqual(transition_fact.to_dict()["event_kind"], "percent_of_remaining")
        self.assertAlmostEqual(transition_fact.to_dict()["delta"], -8.0)

    def test_wme_heuristic_score_is_nonzero_for_intermediate_numeric_facts(self):
        fact = WMNode.create(
            "Remaining",
            activation=0.85,
            layer="semantic",
            source="state_machine",
            owner="alice",
            item="apples",
            qty=12,
        )

        score = score_wme_fact(fact, question="How many apples remain?")
        self.assertGreater(score, 0.5)
        self.assertLess(score, 1.0)


if __name__ == "__main__":
    unittest.main()
