import unittest

from working_memory.semantic_schema import SemanticGraphSchema
from working_memory.working_memory import WorkingMemory


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


if __name__ == "__main__":
    unittest.main()
