import unittest

from working_memory.working_memory import WMNode, WorkingMemory


class ModalContextGuardsTest(unittest.TestCase):
    def test_state_authority_does_not_leak_across_contexts(self):
        wm = WorkingMemory()
        c1 = WMNode.create(
            "Quantity",
            owner="alice",
            item="apple",
            qty=3,
            event_order=1,
            ctx_id="ctx:actual",
            source="system2",
            layer="semantic",
        )
        c2 = WMNode.create(
            "Quantity",
            owner="alice",
            item="apple",
            qty=9,
            event_order=2,
            ctx_id="ctx:hypo",
            source="system2",
            layer="semantic",
        )
        wm.update([c1, c2])

        vals_actual = {
            f.to_dict().get("qty")
            for f in wm.facts
            if f.concept in {"Quantity", "FinalState", "PreferredState"}
            and f.to_dict().get("owner") == "alice"
            and f.to_dict().get("item") == "apple"
            and f.to_dict().get("ctx_id") == "ctx:actual"
        }
        vals_hypo = {
            f.to_dict().get("qty")
            for f in wm.facts
            if f.concept in {"Quantity", "FinalState", "PreferredState"}
            and f.to_dict().get("owner") == "alice"
            and f.to_dict().get("item") == "apple"
            and f.to_dict().get("ctx_id") == "ctx:hypo"
        }
        self.assertIn(3, vals_actual)
        self.assertIn(9, vals_hypo)

    def test_context_wmes_emit_predicate_atoms_and_accessibility(self):
        wm = WorkingMemory()
        wm.add_context(ctx_id="ctx:actual", ctx_type="actual")
        wm.add_context(ctx_id="ctx:belief:bob", ctx_type="belief", parent_ctx="ctx:actual", agent="bob")
        wm.add_accessibility(from_ctx="ctx:actual", to_ctx="ctx:belief:bob", modality="belief")
        wm.add_proposition(prop_id="p:1", pred="has", args=["alice", "apple", 3])
        wm.add_in_context(prop_id="p:1", ctx_id="ctx:actual")

        atoms = wm.to_predicate_atoms()
        rendered = {str(a) for a in atoms}

        self.assertTrue(any("ctx(" in item for item in rendered))
        self.assertTrue(any("accessible(" in item for item in rendered))
        self.assertTrue(wm.contexts_accessible("ctx:actual", "ctx:belief:bob", "belief"))

    def test_final_state_motif_is_context_local(self):
        wm = WorkingMemory()
        wm.update(
            [
                WMNode.create("StateTransition", owner="alice", item="apple", after_qty=5, event_order=1, ctx_id="ctx:a"),
                WMNode.create("StateTransition", owner="alice", item="apple", after_qty=8, event_order=2, ctx_id="ctx:b"),
            ]
        )
        self.assertEqual(wm.final_state_in_context("alice", "apple", "ctx:a"), 5)
        self.assertEqual(wm.final_state_in_context("alice", "apple", "ctx:b"), 8)


if __name__ == "__main__":
    unittest.main()
