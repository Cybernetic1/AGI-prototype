import unittest

from working_memory.working_memory import WMNode, WorkingMemory


class WorkingMemoryDedupTest(unittest.TestCase):
    def test_duplicate_state_facts_are_coalesced(self):
        wm = WorkingMemory()
        first = WMNode.create(
            "Quantity",
            owner="alice",
            item="apple",
            qty=3,
            event_order=1,
            source="system1",
            layer="reasoning",
        )
        second = WMNode.create(
            "Quantity",
            owner="alice",
            item="apple",
            qty=3,
            event_order=1,
            source="system2",
            layer="semantic",
        )

        wm.update([first, second])

        quantity_fact_count = sum(
            1
            for f in wm.facts
            if f.concept == "Quantity"
            and f.to_dict().get("owner") == "alice"
            and f.to_dict().get("item") == "apple"
            and f.to_dict().get("qty") == 3
        )
        self.assertEqual(quantity_fact_count, 1)

    def test_stale_state_facts_are_pruned_after_newer_authority(self):
        wm = WorkingMemory()
        older = WMNode.create(
            "Quantity",
            owner="alice",
            item="apple",
            qty=3,
            event_order=1,
            source="system1",
        )
        newer = WMNode.create(
            "FinalState",
            owner="alice",
            item="apple",
            qty=4,
            event_order=2,
            source="state_machine",
            phase="final",
        )

        wm.update([older, newer])

        values = {
            f.to_dict().get("qty")
            for f in wm.facts
            if f.to_dict().get("owner") == "alice" and f.to_dict().get("item") == "apple"
        }
        self.assertNotIn(3, values)
        self.assertIn(4, values)


if __name__ == "__main__":
    unittest.main()
