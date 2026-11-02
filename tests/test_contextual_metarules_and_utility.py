import unittest

from rules.contextual_metarules import build_context_policy, candidate_matches_policy
from rules.ilp_rule_utility import apply_reward_updates, utility_components


class ContextualMetarulesAndUtilityTest(unittest.TestCase):
    def test_context_policy_accepts_matching_family_rule(self):
        policy = build_context_policy(
            question="A has apples, gives away some, and 20 percent are eaten. How many remain?",
            family="buy_more_final_gap",
        )
        candidate = {
            "family": "remaining_quantity",
            "head": ("remaining_quantity", ["?owner", "?item", "?ans"]),
            "body": [("quantity", ["?owner", "?item", "?base"]), ("percent", ["?owner", "?item", "?pct"])],
            "calc": "base * (1 - pct)",
        }
        ok, reasons = candidate_matches_policy(candidate, policy)
        self.assertTrue(ok)
        self.assertEqual(reasons, [])

    def test_context_policy_rejects_out_of_scope_predicates(self):
        policy = build_context_policy(
            question="A buys and donates apples then some are spoiled.",
            family="buy_more_final_gap",
        )
        candidate = {
            "family": "unit_compatibility",
            "head": ("unit_compatibility", ["?v", "?from", "?to", "?ans"]),
            "body": [("unit_rate", ["?v", "?from", "?to"]), ("unitconvert", ["?v", "?from", "?to", "?ans"])],
            "calc": "ans",
        }
        ok, reasons = candidate_matches_policy(candidate, policy)
        self.assertFalse(ok)
        self.assertTrue(any("not_allowed" in r for r in reasons))

    def test_utility_updates_increase_average_reward(self):
        state = {"meta": {"total_updates": 0}, "rules": {}}
        signature = "sig-1"
        before = utility_components(state, signature)
        self.assertEqual(before["value"], 0.0)
        apply_reward_updates(
            state,
            [
                {"signature": signature, "reward": 1.0},
                {"signature": signature, "reward": 0.5},
            ],
        )
        after = utility_components(state, signature)
        self.assertGreater(after["value"], 0.0)
        self.assertGreater(after["count"], 0.0)


if __name__ == "__main__":
    unittest.main()
