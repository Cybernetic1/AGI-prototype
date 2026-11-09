import unittest

from rules.ilp_candidate_pipeline import (
    CandidateRecord,
    ILPPipelineConfig,
    candidate_signature,
    prescore_and_filter,
    run_ilp_learning_stage,
    selective_llm_validate,
)


class ILPCandidatePipelineTest(unittest.TestCase):
    def test_candidate_signature_is_stable(self):
        candidate = {
            "family": "remaining_quantity",
            "head": ("remaining_quantity", ["?owner", "?item", "?ans"]),
            "body": [("quantity", ["?owner", "?item", "?base"]), ("percent", ["?owner", "?item", "?pct"])],
            "calc": "base * (1 - pct)",
            "target": "?ans",
        }
        sig1 = candidate_signature(candidate)
        sig2 = candidate_signature(dict(candidate))
        self.assertEqual(sig1, sig2)

    def test_prescore_respects_threshold_and_per_family_budget(self):
        cfg = ILPPipelineConfig(
            candidate_top_n=6,
            prescore_min=0.6,
            max_candidates_per_family=1,
            llm_enabled=False,
        )
        candidates = [
            {
                "family": "state_transition",
                "head": ("state", ["?owner", "?item", "?ans"]),
                "body": [("state_transition", ["?owner", "?item", "?before", "?delta", "?after"])],
                "calc": "before + delta",
                "ranking": {"score": 0.95, "coverage": 1.0, "simplicity": 1.0, "symbolic_validity": 1.0, "constraint_validity": 1.0},
            },
            {
                "family": "state_transition",
                "head": ("state", ["?owner", "?item", "?ans2"]),
                "body": [("state_transition", ["?owner", "?item", "?before", "?delta", "?after"])],
                "calc": "before + delta",
                "ranking": {"score": 0.92, "coverage": 1.0, "simplicity": 1.0, "symbolic_validity": 1.0, "constraint_validity": 1.0},
            },
            {
                "family": "remaining_quantity",
                "head": ("remaining_quantity", ["?owner", "?item", "?ans"]),
                "body": [("quantity", ["?owner", "?item", "?base"])],
                "calc": "base",
                "ranking": {"score": 0.1, "coverage": 0.4, "simplicity": 0.5, "symbolic_validity": 0.5, "constraint_validity": 0.2},
            },
        ]
        accepted, dropped = prescore_and_filter(candidates, cfg=cfg)
        self.assertEqual(len(dropped), 0)
        self.assertEqual(len(accepted), 3)
        reasons = {a.dropped_reason for a in accepted}
        self.assertIn("per_family_budget", reasons)
        self.assertTrue(any(reason.startswith("below_prescore_soft:") for reason in reasons))

    def test_selective_validator_respects_latency_budget(self):
        cfg = ILPPipelineConfig(
            llm_enabled=True,
            llm_validate_max=3,
            llm_latency_budget_sec=0.0,
        )
        recs = [
            CandidateRecord("c1", "state_transition", "sig1", {"family": "state_transition"}, 0.9),
            CandidateRecord("c2", "state_transition", "sig2", {"family": "state_transition"}, 0.8),
        ]
        accepted, rejected, metrics = selective_llm_validate(recs, question="q", cfg=cfg, cache={})
        self.assertEqual(len(accepted), 2)
        self.assertEqual(len(rejected), 2)
        self.assertTrue(metrics["latency_budget_exhausted"])
        self.assertTrue(all(rec.llm_reason == "llm_latency_budget_exhausted" for rec in rejected))

    def test_selective_validator_uses_cache(self):
        calls = {"count": 0}

        def fake_validator_call(**kwargs):
            calls["count"] += 1
            return {"valid": True, "score": 0.9, "reason": "ok"}, ""

        cfg = ILPPipelineConfig(
            llm_enabled=True,
            llm_validate_max=1,
            llm_latency_budget_sec=10.0,
        )
        cache = {}
        rec = CandidateRecord("c1", "state_transition", "sig-cache", {"family": "state_transition"}, 0.9)

        accepted1, rejected1, _ = selective_llm_validate([rec], question="q1", cfg=cfg, cache=cache, validator_call=fake_validator_call)
        self.assertEqual(len(accepted1), 1)
        self.assertEqual(len(rejected1), 0)
        self.assertEqual(calls["count"], 1)

        rec2 = CandidateRecord("c2", "state_transition", "sig-cache", {"family": "state_transition"}, 0.9)
        accepted2, rejected2, metrics2 = selective_llm_validate([rec2], question="q2", cfg=cfg, cache=cache, validator_call=fake_validator_call)
        self.assertEqual(len(accepted2), 1)
        self.assertEqual(len(rejected2), 0)
        self.assertEqual(calls["count"], 1)
        self.assertEqual(metrics2["cache_hits"], 1)

    def test_prescore_drops_malformed_candidates_only(self):
        cfg = ILPPipelineConfig(candidate_top_n=4, llm_enabled=False)
        candidates = [
            {
                "family": "state_transition",
                "head": ("state", ["?owner", "?item", "?ans"]),
                "body": [("state_transition", ["?owner", "?item", "?before", "?delta", "?after"])],
                "calc": "before + delta",
                "ranking": {"score": 0.9},
            },
            {
                "family": "state_transition",
                "head": ("state", ["?owner", "?item", "?ans"]),
                "body": [],
                "calc": "before + delta",
                "ranking": {"score": 0.9},
            },
        ]
        accepted, dropped = prescore_and_filter(candidates, cfg=cfg)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0].dropped_reason, "empty_body")

    def test_run_stage_accepts_external_candidates(self):
        cfg = ILPPipelineConfig(
            candidate_top_n=4,
            prescore_min=0.3,
            max_candidates_per_family=3,
            llm_enabled=False,
        )
        external = [
            {
                "family": "state_transition",
                "head": ("state_transition", ["?owner", "?item", "?delta", "?after"]),
                "body": [("before_state", ["?owner", "?item", "?before"]), ("event_delta", ["?owner", "?item", "?delta"])],
                "calc": "before + delta",
                "target": "?after",
                "source": "lt_synergy",
                "ranking": {"score": 0.75, "coverage": 0.8, "simplicity": 0.5, "symbolic_validity": 1.0, "constraint_validity": 0.9},
            }
        ]
        report = run_ilp_learning_stage(
            question="John gave Mary 3 apples and then bought 2 more apples.",
            family="state_transition",
            cfg=cfg,
            external_candidates=external,
            validation_cache={},
            utility_state={"meta": {}, "rules": {}},
        )
        self.assertGreaterEqual(report["counts"]["external_injected"], 1)
        self.assertGreaterEqual(report["counts"]["generated"], 1)
        self.assertTrue(any((c.get("candidate") or {}).get("source") == "lt_synergy" for c in report["accepted_candidates"]))

    def test_large_pool_mode_keeps_low_prescore_candidates_for_q_learning(self):
        cfg = ILPPipelineConfig(
            candidate_top_n=4,
            prescore_min=0.95,
            max_candidates_per_family=1,
            large_pool_mode=True,
            llm_enabled=False,
        )
        candidates = [
            {
                "family": "state_transition",
                "head": ("state", ["?owner", "?item", "?ans"]),
                "body": [("state_transition", ["?owner", "?item", "?before", "?delta", "?after"])],
                "calc": "before + delta",
                "ranking": {"score": 0.2, "coverage": 0.2, "simplicity": 1.0, "symbolic_validity": 1.0, "constraint_validity": 1.0},
            },
            {
                "family": "state_transition",
                "head": ("state", ["?owner", "?item", "?ans2"]),
                "body": [("state_transition", ["?owner", "?item", "?before", "?delta", "?after"])],
                "calc": "before + delta",
                "ranking": {"score": 0.21, "coverage": 0.2, "simplicity": 1.0, "symbolic_validity": 1.0, "constraint_validity": 1.0},
            },
        ]
        accepted, dropped = prescore_and_filter(candidates, cfg=cfg)
        self.assertEqual(len(dropped), 0)
        self.assertEqual(len(accepted), 2)
        self.assertTrue(any((rec.dropped_reason or "").startswith("below_prescore_soft:") or rec.dropped_reason == "per_family_budget" for rec in accepted))

    def test_candidate_top_n_zero_keeps_full_generated_pool(self):
        cfg = ILPPipelineConfig(
            candidate_top_n=0,
            prescore_min=0.3,
            max_candidates_per_family=4,
            llm_enabled=False,
        )
        external = [
            {
                "family": "state_transition",
                "head": ("state_transition", ["?owner", "?item", "?delta", "?after"]),
                "body": [("before_state", ["?owner", "?item", "?before"]), ("event_delta", ["?owner", "?item", "?delta"])],
                "calc": "before + delta",
                "target": "?after",
                "source": f"lt_synergy_{i}",
                "ranking": {"score": 0.6 + i * 0.01, "coverage": 0.7, "simplicity": 0.5, "symbolic_validity": 1.0, "constraint_validity": 0.9},
            }
            for i in range(4)
        ]
        report = run_ilp_learning_stage(
            question="John gave Mary 3 apples and then bought 2 more apples.",
            family="state_transition",
            cfg=cfg,
            external_candidates=external,
            validation_cache={},
            utility_state={"meta": {}, "rules": {}},
        )
        self.assertGreaterEqual(report["counts"]["generated"], 4)
        self.assertGreaterEqual(report["counts"]["external_injected"], 4)


if __name__ == "__main__":
    unittest.main()
