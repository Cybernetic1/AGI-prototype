from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ilp_learner.abductive_ilp import KnowledgeBase, OntologicalILPLearner
from ilp_learner.metarule_loop import propose_ranked_for_examples
from ilp_learner.metagol_nds import (
    evaluate_candidate_constraints,
    generate_synthetic_family_examples,
)


def main() -> None:
    examples = [
        ("ratio_compose", ["alice", "bob", "apples", 2.0], 20.0),
        ("ratio_compose", ["carol", "dave", "apples", 3.0], 30.0),
    ]
    ranked = propose_ranked_for_examples(examples, top_n=2)
    assert ranked, "Expected ranked metarule candidates"
    assert "ranking" in ranked[0], "Expected ranking metadata"

    kb = KnowledgeBase()
    learner = OntologicalILPLearner(kb)
    rule = learner.meta_rule_template("ratio_compose", ["alice", "apples", "?ans"], examples=examples)
    assert rule is not None, "Expected metarule template from integrated ranked proposer"
    assert "head" in rule and "body" in rule and "calc" in rule and "target" in rule

    baseline = generate_synthetic_family_examples(include_agi_transfer_families=False)
    expanded = generate_synthetic_family_examples(include_agi_transfer_families=True)
    assert "part_whole_balance" not in baseline
    for family in ("part_whole_balance", "temporal_causal_chain", "unit_compatibility"):
        assert family in expanded, f"Expected AGI transfer family in expanded set: {family}"

    transfer_examples = expanded["part_whole_balance"][:2]
    transfer_ranked = propose_ranked_for_examples(
        transfer_examples,
        top_n=1,
        include_agi_transfer_families=True,
    )
    assert transfer_ranked, "Expected ranked candidates for transfer family"
    assert transfer_ranked[0].get("family") == "part_whole_balance"
    assert transfer_ranked[0].get("ranking", {}).get("constraint_validity", 0.0) >= 1.0

    unit_ranked = propose_ranked_for_examples(
        expanded["unit_compatibility"][:2],
        top_n=1,
        include_agi_transfer_families=True,
    )
    assert unit_ranked, "Expected unit compatibility candidate with transfer families enabled"
    unit_candidate = unit_ranked[0]
    ok_good, _ = evaluate_candidate_constraints(unit_candidate, expanded["unit_compatibility"][:1])
    assert ok_good, "Expected valid unit-compatible examples to satisfy typed constraints"
    bad_examples = [("unit_compatibility", [3.0, "hour", "dollar", 3.0], 3.0)]
    ok_bad, bad_report = evaluate_candidate_constraints(unit_candidate, bad_examples)
    assert not ok_bad, "Expected incompatible unit dimensions to fail typed constraints"
    assert any("compatibility" in item for item in bad_report.get("violations", []))

    print("Metarule integration validation passed.")


if __name__ == "__main__":
    main()
