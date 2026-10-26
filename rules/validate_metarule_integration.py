from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ilp_learner.abductive_ilp import KnowledgeBase, OntologicalILPLearner
from ilp_learner.metarule_loop import propose_ranked_for_examples


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
    print("Metarule integration validation passed.")


if __name__ == "__main__":
    main()
