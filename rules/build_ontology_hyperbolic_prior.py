#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from working_memory.ontology_hyperbolic_prior import (
    DEFAULT_ONTOLOGY_PRIOR_PATH,
    DEFAULT_UNIFIED_ONTOLOGY_PATH,
    build_hyperbolic_prior_from_unified,
    save_ontology_prior,
)
from working_memory.ontology_unifier import load_unified_ontology


def main() -> int:
    parser = argparse.ArgumentParser(description="Build LT-only ontology hyperbolic prior from unified ontology bundle.")
    parser.add_argument("--unified", default=str(DEFAULT_UNIFIED_ONTOLOGY_PATH))
    parser.add_argument("--output", default=str(DEFAULT_ONTOLOGY_PRIOR_PATH))
    parser.add_argument("--dim", type=int, default=16)
    args = parser.parse_args()

    unified_path = Path(str(args.unified))
    if not unified_path.is_absolute():
        unified_path = REPO_ROOT / unified_path
    payload = load_unified_ontology(unified_path)
    prior = build_hyperbolic_prior_from_unified(payload, dim=max(4, int(args.dim)))

    output_path = Path(str(args.output))
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    save_ontology_prior(prior, output_path)

    print(
        "Built ontology hyperbolic prior: "
        f"lemmas={prior.get('lemma_count', 0)} dim={prior.get('dim', 0)}"
    )
    print(f"Saved prior to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
