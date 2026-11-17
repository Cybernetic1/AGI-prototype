#!/usr/bin/env python3
"""Automate a first-pass alignment of semantic-AR seed predicates to ontology concepts.

The repository is still in the "seed-to-ontology bridge" stage: we can normalize
symbols, map them to canonical lemma anchors, and keep ambiguous cases for human
review instead of forcing a brittle one-to-one assignment.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from working_memory.ontology_seed_alignment import (
    DEFAULT_SEMANTIC_AR_PATH,
    DEFAULT_UNIFIED_ONTOLOGY_PATH,
    align_semantic_ar_file,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Align semantic-AR examples to ontology concepts.")
    parser.add_argument("--semantic-ar", default=str(DEFAULT_SEMANTIC_AR_PATH), help="Path to the semantic-ar example JSON report.")
    parser.add_argument("--ontology", default=str(DEFAULT_UNIFIED_ONTOLOGY_PATH), help="Path to the unified ontology JSON bundle.")
    parser.add_argument("--output", default="", help="Optional output path. Defaults to a sibling aligned report.")
    args = parser.parse_args()

    semantic_ar_path = Path(args.semantic_ar)
    ontology_path = Path(args.ontology)
    if not semantic_ar_path.is_absolute():
        semantic_ar_path = REPO_ROOT / semantic_ar_path
    if not ontology_path.is_absolute():
        ontology_path = REPO_ROOT / ontology_path

    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = REPO_ROOT / out_path
    else:
        out_path = semantic_ar_path.with_name(semantic_ar_path.name.replace("semantic_ar_worked_examples", "semantic_ar_worked_examples_ontology_aligned"))

    aligned = align_semantic_ar_file(semantic_ar_path, ontology_path, out_path)
    summary = aligned.get("ontology_alignment_summary", {})
    print(
        "Aligned semantic-AR seeds: "
        f"examples={summary.get('total_examples', 0)} "
        f"needs_review={summary.get('review_required_examples', 0)} "
        f"output={out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
