#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from working_memory.basic_english_grounding import curate_noun_knowledge, propose_noun_facts
from working_memory.ltm import LongTermMemory


def _interactive_reviewer(noun: str, candidate: dict[str, object]) -> bool:
    attr = str(candidate.get("attribute") or "")
    value = str(candidate.get("value") or "")
    conf = float(candidate.get("confidence", 0.0) or 0.0)
    prompt = f"{noun}: keep fact {attr}={value} (confidence={conf:.2f})? [y/N] "
    answer = input(prompt).strip().lower()
    return answer in {"y", "yes"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactively vet and store noun facts into LTM.")
    parser.add_argument("noun", help="Noun to curate (e.g., apple).")
    parser.add_argument("--gloss", default="", help="Optional dictionary gloss for better candidate proposal.")
    parser.add_argument("--max-facts", type=int, default=5, help="Maximum number of candidate facts.")
    parser.add_argument("--accept-all", action="store_true", help="Accept all candidates without prompting.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing noun facts instead of merging.")
    parser.add_argument("--ltm-path", default="", help="Optional custom LTM YAML path.")
    args = parser.parse_args()

    ltm = LongTermMemory(Path(args.ltm_path)) if args.ltm_path else LongTermMemory.default()
    if args.accept_all:
        callback = lambda _noun, _candidate: True
    else:
        callback = _interactive_reviewer

    print(f"Proposing up to {max(1, int(args.max_facts))} facts for noun={args.noun!r}")
    for cand in propose_noun_facts(noun=args.noun, gloss=args.gloss, max_facts=max(1, int(args.max_facts))):
        print(
            f"  - attribute={cand.get('attribute')} value={cand.get('value')} "
            f"confidence={float(cand.get('confidence', 0.0)):.2f}"
        )

    result = curate_noun_knowledge(
        noun=args.noun,
        gloss=args.gloss,
        decision_callback=callback,
        ltm=ltm,
        max_facts=max(1, int(args.max_facts)),
        overwrite=bool(args.overwrite),
    )
    accepted = result.get("accepted") or []
    rejected = result.get("rejected") or []
    print(
        f"stored noun={result.get('noun')} accepted={len(accepted)} rejected={len(rejected)} "
        f"ltm_updated={bool((result.get('store') or {}).get('updated'))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
