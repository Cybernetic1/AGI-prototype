#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from working_memory.basic_english_grounding import curate_verb_knowledge, propose_verb_frames
from working_memory.ltm import LongTermMemory


def _interactive_reviewer(verb: str, candidate: dict[str, object]) -> bool:
    paradigm = str(candidate.get("paradigm") or "")
    frame = str(candidate.get("frame") or "")
    roles = ",".join(str(r) for r in (candidate.get("roles") or []))
    conf = float(candidate.get("confidence", 0.0) or 0.0)
    prompt = f"{verb}: keep frame {paradigm}/{frame} roles=[{roles}] (confidence={conf:.2f})? [y/N] "
    answer = input(prompt).strip().lower()
    return answer in {"y", "yes"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactively vet and store verb frames into LTM.")
    parser.add_argument("verb", help="Verb to curate (e.g., give).")
    parser.add_argument("--gloss", default="", help="Optional dictionary gloss for better frame proposal.")
    parser.add_argument("--max-frames", type=int, default=5, help="Maximum number of candidate frames.")
    parser.add_argument("--accept-all", action="store_true", help="Accept all candidates without prompting.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing verb frames instead of merging.")
    parser.add_argument("--ltm-path", default="", help="Optional custom LTM YAML path.")
    args = parser.parse_args()

    ltm = LongTermMemory(Path(args.ltm_path)) if args.ltm_path else LongTermMemory.default()
    if args.accept_all:
        callback = lambda _verb, _candidate: True
    else:
        callback = _interactive_reviewer

    print(f"Proposing up to {max(1, int(args.max_frames))} frames for verb={args.verb!r}")
    for cand in propose_verb_frames(verb=args.verb, gloss=args.gloss, max_frames=max(1, int(args.max_frames))):
        print(
            f"  - paradigm={cand.get('paradigm')} frame={cand.get('frame')} roles={cand.get('roles')} "
            f"confidence={float(cand.get('confidence', 0.0)):.2f} source={cand.get('source')}"
        )

    result = curate_verb_knowledge(
        verb=args.verb,
        gloss=args.gloss,
        decision_callback=callback,
        ltm=ltm,
        max_frames=max(1, int(args.max_frames)),
        overwrite=bool(args.overwrite),
    )
    accepted = result.get("accepted") or []
    rejected = result.get("rejected") or []
    print(
        f"stored verb={result.get('verb')} accepted={len(accepted)} rejected={len(rejected)} "
        f"ltm_updated={bool((result.get('store') or {}).get('updated'))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
