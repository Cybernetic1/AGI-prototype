#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from working_memory.ontology_unifier import (
    DEFAULT_UNIFIED_ONTOLOGY_PATH,
    build_unified_ontology,
    export_ltm_triples,
    export_embedding_edges,
    save_unified_ontology,
)
from working_memory.ltm import LongTermMemory


def main() -> int:
    parser = argparse.ArgumentParser(description="Build unified WordNet/VerbNet/FrameNet ontology bundle.")
    parser.add_argument("--output", default=str(DEFAULT_UNIFIED_ONTOLOGY_PATH))
    parser.add_argument("--skip-wordnet", action="store_true")
    parser.add_argument("--skip-verbnet", action="store_true")
    parser.add_argument("--skip-framenet", action="store_true")
    parser.add_argument("--framenet-full", action="store_true", help="Import all FrameNet lexical units, not only verbs.")
    parser.add_argument("--embedding-edges-out", default="", help="Optional path to export embedding edge tuples as JSON.")
    parser.add_argument("--sync-ltm", action="store_true", help="Also sync exported ontology triples into LTM store.")
    parser.add_argument("--ltm-path", default="", help="Optional LTM store path override for --sync-ltm.")
    args = parser.parse_args()

    payload = build_unified_ontology(
        include_wordnet=not bool(args.skip_wordnet),
        include_verbnet=not bool(args.skip_verbnet),
        include_framenet=not bool(args.skip_framenet),
        framenet_verb_focused=not bool(args.framenet_full),
    )
    out = Path(str(args.output))
    if not out.is_absolute():
        out = Path(__file__).resolve().parents[1] / out
    save_unified_ontology(payload, out)

    if str(args.embedding_edges_out or "").strip():
        emb_path = Path(str(args.embedding_edges_out))
        if not emb_path.is_absolute():
            emb_path = Path(__file__).resolve().parents[1] / emb_path
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        emb_path.write_text(json.dumps(export_embedding_edges(payload), indent=2), encoding="utf-8")
        print(f"Saved embedding edge export to {emb_path}")

    if args.sync_ltm:
        ltm_path = Path(str(args.ltm_path)) if str(args.ltm_path or "").strip() else LongTermMemory.default().path
        if not ltm_path.is_absolute():
            ltm_path = Path(__file__).resolve().parents[1] / ltm_path
        ltm = LongTermMemory(ltm_path)
        triples = export_ltm_triples(payload)
        report = ltm.upsert_ontology_triples(triples=triples, source="unified_ontology", overwrite=True)
        print(f"Synced ontology triples to LTM: path={ltm_path} count={report.get('count', 0)}")

    print(
        "Built unified ontology: "
        f"records={payload.get('record_count', 0)} "
        f"lemmas={payload.get('lemma_count', 0)} "
        f"sources={payload.get('source_counts', {})}"
    )
    print(f"Saved unified ontology to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
