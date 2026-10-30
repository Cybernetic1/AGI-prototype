#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from working_memory.pot_dsl import program_from_dict, program_feature_keys, score_program


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _train(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pos = [r for r in rows if int(r.get("label", 0)) == 1]
    neg = [r for r in rows if int(r.get("label", 0)) == 0]
    pos_count = Counter()
    neg_count = Counter()
    vocab = set()
    for row in pos:
        feats = program_feature_keys(program_from_dict(row.get("program") or {}))
        vocab.update(feats)
        pos_count.update(feats)
    for row in neg:
        feats = program_feature_keys(program_from_dict(row.get("program") or {}))
        vocab.update(feats)
        neg_count.update(feats)

    pos_total = max(1, sum(pos_count.values()))
    neg_total = max(1, sum(neg_count.values()))
    vsize = max(1, len(vocab))
    weights: dict[str, float] = {}
    for feat in sorted(vocab):
        p_pos = (pos_count.get(feat, 0) + 1.0) / (pos_total + vsize)
        p_neg = (neg_count.get(feat, 0) + 1.0) / (neg_total + vsize)
        weights[feat] = math.log(p_pos / p_neg)

    prior_pos = (len(pos) + 1.0) / (len(rows) + 2.0)
    prior_neg = (len(neg) + 1.0) / (len(rows) + 2.0)
    bias = math.log(prior_pos / prior_neg)
    return {"bias": bias, "weights": weights}


def _evaluate(rows: list[dict[str, Any]], model: dict[str, Any]) -> dict[str, float]:
    tp = tn = fp = fn = 0
    for row in rows:
        score = score_program(program_from_dict(row.get("program") or {}), model)
        pred = 1 if score >= 0.0 else 0
        gold = int(row.get("label", 0))
        if pred == 1 and gold == 1:
            tp += 1
        elif pred == 0 and gold == 0:
            tn += 1
        elif pred == 1 and gold == 0:
            fp += 1
        else:
            fn += 1
    total = max(1, tp + tn + fp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return {
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a simple PoT proposer scorer from corpus traces.")
    parser.add_argument("--data", default="rules/reports/pot_corpus_v1.jsonl")
    parser.add_argument("--output", default="rules/reports/pot_proposer_model_v1.json")
    args = parser.parse_args()

    rows = _read_jsonl(Path(args.data))
    if not rows:
        raise ValueError(f"no rows in {args.data}")
    model = _train(rows)
    metrics = _evaluate(rows, model)
    payload = {
        "model": model,
        "metrics": metrics,
        "rows": len(rows),
        "notes": "Naive Bayes style feature log-odds over PoT op/bigram/length features.",
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved PoT proposer model to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
