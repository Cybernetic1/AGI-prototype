#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any


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


def _predict(
    row: dict[str, Any],
    *,
    weight: float,
    lexical_weight: float,
    role_weight: float,
    prior_weight: float,
    gate_fail_penalty: float,
    threshold: float,
) -> int:
    features = dict(row.get("features") or {})
    lx = float(features.get("lexical_alignment", 0.0))
    rc = float(features.get("role_coverage", 0.0))
    cp = float(features.get("concept_prior", 0.0))
    gate_pass = features.get("symbolic_gate_pass") is True

    base = float(row.get("base_score", 0.6))
    denom = max(1e-9, lexical_weight + role_weight + prior_weight)
    sar_raw = (lexical_weight * lx + role_weight * rc + prior_weight * cp) / denom
    if not gate_pass:
        sar_raw = 0.0
    score = min(1.0, max(0.0, base + weight * sar_raw - (gate_fail_penalty if not gate_pass else 0.0)))
    return 1 if score >= threshold else 0


def _metrics(rows: list[dict[str, Any]], **cfg: Any) -> dict[str, float]:
    tp = tn = fp = fn = 0
    for row in rows:
        gold = int(row.get("label", 0))
        pred = _predict(row, **cfg)
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
    f1 = 0.0 if (precision + recall) == 0.0 else (2.0 * precision * recall / (precision + recall))
    return {
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train SAR config by grid-searching symbolic-ranking weights.")
    parser.add_argument("--data", default="rules/reports/sar_training_data_v1.jsonl")
    parser.add_argument("--output", default="rules/reports/sar_trained_config_v1.json")
    parser.add_argument("--threshold", type=float, default=0.65)
    args = parser.parse_args()

    rows = _read_jsonl(Path(args.data))
    if not rows:
        raise ValueError(f"no SAR training rows in {args.data}")

    weight_grid = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    mix_grid = [0.2, 0.3, 0.4, 0.5, 0.6]
    penalty_grid = [0.0, 0.02, 0.05, 0.08, 0.1]

    best = None
    best_metrics = None
    for weight, lex_w, role_w, prior_w, penalty in product(weight_grid, mix_grid, mix_grid, mix_grid, penalty_grid):
        score = _metrics(
            rows,
            weight=weight,
            lexical_weight=lex_w,
            role_weight=role_w,
            prior_weight=prior_w,
            gate_fail_penalty=penalty,
            threshold=float(args.threshold),
        )
        rank_key = (score["f1"], score["accuracy"], score["precision"], score["recall"])
        if best_metrics is None or rank_key > (
            best_metrics["f1"],
            best_metrics["accuracy"],
            best_metrics["precision"],
            best_metrics["recall"],
        ):
            best = {
                "enabled": True,
                "weight": weight,
                "lexical_weight": lex_w,
                "role_weight": role_w,
                "prior_weight": prior_w,
                "gate_fail_penalty": penalty,
                "couple_to_symbolic_gate": True,
                "emit_metadata": False,
            }
            best_metrics = score

    payload = {
        "config": best,
        "metrics": best_metrics,
        "train_rows": len(rows),
        "threshold": float(args.threshold),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved SAR trained config to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
