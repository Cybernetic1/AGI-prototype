#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def _safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _safe_median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _resolve_inputs(paths: list[str], pattern: str) -> list[Path]:
    resolved: list[Path] = []
    for item in paths:
        p = Path(item)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if p.is_dir():
            resolved.extend(sorted(p.glob(pattern)))
        elif p.exists():
            resolved.append(p)
    # de-dup while preserving order
    seen: set[str] = set()
    ordered: list[Path] = []
    for path in resolved:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        ordered.append(path)
    return ordered


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize one or more GPU trial matrix reports.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["rules/reports/trials/gpu_trial_v1"],
        help="JSON report files and/or directories containing trial matrix reports.",
    )
    parser.add_argument("--glob", default="gpu_trial_matrix_*.json", help="Glob used when an input path is a directory.")
    parser.add_argument("--output", default="rules/reports/trials/gpu_trial_v1/summary_latest.json")
    args = parser.parse_args()

    report_paths = _resolve_inputs(list(args.inputs), str(args.glob))
    if not report_paths:
        raise ValueError("no input trial reports found")

    all_runs: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for path in report_paths:
        payload = _load_json(path)
        records = payload.get("records") if isinstance(payload.get("records"), list) else []
        all_runs.extend([row for row in records if isinstance(row, dict)])
        source_rows.append(
            {
                "path": str(path),
                "trial_id": str(payload.get("trial_id", "")),
                "mode": str(payload.get("mode", "")),
                "run_count": int(payload.get("run_count", 0) or 0),
            }
        )

    by_arm: dict[str, dict[str, Any]] = {}
    for row in all_runs:
        arm = str(row.get("arm") or "unknown")
        bucket = by_arm.setdefault(
            arm,
            {
                "cost_usd": [],
                "wall_sec": [],
                "validator_solve": [],
                "validator_answer_gen": [],
                "baseline_solve": [],
                "baseline_symbolic": [],
                "run_ids": [],
            },
        )
        bucket["cost_usd"].append(float(row.get("estimated_cost_usd", 0.0) or 0.0))
        bucket["wall_sec"].append(float(row.get("wall_time_sec_total", 0.0) or 0.0))
        v = row.get("validator_metrics") or {}
        b = row.get("baseline_metrics") or {}
        bucket["validator_solve"].append(float(v.get("solve_rate", 0.0) or 0.0))
        bucket["validator_answer_gen"].append(float(v.get("answer_generation_rate", 0.0) or 0.0))
        bucket["baseline_solve"].append(float(b.get("solve_rate_mean", 0.0) or 0.0))
        bucket["baseline_symbolic"].append(float(b.get("symbolic_valid_rate_mean", 0.0) or 0.0))
        bucket["run_ids"].append(str(row.get("run_id") or "run"))

    arm_summary: list[dict[str, Any]] = []
    for arm, stats in sorted(by_arm.items()):
        cost_total = float(sum(stats["cost_usd"]))
        baseline_symbolic_mean = _safe_mean(stats["baseline_symbolic"])
        validator_solve_mean = _safe_mean(stats["validator_solve"])
        arm_summary.append(
            {
                "arm": arm,
                "run_count": len(stats["run_ids"]),
                "run_ids": stats["run_ids"],
                "cost_usd_total": cost_total,
                "cost_usd_mean": _safe_mean(stats["cost_usd"]),
                "cost_usd_median": _safe_median(stats["cost_usd"]),
                "wall_time_sec_total": float(sum(stats["wall_sec"])),
                "baseline_solve_rate_mean": _safe_mean(stats["baseline_solve"]),
                "baseline_symbolic_valid_rate_mean": baseline_symbolic_mean,
                "validator_solve_rate_mean": validator_solve_mean,
                "validator_answer_generation_rate_mean": _safe_mean(stats["validator_answer_gen"]),
                "cost_per_baseline_symbolic_valid_point": (cost_total / baseline_symbolic_mean) if baseline_symbolic_mean > 0 else None,
                "cost_per_validator_solve_point": (cost_total / validator_solve_mean) if validator_solve_mean > 0 else None,
            }
        )

    payload = {
        "summary_id": "gpu-trial-cost-summary-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_count": len(report_paths),
        "run_count": len(all_runs),
        "sources": source_rows,
        "arm_summary": arm_summary,
    }

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved trial cost summary to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
