#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_RUNNER = REPO_ROOT / "rules" / "run_bootstrap_baseline.py"

DEFAULT_FAMILIES = [
    "buy_more_final_gap",
    "donation_after_buy_more",
    "buy_more_then_donation_and_spoilage",
    "buy_more_then_donation_after_spoilage_long_tail",
    "unit_conversion_dozen",
    "distance_over_speed",
    "percent_sequence_growth",
    "percent_sequence_discount",
]
DEFAULT_SEEDS = [11, 23, 37, 53, 71]
DEFAULT_PROFILES = ["lt-heavy", "hybrid", "ilp-heavy"]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def _metric(profile: dict[str, Any], key: str) -> float:
    if key == "solve_rate":
        return float(profile["capability"]["solve_rate"])
    if key == "symbolic_valid_rate":
        return float(profile["symbolic_validity"]["symbolic_valid_rate"])
    if key == "valid_solves_per_sample":
        return float(profile["sample_efficiency"]["valid_solves_per_sample"])
    if key == "wall_time_sec_total":
        return float(profile["compute_cost"]["wall_time_sec_total"])
    if key == "wall_time_sec_per_valid_solve":
        value = profile["compute_cost"]["wall_time_sec_per_valid_solve"]
        return float(value if value is not None else 0.0)
    raise KeyError(key)


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "stdev": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(statistics.mean(values)),
        "stdev": float(statistics.pstdev(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _run_one(
    *,
    python_exec: str,
    output_path: Path,
    suite: str,
    families: list[str],
    max_cycles: int,
    seed: int,
    treatment: bool,
    teacher_proposals: str,
    sar_config_file: str,
    profiles: list[str],
    ilp_max_depth: int,
    ilp_max_fact_matches: int,
    ilp_max_rule_matches: int,
    ilp_max_subgoal_results: int,
    numeric_authority_strict: bool,
    pot_enable: bool,
) -> dict[str, Any]:
    cmd = [
        python_exec,
        str(BASELINE_RUNNER),
        "--suite",
        suite,
        "--profiles",
        *profiles,
        "--max-cycles",
        str(max_cycles),
        "--families",
        *families,
        "--output",
        str(output_path),
        "--quiet",
        "--seed",
        str(seed),
    ]
    if str(sar_config_file or "").strip():
        cmd.extend(["--sar-config-file", str(sar_config_file)])
    if ilp_max_depth > 0:
        cmd.extend(["--ilp-max-depth", str(ilp_max_depth)])
    if ilp_max_fact_matches > 0:
        cmd.extend(["--ilp-max-fact-matches", str(ilp_max_fact_matches)])
    if ilp_max_rule_matches > 0:
        cmd.extend(["--ilp-max-rule-matches", str(ilp_max_rule_matches)])
    if ilp_max_subgoal_results > 0:
        cmd.extend(["--ilp-max-subgoal-results", str(ilp_max_subgoal_results)])
    if numeric_authority_strict:
        cmd.append("--numeric-authority-strict")
    if pot_enable:
        cmd.append("--pot-enable")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{REPO_ROOT / 'logic_transformer'}:{env.get('PYTHONPATH','')}".rstrip(":")
    if treatment:
        env["AGI_ENABLE_TRANSFER_METARULES"] = "1"
        cmd.extend(
            [
                "--sar-enabled",
                "--sar-weight",
                "0.25",
                "--sar-lexical-weight",
                "0.55",
                "--sar-role-weight",
                "0.3",
                "--sar-prior-weight",
                "0.15",
                "--sar-gate-fail-penalty",
                "0.05",
                "--teacher-enable-ingestion",
                "--teacher-ab-arm",
                "treatment",
                "--teacher-low-volume",
            ]
        )
        if teacher_proposals:
            cmd.extend(["--teacher-proposals", teacher_proposals])
    else:
        env["AGI_ENABLE_TRANSFER_METARULES"] = "0"
        cmd.extend(["--teacher-ab-arm", "control"])

    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return _read_json(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run repeated fixed-seed A/B bootstrap studies and aggregate variance.")
    parser.add_argument("--suite", default="combined")
    parser.add_argument("--families", nargs="+", default=DEFAULT_FAMILIES)
    parser.add_argument("--max-cycles", type=int, default=2)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--profiles", nargs="+", default=DEFAULT_PROFILES, choices=DEFAULT_PROFILES)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--sar-config-file", default="")
    parser.add_argument("--teacher-proposals", default="rules/reports/teacher_proposals_ab_smoke.json")
    parser.add_argument("--ilp-max-depth", type=int, default=0)
    parser.add_argument("--ilp-max-fact-matches", type=int, default=0)
    parser.add_argument("--ilp-max-rule-matches", type=int, default=0)
    parser.add_argument("--ilp-max-subgoal-results", type=int, default=0)
    parser.add_argument("--numeric-authority-strict", action="store_true")
    parser.add_argument("--pot-enable-treatment", action="store_true", help="Enable PoT DSL treatment arm while keeping control unchanged.")
    parser.add_argument("--outdir", default="rules/reports")
    parser.add_argument("--output", default="rules/reports/agi_bootstrap_ab_efficiency_repeated_v1.json")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    per_run = []

    for idx, seed in enumerate(args.seeds, start=1):
        control_path = outdir / f"agi_bootstrap_ab_control_seed{seed}.json"
        treatment_path = outdir / f"agi_bootstrap_ab_treatment_seed{seed}.json"
        control = _run_one(
            python_exec=args.python,
            output_path=control_path,
            suite=args.suite,
            families=list(args.families),
            max_cycles=int(args.max_cycles),
            seed=int(seed),
            treatment=False,
            teacher_proposals=str(args.teacher_proposals),
            sar_config_file=str(args.sar_config_file),
            profiles=list(args.profiles),
            ilp_max_depth=int(args.ilp_max_depth),
            ilp_max_fact_matches=int(args.ilp_max_fact_matches),
            ilp_max_rule_matches=int(args.ilp_max_rule_matches),
            ilp_max_subgoal_results=int(args.ilp_max_subgoal_results),
            numeric_authority_strict=bool(args.numeric_authority_strict),
            pot_enable=False,
        )
        treatment = _run_one(
            python_exec=args.python,
            output_path=treatment_path,
            suite=args.suite,
            families=list(args.families),
            max_cycles=int(args.max_cycles),
            seed=int(seed),
            treatment=True,
            teacher_proposals=str(args.teacher_proposals),
            sar_config_file=str(args.sar_config_file),
            profiles=list(args.profiles),
            ilp_max_depth=int(args.ilp_max_depth),
            ilp_max_fact_matches=int(args.ilp_max_fact_matches),
            ilp_max_rule_matches=int(args.ilp_max_rule_matches),
            ilp_max_subgoal_results=int(args.ilp_max_subgoal_results),
            numeric_authority_strict=bool(args.numeric_authority_strict),
            pot_enable=bool(args.pot_enable_treatment),
        )
        per_run.append(
            {
                "run_index": idx,
                "seed": seed,
                "control_report": str(control_path),
                "treatment_report": str(treatment_path),
                "control": control,
                "treatment": treatment,
            }
        )

    profile_summary: list[dict[str, Any]] = []
    for profile_name in args.profiles:
        c_profiles = [next(p for p in run["control"]["profiles"] if p["profile"] == profile_name) for run in per_run]
        t_profiles = [next(p for p in run["treatment"]["profiles"] if p["profile"] == profile_name) for run in per_run]
        metric_rows = {}
        for metric in [
            "solve_rate",
            "symbolic_valid_rate",
            "valid_solves_per_sample",
            "wall_time_sec_total",
            "wall_time_sec_per_valid_solve",
        ]:
            c_vals = [_metric(p, metric) for p in c_profiles]
            t_vals = [_metric(p, metric) for p in t_profiles]
            delta = [tv - cv for tv, cv in zip(t_vals, c_vals)]
            metric_rows[metric] = {
                "control": _stats(c_vals),
                "treatment": _stats(t_vals),
                "delta_treatment_minus_control": _stats(delta),
            }
        profile_summary.append({"profile": profile_name, "metrics": metric_rows})

    macro_delta_wall = []
    macro_delta_solve = []
    macro_delta_symbolic = []
    for run in per_run:
        c = run["control"]["profiles"]
        t = run["treatment"]["profiles"]
        c_wall = sum(float(p["compute_cost"]["wall_time_sec_total"]) for p in c)
        t_wall = sum(float(p["compute_cost"]["wall_time_sec_total"]) for p in t)
        c_solve = statistics.mean(float(p["capability"]["solve_rate"]) for p in c)
        t_solve = statistics.mean(float(p["capability"]["solve_rate"]) for p in t)
        c_sym = statistics.mean(float(p["symbolic_validity"]["symbolic_valid_rate"]) for p in c)
        t_sym = statistics.mean(float(p["symbolic_validity"]["symbolic_valid_rate"]) for p in t)
        macro_delta_wall.append(t_wall - c_wall)
        macro_delta_solve.append(t_solve - c_solve)
        macro_delta_symbolic.append(t_sym - c_sym)

    payload = {
        "study_id": "agi-bootstrap-ab-efficiency-repeated-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "suite": args.suite,
        "families": list(args.families),
        "max_cycles": int(args.max_cycles),
        "seeds": list(args.seeds),
        "profiles_requested": list(args.profiles),
        "search_budget": {
            "strict_numeric_authority": bool(args.numeric_authority_strict),
            "ilp_max_depth": int(args.ilp_max_depth),
            "ilp_max_fact_matches": int(args.ilp_max_fact_matches),
            "ilp_max_rule_matches": int(args.ilp_max_rule_matches),
            "ilp_max_subgoal_results": int(args.ilp_max_subgoal_results),
        },
        "pot_treatment": {"enabled": bool(args.pot_enable_treatment)},
        "run_count": len(per_run),
        "per_run": per_run,
        "profiles": profile_summary,
        "macro_delta_treatment_minus_control": {
            "wall_time_sec_total_sum": _stats(macro_delta_wall),
            "solve_rate_mean": _stats(macro_delta_solve),
            "symbolic_valid_rate_mean": _stats(macro_delta_symbolic),
        },
        "reproducible_command": (
            "python rules/run_bootstrap_ab_repeated.py "
            f"--suite {args.suite} --max-cycles {args.max_cycles} "
            + (f"--sar-config-file {args.sar_config_file} " if str(args.sar_config_file or "").strip() else "")
            + "--profiles "
            + " ".join(str(p) for p in args.profiles)
            + " "
            + ("--pot-enable-treatment " if bool(args.pot_enable_treatment) else "")
            + "--seeds "
            + " ".join(str(s) for s in args.seeds)
        ),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved repeated A/B report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
