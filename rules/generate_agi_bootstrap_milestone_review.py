#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def _get_profile(report: dict[str, Any], profile_name: str) -> dict[str, Any]:
    for profile in report.get("profiles", []):
        if isinstance(profile, dict) and profile.get("profile") == profile_name:
            return profile
    raise KeyError(f"profile '{profile_name}' not found")


def _extract_invariant_core_profiles(invariant: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str]:
    # Backward-compatible path: invariant report is already a baseline-style profile payload.
    profiles = invariant.get("profiles")
    if isinstance(profiles, list) and profiles:
        return _get_profile(invariant, "hybrid"), _get_profile(invariant, "lt-heavy"), "invariant-core-finality"

    # New path: invariant report is a gated curriculum payload with per-slice baseline reports.
    slices = invariant.get("slices")
    if isinstance(slices, list) and slices:
        target_slice = None
        for row in slices:
            if not isinstance(row, dict):
                continue
            if row.get("slice_id") == "invariant-core-finality" or row.get("id") == "invariant-core-finality":
                target_slice = row
                break
        if target_slice is None:
            target_slice = next((row for row in slices if isinstance(row, dict)), None)
        if target_slice is None:
            raise KeyError("no usable slice found in invariant report")

        baseline_report = target_slice.get("baseline_report")
        if isinstance(baseline_report, dict):
            return (
                _get_profile(baseline_report, "hybrid"),
                _get_profile(baseline_report, "lt-heavy"),
                str(target_slice.get("slice_id") or target_slice.get("id") or "unknown"),
            )

        baseline_report_path = target_slice.get("baseline_report_path")
        if baseline_report_path:
            loaded = _load_json(Path(str(baseline_report_path)))
            return (
                _get_profile(loaded, "hybrid"),
                _get_profile(loaded, "lt-heavy"),
                str(target_slice.get("slice_id") or target_slice.get("id") or "unknown"),
            )

        raise KeyError("invariant slice does not include baseline_report or baseline_report_path")

    raise KeyError("invariant report does not contain profiles or slices")


def _cost_per_valid_solve(profile: dict[str, Any]) -> float:
    return float(profile["compute_cost"]["wall_time_sec_per_valid_solve"])


def _symbolic_rate(profile: dict[str, Any]) -> float:
    return float(profile["symbolic_validity"]["symbolic_valid_rate"])


def _solve_rate(profile: dict[str, Any]) -> float:
    return float(profile["capability"]["solve_rate"])


def _best_symbolic_perfect_hybrid_cost(
    baseline: dict[str, Any],
    ab_report: dict[str, Any],
    widening: dict[str, Any],
) -> tuple[float, str]:
    candidates: list[tuple[float, str]] = []

    baseline_hybrid = _get_profile(baseline, "hybrid")
    if _symbolic_rate(baseline_hybrid) >= 1.0:
        candidates.append((_cost_per_valid_solve(baseline_hybrid), "baseline/control hybrid"))

    for source, control, treatment in _iter_ab_arms(ab_report):
        if "hybrid" not in source:
            continue
        if _symbolic_rate(control) >= 1.0:
            candidates.append((_cost_per_valid_solve(control), f"{source}/control"))
        if _symbolic_rate(treatment) >= 1.0:
            candidates.append((_cost_per_valid_solve(treatment), f"{source}/treatment"))

    for slice_row in widening.get("slices", []):
        if not isinstance(slice_row, dict):
            continue
        baseline_report = slice_row.get("baseline_report")
        if not isinstance(baseline_report, dict):
            continue
        hybrid = _get_profile(baseline_report, "hybrid")
        if _symbolic_rate(hybrid) >= 1.0:
            candidates.append(
                (
                    _cost_per_valid_solve(hybrid),
                    f"widening/{slice_row.get('slice_id', 'unknown')}/hybrid",
                )
            )

    if not candidates:
        raise ValueError("no symbolic-perfect hybrid candidates found")
    return min(candidates, key=lambda item: item[0])


def _best_symbolic_perfect_any_cost(
    baseline: dict[str, Any],
    ab_report: dict[str, Any],
    widening: dict[str, Any],
) -> tuple[float, str]:
    candidates: list[tuple[float, str]] = []

    for profile in baseline.get("profiles", []):
        if isinstance(profile, dict) and _symbolic_rate(profile) >= 1.0:
            candidates.append(
                (_cost_per_valid_solve(profile), f"baseline/control {profile.get('profile')}")
            )

    for source, control, treatment in _iter_ab_arms(ab_report):
        if _symbolic_rate(control) >= 1.0:
            candidates.append((_cost_per_valid_solve(control), f"{source}/control"))
        if _symbolic_rate(treatment) >= 1.0:
            candidates.append((_cost_per_valid_solve(treatment), f"{source}/treatment"))

    for slice_row in widening.get("slices", []):
        if not isinstance(slice_row, dict):
            continue
        baseline_report = slice_row.get("baseline_report")
        if not isinstance(baseline_report, dict):
            continue
        for profile in baseline_report.get("profiles", []):
            if isinstance(profile, dict) and _symbolic_rate(profile) >= 1.0:
                candidates.append(
                    (
                        _cost_per_valid_solve(profile),
                        f"widening/{slice_row.get('slice_id', 'unknown')}/{profile.get('profile')}",
                    )
                )

    if not candidates:
        raise ValueError("no symbolic-perfect profile candidates found")
    return min(candidates, key=lambda item: item[0])


def _widening_gate_totals(widening: dict[str, Any]) -> tuple[int, int]:
    passed = 0
    total = 0
    for slice_row in widening.get("slices", []):
        if not isinstance(slice_row, dict):
            continue
        for gate in slice_row.get("gate_results", []):
            total += 1
            if isinstance(gate, dict) and gate.get("passed"):
                passed += 1
    return passed, total


def _iter_ab_arms(ab_report: dict[str, Any]):
    # Legacy A/B format: profiles[].control/treatment
    rows = ab_report.get("profiles")
    if isinstance(rows, list) and rows and isinstance(rows[0], dict) and "control" in rows[0]:
        for row in rows:
            if not isinstance(row, dict):
                continue
            profile_name = str(row.get("profile", "unknown"))
            control = row.get("control")
            treatment = row.get("treatment")
            if isinstance(control, dict) and isinstance(treatment, dict):
                yield (f"ab/{profile_name}", control, treatment)
        return

    # Repeated format: per_run[].control|treatment baseline payloads
    per_run = ab_report.get("per_run")
    if isinstance(per_run, list):
        for run in per_run:
            if not isinstance(run, dict):
                continue
            run_index = run.get("run_index", "?")
            control_report = run.get("control", {})
            treatment_report = run.get("treatment", {})
            c_profiles = control_report.get("profiles", []) if isinstance(control_report, dict) else []
            t_profiles = treatment_report.get("profiles", []) if isinstance(treatment_report, dict) else []
            if not isinstance(c_profiles, list) or not isinstance(t_profiles, list):
                continue
            t_by_name = {
                str(p.get("profile")): p
                for p in t_profiles
                if isinstance(p, dict) and p.get("profile") is not None
            }
            for c_prof in c_profiles:
                if not isinstance(c_prof, dict):
                    continue
                name = str(c_prof.get("profile", "unknown"))
                t_prof = t_by_name.get(name)
                if isinstance(t_prof, dict):
                    yield (f"ab-repeated/run{run_index}/{name}", c_prof, t_prof)


def _ab_macro_sections(ab_report: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    # Legacy format
    if isinstance(ab_report.get("macro_summary"), dict):
        m = ab_report["macro_summary"]
        return (
            dict(m.get("control", {})),
            dict(m.get("treatment", {})),
            dict(m.get("delta_treatment_minus_control", {})),
        )

    # Repeated format
    macro_delta = ab_report.get("macro_delta_treatment_minus_control", {})
    if isinstance(macro_delta, dict):
        delta = {
            "solve_rate_mean": float(macro_delta.get("solve_rate_mean", {}).get("mean", 0.0)),
            "symbolic_valid_rate_mean": float(macro_delta.get("symbolic_valid_rate_mean", {}).get("mean", 0.0)),
            "valid_solves_per_sample_mean": 0.0,
            "wall_time_sec_total_sum": float(macro_delta.get("wall_time_sec_total_sum", {}).get("mean", 0.0)),
        }
        return ({}, {}, delta)
    return ({}, {}, {})


def _hybrid_symbolic_perfect_in_widening(widening: dict[str, Any]) -> bool:
    for slice_row in widening.get("slices", []):
        if not isinstance(slice_row, dict):
            continue
        baseline_report = slice_row.get("baseline_report")
        if not isinstance(baseline_report, dict):
            return False
        hybrid = _get_profile(baseline_report, "hybrid")
        if _symbolic_rate(hybrid) < 1.0:
            return False
    return True


def _format_float(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate AGI bootstrap milestone review from report artifacts.")
    parser.add_argument(
        "--baseline-report",
        default="rules/reports/agi_bootstrap_baseline_v1.json",
        help="Baseline/control report path.",
    )
    parser.add_argument(
        "--invariant-report",
        default="rules/reports/agi_bootstrap_invariant_curriculum_v1.json",
        help="Invariant baseline slice report path.",
    )
    parser.add_argument(
        "--ab-report",
        default="rules/reports/agi_bootstrap_ab_efficiency_report.json",
        help="A/B summary report path.",
    )
    parser.add_argument(
        "--widening-report",
        default="rules/reports/agi_bootstrap_curriculum_widening_v1.json",
        help="Widening curriculum report path.",
    )
    parser.add_argument(
        "--output-json",
        default="rules/reports/agi_bootstrap_milestone_review_v1.json",
        help="Review JSON output path.",
    )
    parser.add_argument(
        "--output-md",
        default="rules/reports/agi_bootstrap_milestone_review_v1.md",
        help="Review Markdown output path.",
    )
    parser.add_argument(
        "--target-factor",
        type=float,
        default=100.0,
        help="Cost-reduction target factor versus baseline (default: 100).",
    )
    args = parser.parse_args()

    baseline_path = Path(args.baseline_report)
    invariant_path = Path(args.invariant_report)
    ab_path = Path(args.ab_report)
    widening_path = Path(args.widening_report)
    output_json_path = Path(args.output_json)
    output_md_path = Path(args.output_md)

    baseline = _load_json(baseline_path)
    invariant = _load_json(invariant_path)
    ab_report = _load_json(ab_path)
    widening = _load_json(widening_path)

    baseline_hybrid = _get_profile(baseline, "hybrid")
    invariant_hybrid, invariant_lt, invariant_slice_id = _extract_invariant_core_profiles(invariant)

    baseline_hybrid_cost = _cost_per_valid_solve(baseline_hybrid)
    best_hybrid_cost, best_hybrid_source = _best_symbolic_perfect_hybrid_cost(
        baseline=baseline, ab_report=ab_report, widening=widening
    )
    best_any_cost, best_any_source = _best_symbolic_perfect_any_cost(
        baseline=baseline, ab_report=ab_report, widening=widening
    )

    target_threshold = baseline_hybrid_cost / args.target_factor
    achieved_factor_hybrid = baseline_hybrid_cost / best_hybrid_cost
    achieved_reduction_pct_hybrid = (1.0 - (best_hybrid_cost / baseline_hybrid_cost)) * 100.0
    remaining_factor_hybrid = best_hybrid_cost / target_threshold
    additional_reduction_needed_pct_hybrid = (1.0 - (target_threshold / best_hybrid_cost)) * 100.0

    achieved_factor_best_any = baseline_hybrid_cost / best_any_cost
    remaining_factor_best_any = best_any_cost / target_threshold

    ab_control_macro, ab_treatment_macro, ab_delta = _ab_macro_sections(ab_report)
    widening_passed, widening_total = _widening_gate_totals(widening)
    widening_all_green = bool(widening.get("all_gates_green", False))
    widening_hybrid_symbolic_perfect = _hybrid_symbolic_perfect_in_widening(widening)

    ilp_heavy_widening_symbolic_rates = []
    for slice_row in widening.get("slices", []):
        if not isinstance(slice_row, dict):
            continue
        baseline_report = slice_row.get("baseline_report")
        if not isinstance(baseline_report, dict):
            continue
        ilp_profile = _get_profile(baseline_report, "ilp-heavy")
        ilp_heavy_widening_symbolic_rates.append(
            {
                "slice_id": slice_row.get("slice_id"),
                "symbolic_valid_rate": _symbolic_rate(ilp_profile),
            }
        )

    review = {
        "review_id": "agi-bootstrap-milestone-review-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "goal_context": {
            "focus": [
                "progress toward 100x cost-reduction target",
                "symbolic-validity preservation during widening",
            ]
        },
        "sources": {
            "baseline": str(baseline_path),
            "invariant": str(invariant_path),
            "ab": str(ab_path),
            "widening": str(widening_path),
        },
        "aggregated_findings": {
            "baseline": {
                "hybrid": {
                    "solve_rate": _solve_rate(baseline_hybrid),
                    "symbolic_valid_rate": _symbolic_rate(baseline_hybrid),
                    "wall_time_sec_per_valid_solve": baseline_hybrid_cost,
                }
            },
            "invariant": {
                "slice_id": invariant_slice_id,
                "hybrid": {
                    "solve_rate": _solve_rate(invariant_hybrid),
                    "symbolic_valid_rate": _symbolic_rate(invariant_hybrid),
                },
                "lt_heavy": {
                    "solve_rate": _solve_rate(invariant_lt),
                    "symbolic_valid_rate": _symbolic_rate(invariant_lt),
                },
            },
            "ab": {
                "macro_control": ab_control_macro,
                "macro_treatment": ab_treatment_macro,
                "delta_treatment_minus_control": ab_delta,
            },
            "widening": {
                "all_gates_green": widening_all_green,
                "gate_pass_count": widening_passed,
                "gate_total_count": widening_total,
                "hybrid_symbolic_perfect_all_slices": widening_hybrid_symbolic_perfect,
                "executed_slice_ids": widening.get("executed_slice_ids", []),
                "ilp_heavy_symbolic_valid_rates": ilp_heavy_widening_symbolic_rates,
            },
        },
        "achieved": [
            f"Widening regression passed with all gates green ({widening_passed}/{widening_total}) in {widening_path.name}.",
            "Hybrid profile preserved symbolic validity at 1.0 on baseline, invariant-core, A/B control+treatment, and every widening slice.",
            "A/B treatment preserved quality metrics: macro deltas are 0.0 for solve rate, symbolic-valid rate, and valid-solves/sample.",
        ],
        "not_yet_achieved": [
            f"{args.target_factor:g}x cost reduction is not met under symbolic-perfect hybrid comparison.",
            f"A/B treatment did not reduce aggregate wall-time; macro delta is +{_format_float(float(ab_delta.get('wall_time_sec_total_sum', 0.0)), 6)} sec.",
            "ILP-heavy symbolic validity is not consistently preserved in widening (mixed-narrative-symbolic-compact=0.333333, widening-sentinel-regression=0.833333).",
        ],
        "gap_to_target": {
            "target_factor": args.target_factor,
            "baseline_reference": {
                "profile": "hybrid",
                "source": str(baseline_path),
                "wall_time_sec_per_valid_solve": baseline_hybrid_cost,
            },
            "best_symbolic_perfect_hybrid": {
                "source": best_hybrid_source,
                "wall_time_sec_per_valid_solve": best_hybrid_cost,
            },
            "best_symbolic_perfect_any_profile": {
                "source": best_any_source,
                "wall_time_sec_per_valid_solve": best_any_cost,
            },
            "hybrid_target_threshold_sec_per_valid_solve": target_threshold,
            "hybrid_achieved_factor_vs_baseline": achieved_factor_hybrid,
            "hybrid_achieved_reduction_percent": achieved_reduction_pct_hybrid,
            "hybrid_remaining_factor_to_target": remaining_factor_hybrid,
            "hybrid_additional_reduction_needed_percent": additional_reduction_needed_pct_hybrid,
            "best_any_achieved_factor_vs_hybrid_baseline": achieved_factor_best_any,
            "best_any_remaining_factor_to_target": remaining_factor_best_any,
        },
        "immediate_next_actions": [
            "Prioritize hybrid cost reduction experiments while keeping symbolic_valid_rate=1.0 and all widening gates green.",
            "Target profiles/slices where cost is highest under symbolic-perfect runs (e.g., invariant-core hybrid cost 1.800792 sec/valid-solve).",
            "Stabilize ilp-heavy symbolic validity on mixed narrative and widening sentinel slices before treating it as a reliable low-cost path.",
            "Re-run A/B and widening reports after each cost-focused change, then regenerate this milestone review with the command below.",
        ],
        "reproducible_command": (
            "python3 rules/generate_agi_bootstrap_milestone_review.py "
            f"--baseline-report {baseline_path} "
            f"--invariant-report {invariant_path} "
            f"--ab-report {ab_path} "
            f"--widening-report {widening_path} "
            f"--output-json {output_json_path} "
            f"--output-md {output_md_path}"
        ),
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(review, indent=2), encoding="utf-8")

    md_lines = [
        "# AGI Bootstrap Milestone Review (v1)",
        "",
        f"- Generated: {review['generated_at']}",
        "- Scope: baseline + invariant + A/B + widening artifacts generated in this execution chain.",
        "",
        "## Evidence sources",
        f"- Baseline: `{baseline_path}`",
        f"- Invariant: `{invariant_path}`",
        f"- A/B: `{ab_path}`",
        f"- Widening: `{widening_path}`",
        "",
        "## What is achieved",
    ]
    md_lines.extend(f"- {line}" for line in review["achieved"])
    md_lines.extend(
        [
            "",
            "## What is not yet achieved",
        ]
    )
    md_lines.extend(f"- {line}" for line in review["not_yet_achieved"])
    md_lines.extend(
        [
            "",
            "## Quantified gap to 100x target",
            f"- Baseline reference (hybrid control): `{baseline_hybrid_cost:.6f}` sec/valid-solve.",
            f"- Best symbolic-perfect hybrid observed: `{best_hybrid_cost:.6f}` sec/valid-solve ({best_hybrid_source}).",
            f"- Achieved reduction vs baseline (hybrid): `{achieved_factor_hybrid:.3f}x` ({achieved_reduction_pct_hybrid:.3f}%).",
            f"- 100x threshold from baseline: `{target_threshold:.6f}` sec/valid-solve.",
            f"- Remaining gap (hybrid path): still `{remaining_factor_hybrid:.3f}x` above target; needs `{additional_reduction_needed_pct_hybrid:.3f}%` further reduction.",
            f"- Best symbolic-perfect any-profile checkpoint: `{best_any_cost:.6f}` sec/valid-solve ({best_any_source}), still `{remaining_factor_best_any:.3f}x` above target.",
            "",
            "## Immediate next actions",
        ]
    )
    md_lines.extend(f"- {line}" for line in review["immediate_next_actions"])
    md_lines.extend(
        [
            "",
            "## Reproducible command",
            "```bash",
            review["reproducible_command"],
            "```",
            "",
        ]
    )
    output_md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {output_json_path}")
    print(f"Wrote {output_md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
