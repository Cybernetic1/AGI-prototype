#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "logic_transformer") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "logic_transformer"))

from semantic_ar import SemanticARConfig, load_semantic_ar_config

DEFAULT_CONTRACT_PATH = Path("rules/bootstrap_invariant_curriculum.json")
DEFAULT_OUTPUT_PATH = Path("rules/reports/agi_bootstrap_invariant_curriculum_v1.json")
BASELINE_CONTRACT_ID = "agi-bootstrap-baseline-contract-v1"

_OPS = {"==", ">=", ">", "<=", "<"}
_SUITE_CONSTANTS = {
    "template_generalization": "GENERALIZED_TEMPLATE_FAMILIES",
    "finality_regression": "FINALITY_REGRESSION_FAMILIES",
    "tinystories_state_tracking": "TINYSTORIES_STATE_TRACKING_FAMILIES",
    "mixed_narrative_symbolic": "MIXED_NARRATIVE_SYMBOLIC_FAMILIES",
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"contract must be a JSON object: {path}")
    return payload


def _extract_family_constants(cognitive_loop_path: Path) -> dict[str, list[dict[str, Any]]]:
    tree = ast.parse(cognitive_loop_path.read_text(encoding="utf-8"), filename=str(cognitive_loop_path))
    constants: dict[str, list[dict[str, Any]]] = {}
    wanted = set(_SUITE_CONSTANTS.values())

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name) or target.id not in wanted:
                continue
            value = ast.literal_eval(node.value)
            if not isinstance(value, list):
                raise ValueError(f"{target.id} must be a list in {cognitive_loop_path}")
            constants[target.id] = value

    missing = sorted(wanted.difference(constants.keys()))
    if missing:
        raise ValueError(
            f"missing expected family constants in {cognitive_loop_path}: {', '.join(missing)}"
        )
    return constants


def _suite_family_catalog(cognitive_loop_path: Path) -> dict[str, set[str]]:
    constants = _extract_family_constants(cognitive_loop_path)
    template_rows = constants[_SUITE_CONSTANTS["template_generalization"]]
    finality_rows = constants[_SUITE_CONSTANTS["finality_regression"]]
    tinystory_rows = constants[_SUITE_CONSTANTS["tinystories_state_tracking"]]
    mixed_rows = constants[_SUITE_CONSTANTS["mixed_narrative_symbolic"]]

    def _families(rows: list[dict[str, Any]]) -> set[str]:
        out = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            family = row.get("family")
            if family:
                out.add(str(family))
        return out

    template_families = _families(template_rows)
    finality_families = _families(finality_rows)
    tinystory_families = _families(tinystory_rows)
    mixed_families = _families(mixed_rows)
    return {
        "template_generalization": template_families,
        "finality_regression": finality_families,
        "tinystories_state_tracking": tinystory_families,
        "mixed_narrative_symbolic": mixed_families,
        "combined": set(finality_families).union(template_families),
        "combined_widening": set(finality_families)
        .union(template_families)
        .union(tinystory_families)
        .union(mixed_families),
    }


def _metric_value(profile_report: dict[str, Any], path: str) -> float:
    current: Any = profile_report
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"metric path not found: {path}")
        current = current[part]
    if isinstance(current, bool):
        return float(int(current))
    if isinstance(current, (int, float)):
        return float(current)
    raise TypeError(f"metric path {path} is not numeric: {current!r}")


def _compare(actual: float, op: str, target: float) -> bool:
    if op == "==":
        return abs(actual - target) <= 1e-9
    if op == ">=":
        return actual >= target
    if op == ">":
        return actual > target
    if op == "<=":
        return actual <= target
    if op == "<":
        return actual < target
    raise ValueError(f"unsupported comparator: {op}")


def _validate_contract(contract: dict[str, Any], catalog: dict[str, set[str]]) -> list[dict[str, Any]]:
    if str(contract.get("baseline_contract_id")) != BASELINE_CONTRACT_ID:
        raise ValueError(
            f"baseline_contract_id must be '{BASELINE_CONTRACT_ID}', got {contract.get('baseline_contract_id')!r}"
        )

    slices = contract.get("slices")
    if not isinstance(slices, list) or not slices:
        raise ValueError("contract must define a non-empty slices list")

    seen_slice_ids = set()
    default_profiles = contract.get("default_profiles") or []
    if not isinstance(default_profiles, list) or not default_profiles:
        raise ValueError("contract must define non-empty default_profiles")

    validated: list[dict[str, Any]] = []
    for raw in slices:
        if not isinstance(raw, dict):
            raise ValueError("each slice must be a JSON object")
        slice_id = str(raw.get("id") or "").strip()
        if not slice_id:
            raise ValueError("slice.id is required")
        if slice_id in seen_slice_ids:
            raise ValueError(f"duplicate slice id: {slice_id}")
        seen_slice_ids.add(slice_id)

        suite = str(raw.get("suite") or "").strip()
        if suite not in catalog:
            raise ValueError(f"slice {slice_id}: unsupported suite '{suite}'")

        families = raw.get("families")
        if not isinstance(families, list) or not families:
            raise ValueError(f"slice {slice_id}: families must be a non-empty list")
        family_names = [str(item).strip() for item in families if str(item).strip()]
        unknown = sorted(set(family_names).difference(catalog[suite]))
        if unknown:
            raise ValueError(
                f"slice {slice_id}: unknown family ids for suite '{suite}': {', '.join(unknown)}"
            )

        profiles = raw.get("profiles")
        if profiles is None:
            profiles = list(default_profiles)
        if not isinstance(profiles, list) or not profiles:
            raise ValueError(f"slice {slice_id}: profiles must be a non-empty list")

        gates = raw.get("gates")
        if not isinstance(gates, list) or not gates:
            raise ValueError(f"slice {slice_id}: gates must be a non-empty list")

        for gate in gates:
            if not isinstance(gate, dict):
                raise ValueError(f"slice {slice_id}: each gate must be a JSON object")
            gate_id = str(gate.get("id") or "").strip()
            if not gate_id:
                raise ValueError(f"slice {slice_id}: gate.id is required")
            profile = str(gate.get("profile") or "").strip()
            if not profile:
                raise ValueError(f"slice {slice_id} gate {gate_id}: profile is required")
            if profile not in profiles:
                raise ValueError(
                    f"slice {slice_id} gate {gate_id}: profile '{profile}' not listed in slice profiles"
                )
            metric = str(gate.get("metric") or "").strip()
            if not metric:
                raise ValueError(f"slice {slice_id} gate {gate_id}: metric is required")
            op = str(gate.get("op") or "").strip()
            if op not in _OPS:
                raise ValueError(
                    f"slice {slice_id} gate {gate_id}: op must be one of {sorted(_OPS)}"
                )
            has_value = "value" in gate
            has_value_from = "value_from" in gate
            if has_value == has_value_from:
                raise ValueError(
                    f"slice {slice_id} gate {gate_id}: exactly one of value or value_from is required"
                )

        validated.append(
            {
                **raw,
                "id": slice_id,
                "suite": suite,
                "families": family_names,
                "profiles": [str(p) for p in profiles],
            }
        )
    return validated


def _build_sar_config(args: argparse.Namespace) -> dict[str, Any]:
    if str(getattr(args, "sar_config_file", "") or "").strip():
        src = Path(str(args.sar_config_file))
        if not src.is_absolute():
            src = REPO_ROOT / src
        loaded = load_semantic_ar_config(src).to_dict()
        if args.sar_enabled:
            loaded["enabled"] = True
        if args.sar_emit_metadata:
            loaded["emit_metadata"] = True
        return SemanticARConfig.from_mapping(loaded).to_dict()
    return SemanticARConfig.from_mapping(
        {
            "enabled": args.sar_enabled,
            "weight": args.sar_weight,
            "lexical_weight": args.sar_lexical_weight,
            "role_weight": args.sar_role_weight,
            "prior_weight": args.sar_prior_weight,
            "gate_fail_penalty": args.sar_gate_fail_penalty,
            "couple_to_symbolic_gate": not args.sar_disable_symbolic_gate_coupling,
            "emit_metadata": args.sar_emit_metadata,
        }
    ).to_dict()


def _build_baseline_command(
    *,
    python_executable: str,
    baseline_runner: str,
    suite: str,
    profiles: list[str],
    max_cycles: int,
    families: list[str],
    output_path: Path,
    quiet: bool,
    sar_config: dict[str, Any],
) -> list[str]:
    cmd = [
        python_executable,
        str(REPO_ROOT / baseline_runner),
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
    ]
    if quiet:
        cmd.append("--quiet")
    if bool(sar_config.get("enabled", False)):
        cmd.extend(["--sar-enabled"])
    cmd.extend(["--sar-weight", str(sar_config.get("weight", 0.2))])
    cmd.extend(["--sar-lexical-weight", str(sar_config.get("lexical_weight", 0.5))])
    cmd.extend(["--sar-role-weight", str(sar_config.get("role_weight", 0.3))])
    cmd.extend(["--sar-prior-weight", str(sar_config.get("prior_weight", 0.2))])
    cmd.extend(["--sar-gate-fail-penalty", str(sar_config.get("gate_fail_penalty", 0.0))])
    if not bool(sar_config.get("couple_to_symbolic_gate", True)):
        cmd.append("--sar-disable-symbolic-gate-coupling")
    if bool(sar_config.get("emit_metadata", False)):
        cmd.append("--sar-emit-metadata")
    return cmd


def _evaluate_gates(
    *,
    gates: list[dict[str, Any]],
    profile_reports: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    results = []
    has_blocking_failure = False

    for gate in gates:
        gate_id = str(gate.get("id"))
        profile_name = str(gate.get("profile"))
        op = str(gate.get("op"))
        blocking = bool(gate.get("blocking", True))

        profile_report = profile_reports.get(profile_name)
        if profile_report is None:
            result = {
                "id": gate_id,
                "description": str(gate.get("description") or ""),
                "profile": profile_name,
                "metric": str(gate.get("metric")),
                "op": op,
                "blocking": blocking,
                "passed": False,
                "error": f"missing profile report: {profile_name}",
            }
            if blocking:
                has_blocking_failure = True
            results.append(result)
            continue

        try:
            actual = _metric_value(profile_report, str(gate.get("metric")))
            if "value_from" in gate:
                target = _metric_value(profile_report, str(gate.get("value_from")))
                target_source = str(gate.get("value_from"))
            else:
                target = float(gate.get("value"))
                target_source = "value"
            passed = _compare(actual, op, target)
            if blocking and not passed:
                has_blocking_failure = True
            results.append(
                {
                    "id": gate_id,
                    "description": str(gate.get("description") or ""),
                    "profile": profile_name,
                    "metric": str(gate.get("metric")),
                    "op": op,
                    "blocking": blocking,
                    "actual": actual,
                    "target": target,
                    "target_source": target_source,
                    "passed": passed,
                }
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            if blocking:
                has_blocking_failure = True
            results.append(
                {
                    "id": gate_id,
                    "description": str(gate.get("description") or ""),
                    "profile": profile_name,
                    "metric": str(gate.get("metric")),
                    "op": op,
                    "blocking": blocking,
                    "passed": False,
                    "error": str(exc),
                }
            )

    return results, not has_blocking_failure


def _print_slice_summary(slice_result: dict[str, Any]) -> None:
    gate_pass = sum(1 for item in slice_result["gate_results"] if item.get("passed"))
    gate_total = len(slice_result["gate_results"])
    status = "GREEN" if slice_result.get("gates_green") else "RED"
    print(f"[{status}] {slice_result['slice_id']}: {gate_pass}/{gate_total} gates passed")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the invariant-first AGI bootstrap curriculum contract with baseline-compatible slices."
    )
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--slice", nargs="+", default=None, help="Optional slice ids to run.")
    parser.add_argument("--profiles", nargs="+", default=None, help="Override profiles for every selected slice.")
    parser.add_argument("--max-cycles", type=int, default=None, help="Override max cycles for every selected slice.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for baseline subprocess runs.")
    parser.add_argument("--check-only", action="store_true", help="Validate contract and commands only; do not execute baselines.")
    parser.add_argument("--continue-on-fail", action="store_true", help="Continue running later slices after a blocking failure.")
    parser.add_argument("--print-commands", action="store_true", help="Print resolved baseline commands.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--sar-enabled", action="store_true", help="Enable semantic-AR proposal shaping in baseline runs.")
    parser.add_argument("--sar-config-file", default="", help="Optional JSON file containing a trained SAR config.")
    parser.add_argument("--sar-weight", type=float, default=0.2, help="SAR bonus weight added to proposal scores.")
    parser.add_argument("--sar-lexical-weight", type=float, default=0.5, help="Weight for lexical alignment inside SAR.")
    parser.add_argument("--sar-role-weight", type=float, default=0.3, help="Weight for role-coverage inside SAR.")
    parser.add_argument("--sar-prior-weight", type=float, default=0.2, help="Weight for concept prior inside SAR.")
    parser.add_argument("--sar-gate-fail-penalty", type=float, default=0.0, help="Penalty applied when symbolic pre-gate fails.")
    parser.add_argument("--sar-disable-symbolic-gate-coupling", action="store_true", help="Ablation: allow SAR bonus without symbolic pre-gate pass.")
    parser.add_argument("--sar-emit-metadata", action="store_true", help="Attach SAR score metadata to System-1 parse facts.")
    args = parser.parse_args()
    sar_config = _build_sar_config(args)

    contract_path = Path(args.contract)
    if not contract_path.is_absolute():
        contract_path = REPO_ROOT / contract_path
    if not contract_path.exists():
        raise FileNotFoundError(f"contract file not found: {contract_path}")

    contract = _load_json(contract_path)
    baseline_runner = str(contract.get("baseline_runner") or "rules/run_bootstrap_baseline.py")
    if not (REPO_ROOT / baseline_runner).exists():
        raise FileNotFoundError(f"baseline runner not found: {REPO_ROOT / baseline_runner}")

    catalog = _suite_family_catalog(REPO_ROOT / "cognitive_loop.py")
    slices = _validate_contract(contract, catalog)
    selected_ids = set(args.slice or [item["id"] for item in slices])
    selected_slices = [item for item in slices if item["id"] in selected_ids]
    missing_selected = selected_ids.difference({item["id"] for item in selected_slices})
    if missing_selected:
        raise ValueError(f"unknown slice id(s): {', '.join(sorted(missing_selected))}")
    if not selected_slices:
        raise ValueError("no slices selected")

    default_cycles = int(contract.get("default_max_cycles") or 2)
    commands_preview = []
    for item in selected_slices:
        profiles = list(args.profiles or item.get("profiles") or contract.get("default_profiles") or [])
        max_cycles = max(1, int(args.max_cycles if args.max_cycles is not None else item.get("max_cycles", default_cycles)))
        preview_output = Path(args.output)
        preview_baseline_path = (
            preview_output.parent / f"{preview_output.stem}.{item['id']}.baseline.json"
            if preview_output.suffix
            else Path("rules/reports") / f"{item['id']}.baseline.json"
        )
        cmd = _build_baseline_command(
            python_executable=str(args.python),
            baseline_runner=baseline_runner,
            suite=str(item["suite"]),
            profiles=profiles,
            max_cycles=max_cycles,
            families=list(item["families"]),
            output_path=preview_baseline_path,
            quiet=args.quiet,
            sar_config=sar_config,
        )
        commands_preview.append({"slice_id": item["id"], "command": shlex.join(cmd)})

    if args.check_only:
        print(f"Contract {contract.get('contract_id')} validated ({len(selected_slices)} slices).")
        if args.print_commands:
            for entry in commands_preview:
                print(f"{entry['slice_id']}: {entry['command']}")
        return 0

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_results: list[dict[str, Any]] = []
    halted_on_slice = None
    all_gates_green = True

    for item in selected_slices:
        profiles = list(args.profiles or item.get("profiles") or contract.get("default_profiles") or [])
        max_cycles = max(1, int(args.max_cycles if args.max_cycles is not None else item.get("max_cycles", default_cycles)))
        baseline_output_path = output_path.parent / f"{output_path.stem}.{item['id']}.baseline.json"
        cmd = _build_baseline_command(
            python_executable=str(args.python),
            baseline_runner=baseline_runner,
            suite=str(item["suite"]),
            profiles=profiles,
            max_cycles=max_cycles,
            families=list(item["families"]),
            output_path=baseline_output_path,
            quiet=args.quiet,
            sar_config=sar_config,
        )
        if args.print_commands:
            print(shlex.join(cmd))

        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            all_gates_green = False
            slice_result = {
                "slice_id": item["id"],
                "suite": item["suite"],
                "families": list(item["families"]),
                "profiles": profiles,
                "max_cycles": max_cycles,
                "proposal_shaping": {"semantic_ar": sar_config},
                "command": shlex.join(cmd),
                "return_code": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "gate_results": [],
                "gates_green": False,
                "error": "baseline run failed",
            }
            run_results.append(slice_result)
            _print_slice_summary(slice_result)
            halted_on_slice = item["id"]
            if not args.continue_on_fail:
                break
            continue

        baseline_payload = _load_json(baseline_output_path)
        profile_reports = {
            str(report.get("profile")): report
            for report in baseline_payload.get("profiles", [])
            if isinstance(report, dict) and report.get("profile")
        }
        gate_results, gates_green = _evaluate_gates(
            gates=list(item.get("gates") or []),
            profile_reports=profile_reports,
        )
        if not gates_green:
            all_gates_green = False
            halted_on_slice = halted_on_slice or item["id"]

        slice_result = {
            "slice_id": item["id"],
            "description": item.get("description"),
            "suite": item["suite"],
            "families": list(item["families"]),
            "profiles": profiles,
            "max_cycles": max_cycles,
            "proposal_shaping": {"semantic_ar": sar_config},
            "command": shlex.join(cmd),
            "return_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "baseline_report_path": str(baseline_output_path),
            "baseline_report": baseline_payload,
            "gate_results": gate_results,
            "gates_green": gates_green,
        }
        run_results.append(slice_result)
        _print_slice_summary(slice_result)

        if not gates_green and not args.continue_on_fail:
            break

    executed_ids = [item["slice_id"] for item in run_results]
    skipped_ids = [item["id"] for item in selected_slices if item["id"] not in executed_ids]
    output_payload = {
        "contract_id": contract.get("contract_id"),
        "baseline_contract_id": contract.get("baseline_contract_id"),
        "proposal_shaping": {
            "semantic_ar": sar_config,
            "symbolic_validator_authority": "System-2 semantic validation and WM equation-graph checks remain authoritative.",
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "contract_path": str(contract_path),
        "output_path": str(output_path),
        "selected_slice_ids": [item["id"] for item in selected_slices],
        "executed_slice_ids": executed_ids,
        "skipped_slice_ids": skipped_ids,
        "stop_on_fail": not args.continue_on_fail,
        "halted_on_slice": halted_on_slice,
        "all_gates_green": bool(all_gates_green and not skipped_ids),
        "regression": contract.get("regression") or {},
        "slices": run_results,
    }
    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    print(f"Saved invariant curriculum report to {output_path}")
    if output_payload["all_gates_green"]:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
