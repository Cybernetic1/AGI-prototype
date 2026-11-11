#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_RUNNER = REPO_ROOT / "rules" / "run_bootstrap_baseline.py"
REAL_VALIDATOR = REPO_ROOT / "rules" / "validate_real_gsm8k.py"
DEFAULT_CONFIG = REPO_ROOT / "rules" / "configs" / "gpu_trial_v1.json"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def _safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _profile_metrics(report: dict[str, Any]) -> dict[str, float]:
    profiles = report.get("profiles") if isinstance(report.get("profiles"), list) else []
    solve = []
    symbolic = []
    wall = []
    for profile in profiles:
        if not isinstance(profile, dict):
            continue
        solve.append(float(((profile.get("capability") or {}).get("solve_rate", 0.0)) or 0.0))
        symbolic.append(float(((profile.get("symbolic_validity") or {}).get("symbolic_valid_rate", 0.0)) or 0.0))
        wall.append(float(((profile.get("compute_cost") or {}).get("wall_time_sec_total", 0.0)) or 0.0))
    return {
        "solve_rate_mean": _safe_mean(solve),
        "symbolic_valid_rate_mean": _safe_mean(symbolic),
        "baseline_wall_time_sec_sum_profiles": float(sum(wall)),
    }


def _run_command(*, cmd: list[str], cwd: Path, dry_run: bool, env_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "returncode": 0,
            "duration_sec": 0.0,
            "stdout": "",
            "stderr": "",
        }
    env = os.environ.copy()
    for key, value in (env_overrides or {}).items():
        env[str(key)] = str(value)
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, env=env)
    duration = max(0.0, time.perf_counter() - started)
    return {
        "ok": proc.returncode == 0,
        "dry_run": False,
        "returncode": int(proc.returncode),
        "duration_sec": duration,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _arm_hourly_cost(config: dict[str, Any], arm: str) -> float:
    by_arm = config.get("gpu_hourly_usd_by_arm")
    if isinstance(by_arm, dict) and arm in by_arm:
        return float(by_arm.get(arm) or 0.0)
    return float(config.get("gpu_hourly_usd", 0.0) or 0.0)


def _arm_runs(config: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    key = "pilot_runs" if mode == "pilot" else "batch_runs"
    raw = config.get(key)
    if not isinstance(raw, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        if not bool(item.get("enabled", True)):
            continue
        rows.append(item)
    return rows


def _build_baseline_command(
    *,
    python_exec: str,
    run_cfg: dict[str, Any],
    config: dict[str, Any],
    baseline_out: Path,
) -> list[str]:
    profiles = run_cfg.get("profiles") if isinstance(run_cfg.get("profiles"), list) else ["hybrid"]
    default_flags = [str(x) for x in (config.get("default_flags") or [])]
    cmd = [
        python_exec,
        str(BASELINE_RUNNER),
        "--suite",
        str(config.get("baseline_suite", "combined")),
        "--profiles",
        *[str(x) for x in profiles],
        "--max-cycles",
        str(int(config.get("max_cycles", 2))),
        "--output",
        str(baseline_out),
        "--quiet",
        "--seed",
        str(int(run_cfg.get("seed", 0))),
    ]
    limit = int(config.get("baseline_limit", 0) or 0)
    if limit > 0:
        cmd.extend(["--limit", str(limit)])
    cmd.extend(default_flags)
    cmd.extend([str(x) for x in (run_cfg.get("extra_flags") or [])])
    return cmd


def _build_validator_command(
    *,
    python_exec: str,
    run_cfg: dict[str, Any],
    config: dict[str, Any],
    validator_out: Path,
) -> list[str]:
    return [
        python_exec,
        str(REAL_VALIDATOR),
        "--sample-size",
        str(int(config.get("real_gsm8k_sample_size", 20))),
        "--seed",
        str(int(run_cfg.get("seed", 0))),
        "--cycles",
        str(int(config.get("real_gsm8k_cycles", 4))),
        "--json-out",
        str(validator_out),
        "--quiet",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run cost-aware GPU trial matrix for LT/RL/hybrid AGI experiments.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--mode", choices=["pilot", "batch"], default="pilot")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-runs", type=int, default=0, help="Optional cap for enabled runs; 0 means all.")
    parser.add_argument("--dry-run", action="store_true", help="Print and log commands without executing.")
    parser.add_argument("--output", default="", help="Optional explicit output JSON path.")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    config = _load_json(cfg_path)

    out_dir = Path(str(config.get("output_dir", "rules/reports/trials")))
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    output_path = Path(args.output) if str(args.output).strip() else (out_dir / f"gpu_trial_matrix_{args.mode}_{stamp}.json")
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_rows = _arm_runs(config, args.mode)
    if args.max_runs > 0:
        run_rows = run_rows[: int(args.max_runs)]

    if not run_rows:
        raise ValueError(f"no enabled runs found for mode={args.mode}")

    records: list[dict[str, Any]] = []
    for run_cfg in run_rows:
        run_id = str(run_cfg.get("run_id") or "run")
        arm = str(run_cfg.get("arm") or "unknown")
        run_dir = out_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        baseline_out = run_dir / "baseline_report.json"
        validator_out = run_dir / "real_gsm8k_report.json"

        baseline_cmd = _build_baseline_command(
            python_exec=str(args.python),
            run_cfg=run_cfg,
            config=config,
            baseline_out=baseline_out,
        )
        validator_cmd = _build_validator_command(
            python_exec=str(args.python),
            run_cfg=run_cfg,
            config=config,
            validator_out=validator_out,
        )

        env_overrides = run_cfg.get("env") if isinstance(run_cfg.get("env"), dict) else {}
        baseline_exec = _run_command(
            cmd=baseline_cmd,
            cwd=REPO_ROOT,
            dry_run=bool(args.dry_run),
            env_overrides=env_overrides,
        )
        validator_exec = _run_command(
            cmd=validator_cmd,
            cwd=REPO_ROOT,
            dry_run=bool(args.dry_run),
            env_overrides=env_overrides,
        )
        total_wall_sec = float(baseline_exec["duration_sec"]) + float(validator_exec["duration_sec"])
        hourly_usd = _arm_hourly_cost(config, arm)
        cost_usd = (total_wall_sec / 3600.0) * hourly_usd

        baseline_metrics = {}
        validator_metrics = {}
        if (not args.dry_run) and baseline_exec["ok"] and baseline_out.exists():
            baseline_report = _load_json(baseline_out)
            baseline_metrics = _profile_metrics(baseline_report)
        if (not args.dry_run) and validator_exec["ok"] and validator_out.exists():
            validator_report = _load_json(validator_out)
            validator_metrics = {
                "solve_rate": float(validator_report.get("solve_rate", 0.0) or 0.0),
                "answer_generation_rate": float(validator_report.get("answer_generation_rate", 0.0) or 0.0),
                "sample_size": int(validator_report.get("sample_size", 0) or 0),
                "solved": int(validator_report.get("solved", 0) or 0),
            }

        records.append(
            {
                "run_id": run_id,
                "arm": arm,
                "seed": int(run_cfg.get("seed", 0) or 0),
                "profiles": [str(x) for x in (run_cfg.get("profiles") or [])],
                "env": env_overrides,
                "baseline_cmd": baseline_cmd,
                "validator_cmd": validator_cmd,
                "baseline_exec": baseline_exec,
                "validator_exec": validator_exec,
                "baseline_metrics": baseline_metrics,
                "validator_metrics": validator_metrics,
                "gpu_hourly_usd": hourly_usd,
                "wall_time_sec_total": total_wall_sec,
                "estimated_cost_usd": cost_usd,
            }
        )

    by_arm: dict[str, dict[str, Any]] = {}
    for row in records:
        arm = str(row.get("arm") or "unknown")
        bucket = by_arm.setdefault(
            arm,
            {
                "run_count": 0,
                "cost_usd_total": 0.0,
                "wall_time_sec_total": 0.0,
                "validator_solve_rates": [],
                "validator_answer_generation_rates": [],
                "baseline_solve_rate_means": [],
                "baseline_symbolic_valid_rate_means": [],
            },
        )
        bucket["run_count"] += 1
        bucket["cost_usd_total"] += float(row.get("estimated_cost_usd", 0.0) or 0.0)
        bucket["wall_time_sec_total"] += float(row.get("wall_time_sec_total", 0.0) or 0.0)
        v = row.get("validator_metrics") or {}
        b = row.get("baseline_metrics") or {}
        bucket["validator_solve_rates"].append(float(v.get("solve_rate", 0.0) or 0.0))
        bucket["validator_answer_generation_rates"].append(float(v.get("answer_generation_rate", 0.0) or 0.0))
        bucket["baseline_solve_rate_means"].append(float(b.get("solve_rate_mean", 0.0) or 0.0))
        bucket["baseline_symbolic_valid_rate_means"].append(float(b.get("symbolic_valid_rate_mean", 0.0) or 0.0))

    arm_summary = []
    for arm, stats in sorted(by_arm.items()):
        run_count = int(stats["run_count"])
        cost_total = float(stats["cost_usd_total"])
        baseline_symbolic_mean = _safe_mean(stats["baseline_symbolic_valid_rate_means"])
        validator_solve_mean = _safe_mean(stats["validator_solve_rates"])
        summary = {
            "arm": arm,
            "run_count": run_count,
            "cost_usd_total": cost_total,
            "cost_usd_per_run": (cost_total / run_count) if run_count else 0.0,
            "wall_time_sec_total": float(stats["wall_time_sec_total"]),
            "baseline_solve_rate_mean": _safe_mean(stats["baseline_solve_rate_means"]),
            "baseline_symbolic_valid_rate_mean": baseline_symbolic_mean,
            "validator_solve_rate_mean": validator_solve_mean,
            "validator_answer_generation_rate_mean": _safe_mean(stats["validator_answer_generation_rates"]),
            "cost_per_baseline_symbolic_valid_point": (cost_total / baseline_symbolic_mean) if baseline_symbolic_mean > 0 else None,
            "cost_per_validator_solve_point": (cost_total / validator_solve_mean) if validator_solve_mean > 0 else None,
        }
        arm_summary.append(summary)

    payload = {
        "trial_id": str(config.get("trial_id", "gpu-trial")),
        "mode": str(args.mode),
        "generated_at": now.isoformat(),
        "config_path": str(cfg_path),
        "dry_run": bool(args.dry_run),
        "run_count": len(records),
        "records": records,
        "arm_summary": arm_summary,
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest = output_path.parent / f"gpu_trial_matrix_{args.mode}_latest.json"
    latest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved GPU trial matrix report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
