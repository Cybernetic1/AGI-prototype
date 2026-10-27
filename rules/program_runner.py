from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from cognitive_loop import (
    FINALITY_REGRESSION_FAMILIES,
    GENERALIZED_TEMPLATE_FAMILIES,
    benchmark_failure_families,
    benchmark_finality_regressions,
    benchmark_single_problem,
    benchmark_template_generalization,
)


PROGRAMS_DIR = Path(__file__).resolve().parents[0] / "programs"


@dataclass
class ProgramSpec:
    name: str
    description: str
    execution: Dict[str, Any]
    policy: Dict[str, Any]
    seeds: Dict[str, Any]
    readiness: Dict[str, Any]
    background_allowed: bool = False


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def load_program_spec(*, program_id: Optional[str] = None, path: Optional[str | Path] = None) -> ProgramSpec:
    if path is None:
        if not program_id:
            raise ValueError("either program_id or path is required")
        path = PROGRAMS_DIR / f"{program_id}.yaml"
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"program spec not found: {file_path}")
    payload = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
    _validate_spec_payload(payload, source=file_path)

    name = str(payload.get("name") or file_path.stem)
    description = str(payload.get("description") or "")
    execution = payload.get("execution") or {}
    policy = payload.get("policy") or {}
    seeds = payload.get("seeds") or {}
    readiness = payload.get("readiness") or {}
    background_allowed = bool(payload.get("background_allowed", False))

    return ProgramSpec(
        name=name,
        description=description,
        execution=execution,
        policy=policy,
        seeds=seeds,
        readiness=readiness,
        background_allowed=background_allowed,
    )


def _benchmark_subset(size: int) -> List[Dict[str, Any]]:
    n = max(1, min(int(size), len(GENERALIZED_TEMPLATE_FAMILIES)))
    return GENERALIZED_TEMPLATE_FAMILIES[:n]


def _subset(items: List[Dict[str, Any]], size: int) -> List[Dict[str, Any]]:
    n = max(1, min(int(size), len(items)))
    return items[:n]


def list_program_specs() -> List[Path]:
    return sorted(PROGRAMS_DIR.glob("*.yaml"))


def _validate_spec_payload(payload: Dict[str, Any], *, source: Path) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"program spec must be a mapping: {source}")

    for required_key in ("execution", "policy"):
        if required_key not in payload:
            raise ValueError(f"missing required key '{required_key}' in program spec: {source}")

    execution = payload.get("execution")
    if not isinstance(execution, dict):
        raise ValueError(f"execution must be a mapping in program spec: {source}")

    mode = str(execution.get("mode") or "").strip().lower()
    allowed_modes = {"single_problem", "family_benchmark", "template_generalization", "finality_regression"}
    if mode not in allowed_modes:
        raise ValueError(f"unsupported execution.mode '{mode}' in program spec: {source}")

    if mode == "single_problem" and not str(execution.get("question") or "").strip():
        raise ValueError(f"single_problem mode requires execution.question in program spec: {source}")

    readiness = payload.get("readiness") or {}
    if readiness and not isinstance(readiness, dict):
        raise ValueError(f"readiness must be a mapping in program spec: {source}")
    if isinstance(readiness, dict) and bool(readiness.get("enabled", False)):
        if _as_int(readiness.get("subset_size"), 0) <= 0:
            raise ValueError(f"readiness.subset_size must be > 0 in program spec: {source}")
        if _as_int(readiness.get("min_solved"), 0) <= 0:
            raise ValueError(f"readiness.min_solved must be > 0 in program spec: {source}")


def run_program(spec: ProgramSpec, *, quiet: bool = True, quick: bool = False) -> Dict[str, Any]:
    mode = str(spec.execution.get("mode") or "template_generalization").strip().lower()
    max_cycles = _as_int(spec.execution.get("max_cycles"), 2 if quick else 4)

    use_ltm_recall = bool(spec.policy.get("use_ltm_recall", False))
    auto_accept_recall = bool(spec.policy.get("auto_accept_recall", False))
    write_ltm_episode = bool(spec.policy.get("write_ltm_episode", True))

    if mode == "single_problem":
        question = str(spec.execution.get("question") or "")
        if not question:
            raise ValueError("single_problem mode requires execution.question")
        expected = spec.execution.get("expected")
        result = benchmark_single_problem(
            question,
            expected,
            max_cycles=max_cycles,
            quiet=quiet,
            use_ltm_recall=use_ltm_recall,
            auto_accept_recall=auto_accept_recall,
            write_ltm_episode=write_ltm_episode,
        )
        return {"program": spec.name, "mode": mode, "result": result}

    if mode == "family_benchmark":
        subset = _benchmark_subset(_as_int(spec.execution.get("subset_size"), 4 if quick else 8))
        report = benchmark_failure_families(
            subset,
            max_cycles=max_cycles,
            quiet=quiet,
            use_ltm_recall=use_ltm_recall,
            auto_accept_recall=auto_accept_recall,
            write_ltm_episode=write_ltm_episode,
        )
        return {"program": spec.name, "mode": mode, "report": report}

    if mode == "template_generalization":
        if quick:
            subset = _benchmark_subset(_as_int(spec.execution.get("subset_size"), 4))
            report = benchmark_failure_families(
                subset,
                max_cycles=max_cycles,
                quiet=quiet,
                use_ltm_recall=use_ltm_recall,
                auto_accept_recall=auto_accept_recall,
                write_ltm_episode=write_ltm_episode,
            )
        else:
            report = benchmark_template_generalization(
                max_cycles=max_cycles,
                quiet=quiet,
                use_ltm_recall=use_ltm_recall,
                auto_accept_recall=auto_accept_recall,
                write_ltm_episode=write_ltm_episode,
            )
        return {"program": spec.name, "mode": mode, "report": report}

    if mode == "finality_regression":
        if quick:
            subset = _subset(FINALITY_REGRESSION_FAMILIES, _as_int(spec.execution.get("subset_size"), 3))
            report = benchmark_failure_families(
                subset,
                max_cycles=max_cycles,
                quiet=quiet,
                use_ltm_recall=use_ltm_recall,
                auto_accept_recall=auto_accept_recall,
                write_ltm_episode=write_ltm_episode,
            )
        else:
            report = benchmark_finality_regressions(
                max_cycles=max_cycles,
                quiet=quiet,
                use_ltm_recall=use_ltm_recall,
                auto_accept_recall=auto_accept_recall,
                write_ltm_episode=write_ltm_episode,
            )
        return {"program": spec.name, "mode": mode, "report": report}

    raise ValueError(f"unsupported execution.mode: {mode}")


def readiness_check(spec: ProgramSpec, *, quiet: bool = True) -> Dict[str, Any]:
    if not spec.readiness or not bool(spec.readiness.get("enabled", False)):
        return {"enabled": False, "ready": True, "reason": "disabled"}

    subset_size = _as_int(spec.readiness.get("subset_size"), 4)
    min_solved = _as_int(spec.readiness.get("min_solved"), subset_size)
    max_cycles = _as_int(spec.readiness.get("max_cycles"), 1)

    subset = _benchmark_subset(subset_size)
    report = benchmark_failure_families(
        subset,
        max_cycles=max_cycles,
        quiet=quiet,
        use_ltm_recall=bool(spec.policy.get("use_ltm_recall", False)),
        auto_accept_recall=bool(spec.policy.get("auto_accept_recall", False)),
        write_ltm_episode=False,
    )
    solved = int(report.get("solved", 0))
    ready = solved >= min_solved
    return {
        "enabled": True,
        "ready": ready,
        "solved": solved,
        "total": int(report.get("total", 0)),
        "min_solved": min_solved,
    }
