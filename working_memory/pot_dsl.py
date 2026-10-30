from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Tuple


_TRUE_VALUES = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class PoTStep:
    op: str
    out: str
    args: tuple[Any, ...]
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PoTProgram:
    program_id: str
    steps: tuple[PoTStep, ...]


@dataclass(frozen=True)
class PoTValidationIssue:
    step_index: int
    field: str
    reason: str


@dataclass(frozen=True)
class PoTExecutionResult:
    ok: bool
    vars: Dict[str, float]
    states: Dict[Tuple[str, str], float]
    trace: List[Dict[str, Any]]
    issues: List[PoTValidationIssue]


_ARITH_OPS = {"add", "sub", "mul", "div"}
_STATE_OPS = {"state_set", "state_add_delta", "state_apply_percent_reduction", "state_query"}
_UTILITY_OPS = {"const", "percent_of", "remaining_after_percent"}
_ALL_OPS = _ARITH_OPS.union(_STATE_OPS).union(_UTILITY_OPS)


def _to_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value.strip())
    raise TypeError(f"expected numeric value, got {value!r}")


def _resolve_arg(value: Any, variables: Dict[str, float]) -> float:
    # "$x" means read variable x.
    if isinstance(value, str) and value.startswith("$"):
        key = value[1:]
        if key not in variables:
            raise KeyError(f"unknown variable reference: {value}")
        return float(variables[key])
    return _to_float(value)


def _normalize_pct(pct: float) -> float:
    # Allow either 0..1 or 0..100 style percentages.
    if pct > 1.0:
        return pct / 100.0
    return pct


def validate_program(program: PoTProgram) -> List[PoTValidationIssue]:
    issues: List[PoTValidationIssue] = []
    seen_out = set()
    for idx, step in enumerate(program.steps):
        if step.op not in _ALL_OPS:
            issues.append(PoTValidationIssue(idx, "op", f"unsupported op: {step.op}"))
        if not str(step.out).strip():
            issues.append(PoTValidationIssue(idx, "out", "missing out variable"))
        if step.out in seen_out:
            issues.append(PoTValidationIssue(idx, "out", f"duplicate out variable: {step.out}"))
        seen_out.add(step.out)

        argc = len(step.args)
        if step.op in _ARITH_OPS and argc != 2:
            issues.append(PoTValidationIssue(idx, "args", f"{step.op} requires 2 args, got {argc}"))
        if step.op in {"percent_of", "remaining_after_percent"} and argc != 2:
            issues.append(PoTValidationIssue(idx, "args", f"{step.op} requires 2 args, got {argc}"))
        if step.op == "const" and argc != 1:
            issues.append(PoTValidationIssue(idx, "args", f"{step.op} requires 1 arg, got {argc}"))
        if step.op == "state_set" and argc != 3:
            issues.append(PoTValidationIssue(idx, "args", "state_set requires owner, item, qty"))
        if step.op in {"state_add_delta", "state_apply_percent_reduction"} and argc != 3:
            issues.append(PoTValidationIssue(idx, "args", f"{step.op} requires owner, item, delta/pct"))
        if step.op == "state_query" and argc != 2:
            issues.append(PoTValidationIssue(idx, "args", "state_query requires owner, item"))
    return issues


def execute_program(program: PoTProgram) -> PoTExecutionResult:
    issues = validate_program(program)
    if issues:
        return PoTExecutionResult(False, {}, {}, [], issues)

    variables: Dict[str, float] = {}
    states: Dict[Tuple[str, str], float] = {}
    trace: List[Dict[str, Any]] = []

    for idx, step in enumerate(program.steps):
        try:
            if step.op == "const":
                result = _resolve_arg(step.args[0], variables)
            elif step.op == "add":
                result = _resolve_arg(step.args[0], variables) + _resolve_arg(step.args[1], variables)
            elif step.op == "sub":
                result = _resolve_arg(step.args[0], variables) - _resolve_arg(step.args[1], variables)
            elif step.op == "mul":
                result = _resolve_arg(step.args[0], variables) * _resolve_arg(step.args[1], variables)
            elif step.op == "div":
                rhs = _resolve_arg(step.args[1], variables)
                if abs(rhs) < 1e-12:
                    raise ZeroDivisionError("division by zero")
                result = _resolve_arg(step.args[0], variables) / rhs
            elif step.op == "percent_of":
                base = _resolve_arg(step.args[0], variables)
                pct = _normalize_pct(_resolve_arg(step.args[1], variables))
                result = base * pct
            elif step.op == "remaining_after_percent":
                base = _resolve_arg(step.args[0], variables)
                pct = _normalize_pct(_resolve_arg(step.args[1], variables))
                result = base * (1.0 - pct)
            elif step.op == "state_set":
                owner, item = str(step.args[0]), str(step.args[1])
                qty = _resolve_arg(step.args[2], variables)
                states[(owner, item)] = qty
                result = qty
            elif step.op == "state_add_delta":
                owner, item = str(step.args[0]), str(step.args[1])
                key = (owner, item)
                if key not in states:
                    raise KeyError(f"unknown state for state_add_delta: {key}")
                delta = _resolve_arg(step.args[2], variables)
                states[key] = states[key] + delta
                result = states[key]
            elif step.op == "state_apply_percent_reduction":
                owner, item = str(step.args[0]), str(step.args[1])
                key = (owner, item)
                if key not in states:
                    raise KeyError(f"unknown state for state_apply_percent_reduction: {key}")
                pct = _normalize_pct(_resolve_arg(step.args[2], variables))
                states[key] = states[key] * (1.0 - pct)
                result = states[key]
            elif step.op == "state_query":
                owner, item = str(step.args[0]), str(step.args[1])
                key = (owner, item)
                if key not in states:
                    raise KeyError(f"unknown state for state_query: {key}")
                result = states[key]
            else:
                raise ValueError(f"unsupported op: {step.op}")
        except Exception as exc:
            issues.append(PoTValidationIssue(idx, "execute", str(exc)))
            return PoTExecutionResult(False, dict(variables), dict(states), trace, issues)

        variables[step.out] = float(result)
        trace.append(
            {
                "step_index": idx,
                "op": step.op,
                "out": step.out,
                "args": list(step.args),
                "result": float(result),
            }
        )

    return PoTExecutionResult(True, dict(variables), dict(states), trace, issues)


def wm_nodes_from_trace(trace: List[Dict[str, Any]]):
    # Late import avoids heavy dependency for pure validator use.
    from working_memory.working_memory import WMNode

    out = set()
    state_values: Dict[Tuple[str, str], float] = {}
    for row in trace:
        op = str(row.get("op"))
        args = list(row.get("args") or [])
        result = row.get("result")
        concept = None
        attrs: Dict[str, Any] = {"result": result}
        if op == "add":
            concept = "Add"
        elif op == "sub":
            concept = "Subtract"
        elif op == "mul":
            concept = "Multiply"
        elif op == "div":
            concept = "Divide"
        elif op == "percent_of":
            concept = "Multiply"
        elif op == "remaining_after_percent":
            concept = "Subtract"
        if concept is not None and len(args) >= 2:
            attrs["lhs"] = args[0]
            attrs["rhs"] = args[1]
            out.add(
                WMNode.create(
                    concept,
                    layer="reasoning",
                    source="pot-dsl",
                    provenance_id=f"pot:{row.get('step_index')}",
                    **attrs,
                )
            )
        if op == "state_set" and len(args) == 3:
            owner = str(args[0])
            item = str(args[1])
            qty = float(result)
            state_values[(owner, item)] = qty
            out.add(
                WMNode.create(
                    "Quantity",
                    layer="semantic",
                    source="pot-dsl",
                    provenance_id=f"pot:{row.get('step_index')}",
                    owner=owner,
                    item=item,
                    qty=qty,
                    event_order=int(row.get("step_index", 0)) + 1,
                )
            )
        if op in {"state_add_delta", "state_apply_percent_reduction"} and len(args) >= 3:
            owner = str(args[0])
            item = str(args[1])
            key = (owner, item)
            after_qty = float(result)
            before_qty = state_values.get(key)
            if before_qty is None:
                before_qty = after_qty
            delta = after_qty - before_qty
            state_values[key] = after_qty
            out.add(
                WMNode.create(
                    "StateTransition",
                    layer="semantic",
                    source="pot-dsl",
                    provenance_id=f"pot:{row.get('step_index')}",
                    owner=owner,
                    item=item,
                    before_qty=before_qty,
                    delta=delta,
                    after_qty=after_qty,
                    event_kind=op,
                    event_order=int(row.get("step_index", 0)) + 1,
                )
            )
        if op == "state_query" and len(args) == 2:
            owner = str(args[0])
            item = str(args[1])
            out.add(
                WMNode.create(
                    "FinalState",
                    layer="meta",
                    source="pot-dsl",
                    provenance_id=f"pot:{row.get('step_index')}",
                    owner=owner,
                    item=item,
                    qty=float(result),
                    event_order=int(row.get("step_index", 0)) + 1,
                    phase="final",
                )
            )
    return out


def program_from_dict(payload: Dict[str, Any]) -> PoTProgram:
    steps = []
    for raw in payload.get("steps", []):
        if not isinstance(raw, dict):
            continue
        steps.append(
            PoTStep(
                op=str(raw.get("op") or "").strip(),
                out=str(raw.get("out") or "").strip(),
                args=tuple(raw.get("args") or []),
                meta=dict(raw.get("meta") or {}),
            )
        )
    return PoTProgram(program_id=str(payload.get("program_id") or "pot-program"), steps=tuple(steps))


def program_feature_keys(program: PoTProgram) -> List[str]:
    ops = [step.op for step in program.steps]
    feats: List[str] = []
    for op in ops:
        feats.append(f"op:{op}")
    for i in range(len(ops) - 1):
        feats.append(f"bigram:{ops[i]}->{ops[i+1]}")
    feats.append(f"len:{len(ops)}")
    return feats


def score_program(program: PoTProgram, model: Dict[str, Any]) -> float:
    bias = float(model.get("bias", 0.0))
    weights = dict(model.get("weights") or {})
    score = bias
    for feat in program_feature_keys(program):
        score += float(weights.get(feat, 0.0))
    return score


def propose_program_for_question(*, question: str, family: str | None = None) -> PoTProgram | None:
    fam = str(family or "").strip().lower()
    q = str(question or "")

    if fam in {
        "buy_more_final_gap",
        "donation_after_buy_more",
        "buy_more_then_donation_and_spoilage",
        "buy_more_then_donation_after_spoilage_long_tail",
    }:
        m_ratio = re.search(r"\b([A-Za-z]+)\s+has\s+(\d+(?:\.\d+)?)\s+times\s+as\s+many\s+([A-Za-z]+)\s+as\s+([A-Za-z]+)\b", q, re.IGNORECASE)
        m_base = re.search(r"\b([A-Za-z]+)\s+has\s+(\d+(?:\.\d+)?)\s+([A-Za-z]+)\b", q, re.IGNORECASE)
        owner = "A"
        ref_owner = "B"
        item = "apples"
        factor = 3.0
        base_qty = 10.0
        if m_ratio:
            owner = str(m_ratio.group(1))
            factor = float(m_ratio.group(2))
            item = str(m_ratio.group(3))
            ref_owner = str(m_ratio.group(4))
        if m_base and str(m_base.group(1)) == ref_owner:
            base_qty = float(m_base.group(2))

        def _extract_after(keyword: str, default: float = 0.0) -> float:
            mm = re.search(rf"{keyword}\s+(\d+(?:\.\d+)?)", q, re.IGNORECASE)
            if mm:
                return float(mm.group(1))
            return default

        give_away = _extract_after(r"gives?\s+away", 0.0)
        buy_more = _extract_after(r"buys?\s+", 0.0)
        spoil = _extract_after(r"spoiled?\b", 0.0)

        eaten_pct = 0.0
        m_eat = re.search(r"(\d+(?:\.\d+)?)%\s+of\s+the\s+remaining\s+[A-Za-z]+\s+are\s+eaten", q, re.IGNORECASE)
        if m_eat:
            eaten_pct = float(m_eat.group(1)) / 100.0
        donate_pct = 0.0
        m_donate = re.search(r"(\d+(?:\.\d+)?)%\s+of\s+the\s+remaining\s+[A-Za-z]+\s+are\s+donated", q, re.IGNORECASE)
        if m_donate:
            donate_pct = float(m_donate.group(1)) / 100.0

        steps = [
            PoTStep("const", "b_qty", (base_qty,)),
            PoTStep("const", "factor", (factor,)),
            PoTStep("mul", "a_start", ("$b_qty", "$factor")),
            PoTStep("state_set", "state0", (owner, item, "$a_start")),
        ]
        if abs(give_away) > 1e-12:
            steps.append(PoTStep("state_add_delta", "after_giveaway", (owner, item, -give_away)))
        if abs(eaten_pct) > 1e-12:
            steps.append(PoTStep("state_apply_percent_reduction", "after_eat", (owner, item, eaten_pct)))
        if abs(buy_more) > 1e-12:
            steps.append(PoTStep("state_add_delta", "after_buy", (owner, item, buy_more)))
        if abs(donate_pct) > 1e-12:
            steps.append(PoTStep("state_apply_percent_reduction", "after_donate", (owner, item, donate_pct)))
        if abs(spoil) > 1e-12:
            steps.append(PoTStep("state_add_delta", "after_spoil", (owner, item, -spoil)))
        steps.append(PoTStep("state_query", "ans", (owner, item)))

        return PoTProgram(
            program_id=f"pot:{fam or 'finality'}",
            steps=tuple(steps),
        )

    if fam == "distance_over_speed" or ("miles" in q.lower() and "per hour" in q.lower()):
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", q)]
        if len(nums) >= 2:
            return PoTProgram(
                program_id=f"pot:{fam or 'distance_over_speed'}",
                steps=(
                    PoTStep("const", "distance", (nums[0],)),
                    PoTStep("const", "speed", (nums[1],)),
                    PoTStep("div", "ans", ("$distance", "$speed")),
                ),
            )
    return None
