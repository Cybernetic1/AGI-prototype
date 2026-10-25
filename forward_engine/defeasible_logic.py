from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import yaml

from working_memory.predicate_logic import PredicateFact


Atom = Tuple[str, List[str]]


@dataclass(frozen=True)
class DefeasibleRule:
    rule_id: str
    kind: str  # strict | defeasible
    body: List[Atom]
    head: Atom
    notes: str = ""


@dataclass
class DerivedConclusion:
    fact: PredicateFact
    rule_id: str
    kind: str  # strict | defeasible


def _split_conjunction(text: str) -> List[str]:
    return [part.strip() for part in re.split(r"\s+(?:and|∧)\s+", text.strip()) if part.strip()]


def _parse_atom(atom_text: str) -> Atom:
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)$", atom_text.strip())
    if not match:
        raise ValueError(f"Could not parse atom: {atom_text!r}")
    pred = match.group(1)
    args_text = match.group(2).strip()
    if not args_text:
        return pred, []
    args = [piece.strip() for piece in args_text.split(",")]
    return pred, args


def _parse_rule_expression(expr: str) -> Tuple[str, List[Atom], Atom]:
    text = expr.strip()
    if "->" in text:
        left, right = text.split("->", 1)
        kind = "strict"
    elif "=>" in text:
        left, right = text.split("=>", 1)
        kind = "defeasible"
    elif "⟹" in text:
        left, right = text.split("⟹", 1)
        kind = "defeasible"
    else:
        raise ValueError(f"Unsupported rule syntax: {expr!r}")

    body = [_parse_atom(a) for a in _split_conjunction(left)]
    head = _parse_atom(right.strip())
    return kind, body, head


def _literal_key(fact: PredicateFact) -> Tuple[str, Tuple[Any, ...]]:
    return (fact.predicate, tuple(fact.args))


def _opposite_predicate(pred: str) -> str:
    return pred[4:] if pred.startswith("not_") else f"not_{pred}"


class DefeasibleReasoner:
    def __init__(self, rules: Sequence[DefeasibleRule], priorities: Dict[str, set[str]]):
        self.rules = list(rules)
        self.priorities = priorities
        self.facts: List[PredicateFact] = []

    @classmethod
    def from_yaml(cls, path: Path) -> "DefeasibleReasoner":
        if not path.exists():
            return cls([], {})
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        raw_rules = payload.get("rules") or []
        raw_priorities = payload.get("priorities") or []

        rules: List[DefeasibleRule] = []
        for idx, item in enumerate(raw_rules):
            rid = item.get("id") or f"r_{idx+1}"
            expr = item.get("rule") or ""
            kind, body, head = _parse_rule_expression(expr)
            declared_kind = (item.get("kind") or kind).strip().lower()
            if declared_kind in {"strict", "defeasible"}:
                kind = declared_kind
            rules.append(DefeasibleRule(rule_id=rid, kind=kind, body=body, head=head, notes=item.get("notes", "")))

        priorities: Dict[str, set[str]] = {}
        for rel in raw_priorities:
            text = str(rel).strip()
            if ">" not in text:
                continue
            winner, loser = [x.strip() for x in text.split(">", 1)]
            priorities.setdefault(winner, set()).add(loser)
        return cls(rules, priorities)

    def add_facts(self, facts: Iterable[PredicateFact]) -> None:
        self.facts.extend(facts)

    @staticmethod
    def _is_var(token: str) -> bool:
        return bool(token) and token[0].isupper()

    def _unify_atom(self, atom: Atom, fact: PredicateFact, bindings: Dict[str, Any]) -> Dict[str, Any] | None:
        pred, args = atom
        if fact.predicate.lower() != pred.lower():
            return None
        if len(args) != len(fact.args):
            return None
        merged = dict(bindings)
        for pat, actual in zip(args, fact.args):
            if self._is_var(pat):
                bound = merged.get(pat)
                if bound is None:
                    merged[pat] = actual
                elif bound != actual:
                    return None
            elif str(actual) != pat:
                return None
        return merged

    def _ground_rule(self, rule: DefeasibleRule) -> List[DerivedConclusion]:
        bindings_list: List[Dict[str, Any]] = [dict()]
        for atom in rule.body:
            next_bindings: List[Dict[str, Any]] = []
            for b in bindings_list:
                for fact in self.facts:
                    unified = self._unify_atom(atom, fact, b)
                    if unified is not None:
                        next_bindings.append(unified)
            bindings_list = next_bindings
            if not bindings_list:
                return []

        head_pred, head_args = rule.head
        outputs: List[DerivedConclusion] = []
        for b in bindings_list:
            grounded = []
            for arg in head_args:
                grounded.append(b.get(arg, arg))
            outputs.append(
                DerivedConclusion(
                    fact=PredicateFact(head_pred, tuple(grounded)),
                    rule_id=rule.rule_id,
                    kind=rule.kind,
                )
            )
        return outputs

    def _priority_beats(self, winner_rule: str, loser_rule: str, seen: set[str] | None = None) -> bool:
        if winner_rule == loser_rule:
            return False
        if seen is None:
            seen = set()
        if winner_rule in seen:
            return False
        direct = self.priorities.get(winner_rule, set())
        if loser_rule in direct:
            return True
        seen.add(winner_rule)
        return any(self._priority_beats(next_rule, loser_rule, seen) for next_rule in direct)

    def run(self) -> Tuple[List[PredicateFact], List[PredicateFact]]:
        derived = []
        for rule in self.rules:
            derived.extend(self._ground_rule(rule))

        strict_by_literal: Dict[Tuple[str, Tuple[Any, ...]], List[DerivedConclusion]] = {}
        defeasible_by_literal: Dict[Tuple[str, Tuple[Any, ...]], List[DerivedConclusion]] = {}
        for d in derived:
            key = _literal_key(d.fact)
            if d.kind == "strict":
                strict_by_literal.setdefault(key, []).append(d)
            else:
                defeasible_by_literal.setdefault(key, []).append(d)

        accepted: Dict[Tuple[str, Tuple[Any, ...]], PredicateFact] = {}
        meta: List[PredicateFact] = []

        # Strict rules are monotonic and always accepted.
        for key, supports in strict_by_literal.items():
            accepted[key] = supports[0].fact
            for s in supports:
                meta.append(PredicateFact("rule_applied", (s.rule_id, key[0])))

        # Resolve defeasible rules by direct conflict and priority relation.
        for key, supports in defeasible_by_literal.items():
            pred, args = key
            opposite_key = (_opposite_predicate(pred), args)

            if key in accepted:
                continue
            if opposite_key in strict_by_literal:
                for s in supports:
                    meta.append(PredicateFact("defeated", (s.rule_id, "strict_opponent")))
                continue

            opposite_supports = defeasible_by_literal.get(opposite_key, [])
            if not opposite_supports:
                winner = supports[0]
            else:
                winner = None
                for candidate in supports:
                    if all(self._priority_beats(candidate.rule_id, opp.rule_id) for opp in opposite_supports):
                        winner = candidate
                        break
                if winner is None:
                    for s in supports:
                        meta.append(PredicateFact("defeated", (s.rule_id, "unresolved_conflict")))
                    continue
                for opp in opposite_supports:
                    meta.append(PredicateFact("overrides", (winner.rule_id, opp.rule_id)))
                    meta.append(PredicateFact("defeated", (opp.rule_id, winner.rule_id)))

            accepted[key] = winner.fact
            meta.append(PredicateFact("rule_applied", (winner.rule_id, winner.fact.predicate)))

        unique_meta = []
        seen_meta = set()
        for m in meta:
            k = (m.predicate, m.args)
            if k in seen_meta:
                continue
            seen_meta.add(k)
            unique_meta.append(m)

        return list(accepted.values()), unique_meta


def load_default_defeasible_reasoner() -> DefeasibleReasoner:
    path = Path(__file__).resolve().parents[1] / "rules" / "defeasible_rule_kb.yaml"
    return DefeasibleReasoner.from_yaml(path)
