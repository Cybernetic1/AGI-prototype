from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import yaml

from working_memory.predicate_logic import PredicateFact


@dataclass
class RuleTemplate:
    name: str
    family: str
    arity_limit: int
    template: str
    notes: List[str] = field(default_factory=list)

    @staticmethod
    def _split_conjunction(text: str) -> List[str]:
        return [part.strip() for part in re.split(r"\s+and\s+", text) if part.strip()]

    @staticmethod
    def _parse_atom(atom_text: str) -> Tuple[str, List[str]]:
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)$", atom_text.strip())
        if not match:
            raise ValueError(f"Could not parse atom: {atom_text!r}")
        predicate = match.group(1)
        args_text = match.group(2).strip()
        if not args_text:
            return predicate, []
        args = [piece.strip() for piece in args_text.split(",")]
        return predicate, args

    def left_atoms(self) -> List[Tuple[str, List[str]]]:
        left, _ = self.template.split("->", 1)
        return [self._parse_atom(atom) for atom in self._split_conjunction(left)]

    def right_atoms(self) -> List[Tuple[str, List[str]]]:
        _, right = self.template.split("->", 1)
        return [self._parse_atom(atom) for atom in self._split_conjunction(right)]


class CanonicalSystem2:
    """Minimal symbolic reasoner over unary/binary predicate atoms.

    This is a refactoring target: it keeps the logic layer in classical
    predicate-logic form with arity <= 2, while the older record-like WME model
    is treated as a compatibility layer.
    """

    def __init__(self, rule_kb_path: str | Path | None = None):
        self.rule_kb_path = Path(rule_kb_path) if rule_kb_path else Path(__file__).resolve().parents[1] / "rules" / "sys2_rule_kb.yaml"
        self.rules: List[RuleTemplate] = self._load_rules(self.rule_kb_path)
        self.facts: List[PredicateFact] = []

    @staticmethod
    def _load_rules(path: Path) -> List[RuleTemplate]:
        if not path.exists():
            return []
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or []
        rules: List[RuleTemplate] = []
        for item in payload:
            rules.append(
                RuleTemplate(
                    name=item.get("name", "unnamed"),
                    family=item.get("family", "generic"),
                    arity_limit=int(item.get("arity_limit", 2)),
                    template=item.get("template", ""),
                    notes=item.get("notes", []),
                )
            )
        return rules

    def add_fact(self, fact: PredicateFact) -> None:
        self.facts.append(fact)

    def add_facts(self, facts: Iterable[PredicateFact]) -> None:
        for fact in facts:
            self.add_fact(fact)

    def _is_var(self, token: str) -> bool:
        return bool(token) and token[0].isupper()

    def _match_atom(self, pattern: Tuple[str, List[str]], fact: PredicateFact) -> Dict[str, Any] | None:
        pred, pattern_args = pattern
        if fact.predicate.lower() != pred.lower():
            return None
        if len(pattern_args) != len(fact.args):
            return None
        binding: Dict[str, Any] = {}
        for pat, actual in zip(pattern_args, fact.args):
            if self._is_var(pat):
                binding[pat] = actual
            else:
                if pat != str(actual):
                    return None
        return binding

    def _apply_rule(self, rule: RuleTemplate) -> List[PredicateFact]:
        left = rule.left_atoms()
        outcomes: List[PredicateFact] = []
        # This is intentionally a narrow, explicit version of the rule DSL for the
        # current semantic families. It works on canonical unary/binary atoms and is
        # easy to extend when adding new invariant families.
        if rule.name == "state_transition_delta":
            by_event = {}
            for fact in self.facts:
                if fact.predicate in {"has", "lose"}:
                    by_event.setdefault(fact.args[0], {})[fact.predicate] = fact
                elif fact.predicate == "owner":
                    by_event.setdefault(fact.args[0], {})["owner"] = fact.args[1]
                elif fact.predicate == "item":
                    by_event.setdefault(fact.args[0], {})["item"] = fact.args[1]
                elif fact.predicate == "qty":
                    by_event.setdefault(fact.args[0], {})["qty"] = fact.args[1]

            # The arithmetic rule is over a pair of different events with the same
            # owner/item identity, not a single event that contains both predicates.
            seen_pairs = set()
            for event_id, payload in by_event.items():
                owner = payload.get("owner")
                item = payload.get("item")
                q1 = payload.get("qty")
                if owner is None or item is None:
                    continue
                if "has" in payload:
                    base_event = payload
                    target_kind = "lose"
                elif "lose" in payload:
                    base_event = payload
                    target_kind = "has"
                else:
                    continue

                for other_id, other in by_event.items():
                    if other_id == event_id or (event_id, other_id) in seen_pairs or (other_id, event_id) in seen_pairs:
                        continue
                    if other.get("owner") != owner or other.get("item") != item:
                        continue
                    if target_kind not in other:
                        continue
                    q2 = other.get("qty")
                    if q1 is None or q2 is None:
                        continue
                    result = float(q1) - float(q2)
                    state_id = f"s_{owner}_{item}_state"
                    outcomes.extend([
                        PredicateFact("state", (state_id,)),
                        PredicateFact("owner", (state_id, owner)),
                        PredicateFact("item", (state_id, item)),
                        PredicateFact("before", (state_id, q1)),
                        PredicateFact("delta", (state_id, -float(q2))),
                        PredicateFact("after", (state_id, result)),
                    ])
                    seen_pairs.add((event_id, other_id))
                    return outcomes
            return outcomes

        if rule.name == "remaining_quantity_query":
            state_id = None
            owner = None
            item = None
            q_after = None
            for fact in self.facts:
                if fact.predicate == "after":
                    state_id = fact.args[0]
                    q_after = fact.args[1]
                elif fact.predicate == "owner" and fact.args[0] == state_id:
                    owner = fact.args[1]
                elif fact.predicate == "item" and fact.args[0] == state_id:
                    item = fact.args[1]
            if state_id is not None and owner is not None and item is not None and q_after is not None:
                return [
                    PredicateFact("belief", ("answer", q_after)),
                    PredicateFact("answer_owner", ("answer", owner)),
                    PredicateFact("answer_item", ("answer", item)),
                ]
            return outcomes

        # Generic but intentionally bounded matching over a very small template DSL.
        for index, left_pat in enumerate(left):
            if left_pat[0] == "same_owner_item":
                # This is a conceptual helper that is built by looking at owner/item
                # pair consistency in the canonical fact base.
                owner_vals = set()
                item_vals = set()
                for fact in self.facts:
                    if fact.predicate == "owner":
                        owner_vals.add(fact.args[1])
                    if fact.predicate == "item":
                        item_vals.add(fact.args[1])
                if owner_vals and item_vals:
                    for owner in sorted(owner_vals):
                        for item in sorted(item_vals):
                            outcomes.append(PredicateFact("same_owner_item", (owner, item)))
                return outcomes

        return outcomes

    def run(self) -> List[PredicateFact]:
        """Execute the canonical rule set over the currently loaded fact set."""
        derived: List[PredicateFact] = []
        for rule in self.rules:
            derived.extend(self._apply_rule(rule))
        # Ensure we do not duplicate logically identical facts.
        seen = set()
        unique: List[PredicateFact] = []
        for fact in derived:
            key = (fact.predicate, fact.args)
            if key in seen:
                continue
            seen.add(key)
            unique.append(fact)
        return unique


def load_default_system2() -> CanonicalSystem2:
    return CanonicalSystem2(Path(__file__).resolve().parents[1] / "rules" / "sys2_rule_kb.yaml")
