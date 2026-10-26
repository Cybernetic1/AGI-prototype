from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class NLRuleSpec:
    rule_id: str
    family: str
    condition: str
    effect: str
    priority: int = 1
    owner: Optional[str] = None
    item: Optional[str] = None
    value: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        payload = {
            "kind": "dependency",
            "rule_id": self.rule_id,
            "family": self.family,
            "condition": self.condition,
            "effect": self.effect,
            "priority": self.priority,
            "owner": self.owner,
            "item": self.item,
            "value": self.value,
            "source": "nl_rule",
            "relation": "rule",
        }
        payload.update(self.metadata)
        return payload


class NLRuleCompiler:
    """Compile a narrow natural-language rule into the repo's graph contract.

    The goal is intentionally narrow: support the semantic families already in the
    codebase (state precedence, remaining-quantity updates, entity binding) without
    pretending that free-form NL is the actual execution substrate.
    """

    @staticmethod
    def normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()

    @staticmethod
    def parse(raw_rule: str) -> NLRuleSpec:
        text = NLRuleCompiler.normalize(raw_rule)
        low = text.lower()

        if "prefer the later state" in low or "prefer later state" in low or "latest valid state wins" in low:
            return NLRuleSpec(
                rule_id="rule-prefer-latest-state",
                family="prefer_latest_state",
                condition="later_state_exists_for_same_owner_and_item",
                effect="prefer_later_state_over_earlier_quantity",
                priority=100,
                metadata={"rule_text": text},
            )

        if "same owner" in low and "same item" in low and ("bind" in low or "same identity" in low):
            return NLRuleSpec(
                rule_id="rule-bind-owner-item",
                family="bind_same_entity",
                condition="same_owner_and_same_item_references_share_identity",
                effect="bind_references_to_same_entity_id",
                priority=60,
                metadata={"rule_text": text},
            )

        if ("remaining" in low or "discount" in low or "remainder" in low) and "then" in low:
            return NLRuleSpec(
                rule_id="rule-remaining-quantity",
                family="remaining_quantity",
                condition="remaining_quantity_is_base_times_one_minus_percent",
                effect="compute_remaining_quantity_from_base_and_reduction",
                priority=80,
                metadata={"rule_text": text},
            )

        if ("buy" in low or "donate" in low or "spoil" in low or "give" in low) and "state" in low:
            return NLRuleSpec(
                rule_id="rule-state-transition",
                family="state_transition",
                condition="state_transition_updates_same_owner_and_item",
                effect="apply_delta_to_before_quantity_and_recompute_after_quantity",
                priority=90,
                metadata={"rule_text": text},
            )

        return NLRuleSpec(
            rule_id="rule-generic",
            family="generic",
            condition="generic_user_rule",
            effect="store_rule_with_provenance_and_validation",
            priority=10,
            metadata={"rule_text": text},
        )

    @staticmethod
    def compile(raw_rule: str) -> Dict[str, Any]:
        spec = NLRuleCompiler.parse(raw_rule)
        payload = spec.to_payload()
        payload["validation"] = {
            "required_fields": ["rule_id", "family", "condition", "effect", "priority"],
            "accepted": True,
        }
        return payload

    @staticmethod
    def validate(payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
        required = {"rule_id", "family", "condition", "effect", "priority"}
        missing = sorted(required - set(payload.keys()))
        if missing:
            return False, [f"missing required fields: {missing}"]

        family = str(payload.get("family") or "")
        if family not in {"prefer_latest_state", "bind_same_entity", "remaining_quantity", "state_transition", "generic"}:
            return False, [f"unsupported rule family: {family}"]

        semantic_ok, semantic_issues = NLRuleCompiler.validate_semantics(payload)
        if not semantic_ok:
            return False, semantic_issues

        return True, []

    @staticmethod
    def validate_semantics(payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
        family = str(payload.get("family") or "")
        condition = str(payload.get("condition") or "")
        effect = str(payload.get("effect") or "")
        issues: List[str] = []

        expected = {
            "prefer_latest_state": (
                {"later_state", "owner", "item"},
                {"prefer_later_state", "earlier"},
            ),
            "bind_same_entity": (
                {"same_owner", "same_item"},
                {"bind", "entity"},
            ),
            "remaining_quantity": (
                {"remaining_quantity", "base", "percent"},
                {"compute_remaining_quantity", "reduction"},
            ),
            "state_transition": (
                {"state_transition", "owner", "item"},
                {"delta", "after_quantity"},
            ),
        }

        if family == "generic":
            return False, ["generic family is not auto-activatable; add a supported semantic family"]

        constraints = expected.get(family)
        if constraints is None:
            return False, [f"no semantic validator configured for family: {family}"]
        cond_tokens, eff_tokens = constraints
        lower_condition = condition.lower()
        lower_effect = effect.lower()

        missing_cond = sorted(tok for tok in cond_tokens if tok not in lower_condition)
        missing_eff = sorted(tok for tok in eff_tokens if tok not in lower_effect)
        if missing_cond:
            issues.append(f"condition missing semantic anchors: {missing_cond}")
        if missing_eff:
            issues.append(f"effect missing semantic anchors: {missing_eff}")

        priority = payload.get("priority")
        try:
            pr = int(priority)
            if pr <= 0:
                issues.append("priority must be positive")
        except (TypeError, ValueError):
            issues.append("priority must be an integer")

        return (len(issues) == 0), issues

    @staticmethod
    def load_rule_file(path: str | Path) -> List[str]:
        """Load a text file containing newline-delimited NL rules.

        Lines beginning with # are comments; blank lines are ignored.
        """
        file_path = Path(path)
        rules: List[str] = []
        if not file_path.exists():
            return rules
        for raw in file_path.read_text(encoding='utf-8').splitlines():
            text = raw.strip()
            if not text or text.startswith('#'):
                continue
            rules.append(text)
        return rules

    @staticmethod
    def compile_file(path: str | Path) -> List[Dict[str, Any]]:
        return [NLRuleCompiler.compile(rule) for rule in NLRuleCompiler.load_rule_file(path)]

    @staticmethod
    def inject_file(wm: Any, path: str | Path, source: str = 'file') -> List[Dict[str, Any]]:
        result = []
        for rule in NLRuleCompiler.load_rule_file(path):
            payload = NLRuleCompiler.compile(rule)
            ok, issues = NLRuleCompiler.validate(payload)
            if not ok:
                result.append({"accepted": False, "rule": rule, "issues": issues})
                continue
            injected = wm.inject_nl_rule(rule, source=source, rule_id=payload.get('rule_id'))
            result.append(injected)
        return result


if __name__ == '__main__':
    rules_path = Path(__file__).resolve().parents[1] / 'rules' / 'nl_rules.txt'
    print('Rules file:', rules_path)
    print('Compiled:', NLRuleCompiler.compile_file(rules_path))
