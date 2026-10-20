"""Canonical semantic schema for the GSM8K grounding layer.

This module defines the minimal graph contract shared by the LT proposer,
System 2 validation, and System 3 rule induction. The goal is to keep the
semantic interface small, explicit, and variable-bound so that graph fragments
can be checked and reduced reliably.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


class SemanticGraphSchema:
    """Small canonical interface for structured text-to-graph grounding.

    The graph is intentionally compact: each node describes a semantic role and
    each edge carries a dependency relation. Variable identity lives in the graph,
    not in incidental string names in the text.
    """

    NODE_KINDS = {
        "entity",
        "quantity",
        "rate",
        "ratio",
        "percent",
        "operator",
        "query",
        "variable",
        "dependency",
    }

    REQUIRED_BY_KIND = {
        "entity": {"id", "name", "kind"},
        "quantity": {"id", "entity_id", "value", "unit"},
        "rate": {"id", "entity_id", "value", "unit"},
        "ratio": {"id", "entity_id", "factor"},
        "percent": {"id", "entity_id", "pct"},
        "operator": {"id", "op", "lhs_var", "rhs_var", "result_var"},
        "query": {"id", "goal", "target_var"},
        "variable": {"id", "label", "value"},
        "dependency": {"source", "target", "relation"},
    }

    @classmethod
    def canonicalize_fact(cls, concept: str, **attrs: Any) -> Dict[str, Any]:
        """Normalize a fact-like payload to canonical field names.

        This keeps the schema stable even while upstream compilers emit legacy
        names such as arg0/arg1/arg2 or a mix of owner/item/qty.
        """
        canon = dict(attrs)

        # Normalize common legacy slots used across the repo.
        if "arg0" in canon and "owner" not in canon:
            canon["owner"] = canon.pop("arg0")
        if "arg1" in canon and "item" not in canon:
            canon["item"] = canon.pop("arg1")
        if "arg2" in canon and "qty" not in canon and "value" not in canon:
            canon["qty"] = canon.pop("arg2")
        if "arg2" in canon and "price" not in canon and concept.lower() == "sells":
            canon["price"] = canon.pop("arg2")

        # Ensure the canonical semantic keys for quantity-like facts exist.
        if concept in {"Quantity", "Has", "Produced", "Consumed", "Total", "Remaining"}:
            if "qty" in canon and "value" not in canon:
                canon["value"] = canon["qty"]

        if concept == "Rate":
            if "price_per_unit" in canon and "value" not in canon:
                canon["value"] = canon["price_per_unit"]

        if concept == "Percent":
            if "pct" in canon and "value" not in canon:
                canon["value"] = canon["pct"]

        return canon

    @classmethod
    def validate_graph_fragment(cls, payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a graph fragment with the canonical schema."""
        if not isinstance(payload, dict):
            return False, ["graph fragment must be a dict"]

        kind = str(payload.get("kind") or payload.get("node_kind") or "").lower()
        if kind not in cls.NODE_KINDS:
            return False, [f"unsupported graph node kind: {kind!r}"]

        missing = sorted(cls.REQUIRED_BY_KIND[kind] - set(str(k) for k in payload.keys()))
        if missing:
            return False, [f"missing required keys for {kind}: {missing}"]

        return True, []

    @classmethod
    def clamp_predicates(cls, fixed_predicates: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Create a constrained schema for LT or LLM output.

        fixed_predicates names the concrete slots the proposer must keep fixed;
        all other slots are open and may be filled by the model.
        """
        fixed = dict(fixed_predicates or {})
        open_slots = [
            "entity_name",
            "quantity_value",
            "quantity_unit",
            "rate_value",
            "ratio_factor",
            "percent_value",
            "operator",
            "dependency_relation",
        ]
        return {
            "fixed_predicates": fixed,
            "open_slots": open_slots,
            "schema": {
                "node_kinds": sorted(cls.NODE_KINDS),
                "required_by_kind": {k: sorted(v) for k, v in cls.REQUIRED_BY_KIND.items()},
            },
            "validator": "SemanticGraphSchema.validate_graph_fragment",
        }

    @classmethod
    def build_lm_template(cls, target_kind: str, fixed_predicates: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Build a structured output template for LT / LLM semantic graph proposals.

        The template keeps the schema explicit and the output machine-readable so
        that the downstream symbolic checker can verify the produced graph.
        """
        template = {
            "kind": target_kind,
            "id": "<stable-node-id>",
            "fixed": dict(fixed_predicates or {}),
            "open": {
                "value": "<value>",
                "entity_id": "<entity-id>",
                "unit": "<unit>",
                "factor": "<ratio-factor>",
                "pct": "<percentage>",
                "lhs_var": "<left-var>",
                "rhs_var": "<right-var>",
                "result_var": "<result-var>",
                "goal": "<query-goal>",
                "source": "<source-var>",
                "target": "<target-var>",
                "relation": "<dependency-rel>",
            },
        }
        return template

    @classmethod
    def make_quantity_binding(cls, variable_id: str, entity_id: str, value: Any, unit: str = "unit") -> Dict[str, Any]:
        return {
            "kind": "quantity",
            "id": variable_id,
            "entity_id": entity_id,
            "value": value,
            "unit": unit,
        }

    @classmethod
    def make_operator_binding(
        cls,
        op_id: str,
        op: str,
        lhs_var: str,
        rhs_var: str,
        result_var: str,
    ) -> Dict[str, Any]:
        return {
            "kind": "operator",
            "id": op_id,
            "op": op,
            "lhs_var": lhs_var,
            "rhs_var": rhs_var,
            "result_var": result_var,
        }

    @classmethod
    def dependency(cls, source: str, target: str, relation: str, **attrs: Any) -> Dict[str, Any]:
        payload = {
            "kind": "dependency",
            "source": source,
            "target": target,
            "relation": relation,
        }
        payload.update(attrs)
        return payload

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            import re
            match = re.search(r"-?\d+(?:\.\d+)?", value)
            if match:
                return float(match.group(0))
        return None

    @classmethod
    def reduce_operator(cls, fragment: Dict[str, Any], context: Dict[str, Any] | None = None) -> float | None:
        context = context or {}
        op = str(fragment.get("op") or "").lower()
        lhs_key = fragment.get("lhs_var")
        rhs_key = fragment.get("rhs_var")
        if lhs_key is None or rhs_key is None:
            return None

        lhs = context.get(lhs_key)
        rhs = context.get(rhs_key)
        if lhs is None or rhs is None:
            return None

        lhs_val = cls._to_float(lhs)
        rhs_val = cls._to_float(rhs)
        if lhs_val is None or rhs_val is None:
            return None

        if op == "multiply":
            return lhs_val * rhs_val
        if op == "add":
            return lhs_val + rhs_val
        if op == "subtract":
            return lhs_val - rhs_val
        if op == "divide":
            return lhs_val / rhs_val if abs(rhs_val) > 1e-12 else None
        if op == "ratio":
            return lhs_val * rhs_val
        if op == "percent":
            return lhs_val * rhs_val
        return None
