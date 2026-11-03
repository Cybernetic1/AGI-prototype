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
        "state",
        "state_transition",
        "context",
        "proposition",
        "in_context",
        "accessibility",
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
        "state": {"id", "owner", "item", "qty", "phase"},
        "state_transition": {"id", "owner", "item", "before_qty", "delta", "after_qty", "event_kind", "event_order"},
        "context": {"id", "ctx_type"},
        "proposition": {"id", "pred", "args"},
        "in_context": {"prop_id", "ctx_id"},
        "accessibility": {"from_ctx", "to_ctx", "modality"},
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
    def make_state(
        cls,
        state_id: str,
        owner: str,
        item: str,
        qty: Any,
        phase: str,
        **attrs: Any,
    ) -> Dict[str, Any]:
        payload = {
            "kind": "state",
            "id": state_id,
            "owner": owner,
            "item": item,
            "qty": qty,
            "phase": phase,
        }
        payload.update(attrs)
        return payload

    @classmethod
    def make_state_transition(
        cls,
        transition_id: str,
        owner: str,
        item: str,
        before_qty: Any,
        delta: Any,
        after_qty: Any,
        event_kind: str,
        event_order: int,
        **attrs: Any,
    ) -> Dict[str, Any]:
        payload = {
            "kind": "state_transition",
            "id": transition_id,
            "owner": owner,
            "item": item,
            "before_qty": before_qty,
            "delta": delta,
            "after_qty": after_qty,
            "event_kind": event_kind,
            "event_order": event_order,
        }
        payload.update(attrs)
        return payload

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

    @classmethod
    def make_context(
        cls,
        ctx_id: str,
        ctx_type: str,
        *,
        parent_ctx: str | None = None,
        agent: str | None = None,
        t: Any = None,
        assumptions: Any = None,
        status: str = "active",
        **attrs: Any,
    ) -> Dict[str, Any]:
        payload = {
            "kind": "context",
            "id": ctx_id,
            "ctx_type": ctx_type,
            "status": status,
        }
        if parent_ctx is not None:
            payload["parent_ctx"] = parent_ctx
        if agent is not None:
            payload["agent"] = agent
        if t is not None:
            payload["t"] = t
        if assumptions is not None:
            payload["assumptions"] = assumptions
        payload.update(attrs)
        return payload

    @classmethod
    def make_proposition(
        cls,
        prop_id: str,
        pred: str,
        args: Iterable[Any],
        *,
        polarity: str = "pos",
        confidence: float = 1.0,
        **attrs: Any,
    ) -> Dict[str, Any]:
        payload = {
            "kind": "proposition",
            "id": prop_id,
            "pred": pred,
            "args": list(args),
            "polarity": polarity,
            "confidence": confidence,
        }
        payload.update(attrs)
        return payload

    @classmethod
    def make_in_context(
        cls,
        prop_id: str,
        ctx_id: str,
        *,
        truth_status: str = "asserted",
        support: Any = None,
        **attrs: Any,
    ) -> Dict[str, Any]:
        payload = {
            "kind": "in_context",
            "prop_id": prop_id,
            "ctx_id": ctx_id,
            "truth_status": truth_status,
        }
        if support is not None:
            payload["support"] = support
        payload.update(attrs)
        return payload

    @classmethod
    def make_accessibility(
        cls,
        from_ctx: str,
        to_ctx: str,
        modality: str,
        *,
        guard: str | None = None,
        **attrs: Any,
    ) -> Dict[str, Any]:
        payload = {
            "kind": "accessibility",
            "from_ctx": from_ctx,
            "to_ctx": to_ctx,
            "modality": modality,
        }
        if guard:
            payload["guard"] = guard
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

    @classmethod
    def make_semantic_graph(cls, nodes: Iterable[Dict[str, Any]] | None = None, edges: Iterable[Dict[str, Any]] | None = None) -> Dict[str, Any]:
        """Construct the canonical semantic graph used by the graph-reasoning layer.

        The representation is intentionally small and generic: nodes have a {
        kind, id, ... } payload, and edges are generic labeled relations. This
        keeps representation separate from inference, so the same graph can be
        queried by a simple backward-chaining walker without ad hoc predicates.
        """
        graph = {"nodes": {}, "edges": []}
        for node in nodes or []:
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("id") or node.get("node_id") or node.get("name") or "")
            if not node_id:
                continue
            graph["nodes"][node_id] = dict(node)
        for edge in edges or []:
            if not isinstance(edge, dict):
                continue
            if "source" in edge and "target" in edge and "relation" in edge:
                graph["edges"].append(dict(edge))
        return graph

    @classmethod
    def graph_from_fact_rows(cls, rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        """Canonicalize rows like {source, target, relation, value} into the graph view."""
        nodes: Dict[str, Dict[str, Any]] = {}
        edges: List[Dict[str, Any]] = []

        for row in rows or []:
            if not isinstance(row, dict):
                continue
            payload = dict(row)
            source = payload.get("source") or payload.get("subject") or payload.get("entity_id")
            target = payload.get("target") or payload.get("object") or payload.get("entity")
            relation = payload.get("relation") or payload.get("pred") or payload.get("kind") or "depends_on"
            if source is not None and target is not None and relation is not None:
                edges.append({"source": str(source), "target": str(target), "relation": str(relation)})
                nodes.setdefault(str(source), {"id": str(source), "kind": "entity"})
                nodes.setdefault(str(target), {"id": str(target), "kind": "entity"})
                if "value" in payload:
                    value_id = str(payload.get("value_id") or f"value:{source}:{target}:{relation}")
                    nodes[value_id] = {
                        "id": value_id,
                        "kind": "quantity",
                        "entity_id": str(source),
                        "value": payload.get("value"),
                        "unit": payload.get("unit", "unit"),
                    }
                    edges.append({"source": str(source), "target": value_id, "relation": "has_value"})

            if "id" in payload and "kind" in payload:
                nodes[str(payload["id"])] = {"id": str(payload["id"]), "kind": str(payload["kind"]), **{k: v for k, v in payload.items() if k not in {"id", "kind"}}}

        return {"nodes": nodes, "edges": edges}

    @classmethod
    def _node_matches(cls, node: Dict[str, Any], constraint: Dict[str, Any] | None) -> bool:
        if not constraint:
            return True
        for key, expected in constraint.items():
            if node.get(key) != expected:
                return False
        return True

    @classmethod
    def backward_chain(
        cls,
        graph: Dict[str, Any],
        *,
        target: str | None = None,
        relation: str | None = None,
        node_constraint: Dict[str, Any] | None = None,
        max_depth: int = 3,
    ) -> List[Dict[str, Any]]:
        """Retrieve supporting graph paths for a target by walking incoming/outgoing edges.

        This is intentionally a simple backward-chaining interface: the graph stores
        generic labeled relations and the query asks for all paths that explain a
        target or relation, without requiring ad hoc domain-specific predicates.
        """
        if not isinstance(graph, dict):
            return []
        nodes = graph.get("nodes", {}) if isinstance(graph.get("nodes"), dict) else {}
        edges = graph.get("edges", []) if isinstance(graph.get("edges"), list) else []
        if not target and not relation and not node_constraint:
            return []

        matches: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str, str]] = set()

        def walk(current: str, depth: int, via: str | None = None):
            if depth > max_depth:
                return
            for edge in edges:
                if not isinstance(edge, dict):
                    continue
                src = str(edge.get("source") or "")
                dst = str(edge.get("target") or "")
                rel = str(edge.get("relation") or "")
                if relation and rel != relation:
                    continue
                if src == current:
                    candidate = {"source": src, "target": dst, "relation": rel, **{k: v for k, v in edge.items() if k not in {"source", "target", "relation"}}}
                    if (src, dst, rel) not in seen:
                        seen.add((src, dst, rel))
                        matches.append(candidate)
                    walk(dst, depth + 1, rel)
                elif dst == current:
                    candidate = {"source": src, "target": dst, "relation": rel, **{k: v for k, v in edge.items() if k not in {"source", "target", "relation"}}}
                    if (src, dst, rel) not in seen:
                        seen.add((src, dst, rel))
                        matches.append(candidate)
                    walk(src, depth + 1, rel)

        if target is not None:
            target_id = str(target)
            node = nodes.get(target_id)
            if node and cls._node_matches(node, node_constraint):
                walk(target_id, 0)
            else:
                for node_id, node_payload in nodes.items():
                    if node_id == target_id or str(node_payload.get("name")) == target_id:
                        walk(node_id, 0)
        elif node_constraint:
            for node_id, node_payload in nodes.items():
                if cls._node_matches(node_payload, node_constraint):
                    walk(node_id, 0)

        return matches
