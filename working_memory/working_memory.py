import re
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Tuple

from working_memory.semantic_schema import SemanticGraphSchema

# ==========================================
# Semantic graph support for equation/dependency tracing.
# These nodes are intentionally explicit so that an entity, a quantity, and an
# operator can share the same variable identity without collapsing into a flat
# tuple of facts. This mirrors a lightweight dependency graph / TMS substrate.
# ==========================================
@dataclass(frozen=True)
class GraphNode:
    node_id: str
    kind: str
    label: str
    attrs: frozenset = frozenset()

@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    relation: str
    attrs: frozenset = frozenset()

# ==========================================
# GWT Unified Fact Representation
# ==========================================
@dataclass(frozen=True)
class WMNode:
    concept: str
    attrs: frozenset
    activation: float = 1.0  # Continuous salience / confidence score
    layer: str = "reasoning"  # parse | semantic | reasoning | meta
    source: str = "unknown"  # The subsystem that produced the fact
    provenance_id: Optional[str] = None

    @classmethod
    def create(cls, concept: str, activation: float = 1.0, layer: str = "reasoning", source: str = "unknown", provenance_id: Optional[str] = None, **kwargs):
        # Filter out built-in fields if passed in kwargs
        kwargs.pop("activation", None)
        kwargs.pop("layer", None)
        kwargs.pop("source", None)
        kwargs.pop("provenance_id", None)
        return cls(concept, frozenset(sorted(kwargs.items())), activation, layer, source, provenance_id)
    
    def to_dict(self):
        return dict(self.attrs)

    def __eq__(self, other):
        if not isinstance(other, WMNode):
            return False
        # Structural equality ignores activation, but keeps provenance explicit so
        # we do not merge parse/semantic/reasoning facts with different origins.
        return (
            self.concept == other.concept
            and self.attrs == other.attrs
            and self.layer == other.layer
            and self.source == other.source
            and self.provenance_id == other.provenance_id
        )

    def __hash__(self):
        return hash((self.concept, self.attrs, self.layer, self.source, self.provenance_id))

# ==========================================
# Working Memory (Global Workspace)
# ==========================================
class WorkingMemory:
    def __init__(self, default_ttl=3):
        # Dictionary mapping: WMNode -> remaining cycles (TTL)
        self.facts: Dict[WMNode, int] = {}
        self.default_ttl = default_ttl
        self.graph_nodes: Dict[str, GraphNode] = {}
        self.graph_edges: List[GraphEdge] = []

    def add_equation_graph(self, *, op_id: str, op_kind: str, label: str, inputs: Optional[List[Tuple[str, Any]]] = None, outputs: Optional[List[Tuple[str, Any]]] = None, **attrs):
        """Create explicit graph nodes/edges for a symbolic equation or semantic dependency.

        Each graph node is identified by a stable string ID, and edges connect shared
        variables across facts. This preserves provenance and lets the system reason over
        an explicit dependency graph rather than only arbitrary WM fact tuples.
        """
        inputs = inputs or []
        outputs = outputs or []
        if outputs and len(outputs) == 1:
            primary_name, primary_value = outputs[0]
            attrs = dict(attrs)
            attrs.setdefault("value", primary_value)
            attrs.setdefault("result", primary_value)
            attrs.setdefault(primary_name, primary_value)
        operation_node = GraphNode(
            node_id=op_id,
            kind=op_kind,
            label=label,
            attrs=frozenset(sorted(attrs.items()))
        )
        self.graph_nodes[op_id] = operation_node
        self.facts[WMNode.create("GraphNode", layer="semantic", source="equation_graph", node_id=op_id, kind=op_kind, label=label, **attrs)] = self.default_ttl

        for var_name, value in inputs:
            var_id = f"var:{var_name}"
            self.graph_nodes.setdefault(var_id, GraphNode(var_id, "variable", str(var_name), frozenset({('value', value)})))
            self.graph_edges.append(GraphEdge(var_id, op_id, "input", frozenset({('var', var_name)})))
            self.facts[WMNode.create("GraphEdge", layer="semantic", source="equation_graph", source_id=var_id, target_id=op_id, relation="input", var=var_name, value=value)] = self.default_ttl

        for var_name, value in outputs:
            var_id = f"var:{var_name}"
            self.graph_nodes.setdefault(var_id, GraphNode(var_id, "variable", str(var_name), frozenset({('value', value)})))
            self.graph_edges.append(GraphEdge(op_id, var_id, "output", frozenset({('var', var_name)})))
            self.facts[WMNode.create("GraphEdge", layer="semantic", source="equation_graph", source_id=op_id, target_id=var_id, relation="output", var=var_name, value=value)] = self.default_ttl

    def add_state(self, *, state_id: str, owner: str, item: str, qty: Any, phase: str = "state", **attrs):
        """Persist a discrete quantity state in the graph so later events can update it explicitly."""
        payload = SemanticGraphSchema.make_state(state_id, owner, item, qty, phase, **attrs)
        ok, issues = SemanticGraphSchema.validate_graph_fragment(payload)
        if not ok:
            return
        self.graph_nodes[state_id] = GraphNode(
            node_id=state_id,
            kind="state",
            label=f"{phase}:{owner}:{item}",
            attrs=frozenset(sorted(payload.items())),
        )
        self.facts[WMNode.create("State", layer="semantic", source="state_machine", **payload)] = self.default_ttl

    def add_state_transition(
        self,
        *,
        transition_id: str,
        owner: str,
        item: str,
        before_qty: Any,
        delta: Any,
        after_qty: Any,
        event_kind: str,
        event_order: int,
        source_state_id: Optional[str] = None,
        target_state_id: Optional[str] = None,
        **attrs,
    ):
        """Record a single quantity event that transforms one state into the next."""
        payload = SemanticGraphSchema.make_state_transition(
            transition_id,
            owner,
            item,
            before_qty,
            delta,
            after_qty,
            event_kind,
            event_order,
            **attrs,
        )
        ok, issues = SemanticGraphSchema.validate_graph_fragment(payload)
        if not ok:
            return
        self.graph_nodes[transition_id] = GraphNode(
            node_id=transition_id,
            kind="state_transition",
            label=f"{event_kind}:{owner}:{item}",
            attrs=frozenset(sorted(payload.items())),
        )
        self.facts[WMNode.create("StateTransition", layer="semantic", source="state_machine", **payload)] = self.default_ttl

        if source_state_id:
            self.graph_edges.append(GraphEdge(source_state_id, transition_id, "precedes", frozenset({('event_kind', event_kind)})))
        if target_state_id:
            self.graph_edges.append(GraphEdge(transition_id, target_state_id, "updates", frozenset({('event_kind', event_kind)})))

    def update(self, new_facts):
        if not new_facts:
            return

        for fact in new_facts:
            d = fact.to_dict()
            owner = d.get("owner")
            item = d.get("item")
            args = d.get("args") or ()
            goal = d.get("goal")

            # 1. Keep only the newest state for a given owner/item and prefer the final
            #    query candidate for an action chain (e.g. give-away -> percent -> buy-more).
            #    This suppresses stale earlier values that otherwise compete with the real final state.
            if fact.concept in {"Inventory", "Quantity", "Has", "Remaining", "Total", "Produced", "Consumed", "State"}:
                to_remove = [
                    f for f in self.facts
                    if f.concept == fact.concept
                    and f.to_dict().get("owner") == owner
                    and f.to_dict().get("item") == item
                ]
                for f in to_remove:
                    del self.facts[f]
            elif fact.concept == "Query":
                to_remove = [
                    f for f in self.facts
                    if f.concept == "Query"
                    and f.to_dict().get("goal") == goal
                    and len(args) >= 2
                    and len((f.to_dict().get("args") or ())) >= 2
                    and (f.to_dict().get("args") or ())[0] == args[0]
                    and (f.to_dict().get("args") or ())[1] == args[1]
                ]
                for f in to_remove:
                    del self.facts[f]

            # 2. Competitive Selection (Salience/Activation check)
            # NOTE: We keep layer/source/provenance in the node identity so we can
            # preserve System 1 parse/semantic nodes separately from higher-confidence
            # reasoning nodes, avoiding credit assignment loss when soft and hard facts
            # are structurally similar.
            existing_match = next((f for f in self.facts if f == fact), None)
            if existing_match is not None:
                # Keep the one with HIGHER activation/confidence
                if fact.activation >= existing_match.activation:
                    del self.facts[existing_match]
                    self.facts[fact] = self.default_ttl
                else:
                    self.facts[existing_match] = self.default_ttl
            else:
                self.facts[fact] = self.default_ttl

    def facts_by_layer(self, layer):
        return {fact: ttl for fact, ttl in self.facts.items() if fact.layer == layer}

    def validate_equation_graph(self):
        """Validate each graph fragment against the canonical semantic schema.

        This is the Sys-2 validation pass: malformed or under-specified graph nodes
        are rejected before they participate in reasoning. For operator nodes, the
        reducer also tries to compute an expected numeric result from the linked
        input variables whenever all information is available.
        """
        context = {}
        for node in list(self.graph_nodes.values()):
            attrs = dict(node.attrs)
            payload = {"kind": node.kind, "id": node.node_id, **attrs}
            ok, issues = SemanticGraphSchema.validate_graph_fragment(payload)
            if not ok:
                self.graph_nodes.pop(node.node_id, None)
                continue
            for key in ("value", "result", "answer", "total", "distance", "amount", "qty", "price", "factor", "pct", "hours", "remaining", "final_price"):
                val = attrs.get(key)
                if val is not None:
                    context[str(node.node_id)] = val
                    context[str(key)] = val

        for node in list(self.graph_nodes.values()):
            attrs = dict(node.attrs)
            if node.kind.lower() == "operator":
                lhs_key = attrs.get("lhs_var")
                rhs_key = attrs.get("rhs_var")
                res_key = attrs.get("result_var")
                if lhs_key and rhs_key and res_key:
                    reduced = SemanticGraphSchema.reduce_operator({
                        "op": attrs.get("op"),
                        "lhs_var": lhs_key,
                        "rhs_var": rhs_key,
                        "result_var": res_key,
                    }, context)
                    if reduced is not None and res_key in context and abs(float(context.get(res_key, 0.0)) - float(reduced)) > 1e-9:
                        context[res_key] = reduced

        return True

    def decay(self):
        expired = []
        for fact in self.facts:
            self.facts[fact] -= 1
            if self.facts[fact] <= 0:
                expired.append(fact)
        for fact in expired:
            del self.facts[fact]


class EquationGraphSolver:
    """General dependency-graph evaluator for arithmetic and semantic transformations.

    The graph is intentionally generic: it supports both arithmetic/micro-step nodes and
    broader story-level dependency relations. The solver simply propagates known values until
    the graph reaches a fixed point; this is a lightweight TMS-like semantic reduction layer
    rather than a heavyweight theorem prover.
    """

    def __init__(self, graph_nodes=None, graph_edges=None):
        self.graph_nodes = graph_nodes or {}
        self.graph_edges = graph_edges or []

    @classmethod
    def from_wm(cls, current_wm):
        node_map = {}
        for node in current_wm:
            if node.concept != "GraphNode":
                continue
            attrs = node.to_dict()
            node_id = attrs.get("node_id")
            if not node_id:
                continue
            node_map[node_id] = GraphNode(
                node_id=node_id,
                kind=str(attrs.get("kind", "unknown")),
                label=str(attrs.get("label", node_id)),
                attrs=frozenset((k, v) for k, v in sorted(attrs.items()) if k not in {"node_id", "kind", "label"})
            )

        edges = []
        for node in current_wm:
            if node.concept != "GraphEdge":
                continue
            attrs = node.to_dict()
            src = attrs.get("source_id")
            tgt = attrs.get("target_id")
            rel = attrs.get("relation")
            if src is None or tgt is None:
                continue
            edges.append(GraphEdge(
                source=str(src),
                target=str(tgt),
                relation=str(rel),
                attrs=frozenset((k, v) for k, v in sorted(attrs.items()) if k not in {"source_id", "target_id", "relation"})
            ))
        return cls(node_map, edges)

    @staticmethod
    def _to_float(value):
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            match = re.search(r"-?\d+(?:\.\d+)?", value)
            if match:
                return float(match.group(0))
        return None

    def _known_value(self, node_id, known):
        if node_id in known:
            return known[node_id]
        node = self.graph_nodes.get(node_id)
        if not node:
            return None
        attrs = dict(node.attrs)
        for key in ("value", "result", "answer", "total", "distance", "amount", "qty", "price", "factor", "pct", "hours", "remaining", "final_price"):
            val = self._to_float(attrs.get(key))
            if val is not None:
                return val
        return None

    def _evaluate(self, kind, values):
        if not values:
            return None
        kind = str(kind).lower()
        if kind in {"multiply", "ratio", "percent", "product"}:
            if len(values) == 2:
                return float(values[0]) * float(values[1])
            if len(values) == 1:
                return float(values[0])
        if kind in {"add", "sum"}:
            return sum(float(v) for v in values)
        if kind == "subtract":
            if len(values) == 2:
                return float(values[0]) - float(values[1])
            if len(values) == 1:
                return float(values[0])
        if kind == "divide":
            if len(values) == 2:
                if abs(float(values[1])) < 1e-12:
                    return None
                return float(values[0]) / float(values[1])
            if len(values) == 1:
                return float(values[0])
        if kind in {"unitconvert", "unit_convert", "convert"}:
            if len(values) == 1:
                return float(values[0])
            if len(values) >= 2:
                return float(values[0])
        if kind in {"query", "value", "variable", "entity"}:
            return values[0] if len(values) == 1 else None
        return None

    def solve(self, goal=None):
        known = {}
        for node in self.graph_nodes.values():
            attrs = dict(node.attrs)
            for key in ("value", "result", "answer", "total", "distance", "amount", "qty", "price", "factor", "pct", "hours", "remaining", "final_price"):
                val = self._to_float(attrs.get(key))
                if val is not None:
                    known[node.node_id] = val

        # If a direct query node exists, read its output value when available.
        if goal is not None:
            for node in self.graph_nodes.values():
                attrs = dict(node.attrs)
                if node.kind == "query":
                    q_goal = attrs.get("goal") or attrs.get("label")
                    if q_goal == goal and attrs.get("value") is not None:
                        return self._to_float(attrs.get("value"))

        changed = True
        iterations = 0
        while changed and iterations < 50:
            iterations += 1
            changed = False
            for node in self.graph_nodes.values():
                kind = node.kind.lower()
                if kind not in {"multiply", "add", "subtract", "divide", "ratio", "percent", "unitconvert", "unit_convert", "query", "value", "variable"}:
                    continue

                inputs = [edge.source for edge in self.graph_edges if edge.target == node.node_id and edge.relation == "input"]
                if not inputs:
                    continue

                values = []
                for src in inputs:
                    val = self._known_value(src, known)
                    if val is None:
                        break
                    values.append(val)
                if len(values) != len(inputs):
                    continue

                result = self._evaluate(kind, values)
                if result is None:
                    continue

                for edge in self.graph_edges:
                    if edge.source == node.node_id and edge.relation == "output":
                        out_id = edge.target
                        if out_id not in known or abs(float(known[out_id]) - float(result)) > 1e-9:
                            known[out_id] = float(result)
                            changed = True

        if goal is not None:
            for node in self.graph_nodes.values():
                attrs = dict(node.attrs)
                if node.kind == "query":
                    q_goal = attrs.get("goal") or attrs.get("label")
                    if q_goal == goal:
                        for key in ("value", "result", "answer", "total", "distance", "amount"):
                            val = self._to_float(attrs.get(key))
                            if val is not None:
                                return val
                        for edge in self.graph_edges:
                            if edge.source == node.node_id and edge.relation == "output":
                                out_id = edge.target
                                if out_id in known:
                                    return known[out_id]
            for node_id, value in known.items():
                if str(node_id).endswith(str(goal)) or str(goal).lower() in str(node_id).lower():
                    return value

        return None
