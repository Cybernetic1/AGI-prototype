from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Sequence


@dataclass(frozen=True)
class PredicateFact:
    """A unary/binary predicate fact with a canonical, LT-friendly representation."""

    predicate: str
    args: tuple[Any, ...]

    def __str__(self) -> str:
        if not self.args:
            return f"{self.predicate}()"
        rendered = ", ".join(str(a) for a in self.args)
        return f"{self.predicate}({rendered})"


def normalize_predicate_name(name: str) -> str:
    cleaned = str(name).strip()
    if not cleaned:
        return "fact"
    cleaned = cleaned.replace("-", "_")
    cleaned = cleaned.replace(".", "_")
    if cleaned[0].isdigit():
        cleaned = f"p_{cleaned}"
    return cleaned.lower()


def _coerce_arg(value: Any) -> Any:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return value
    return str(value)


def _event_id_for(node_concept: str, attrs: Mapping[str, Any]) -> str:
    event_id = attrs.get("id")
    if event_id is not None:
        return str(event_id)
    owner = attrs.get("owner")
    item = attrs.get("item")
    if owner is not None and item is not None:
        return f"{normalize_predicate_name(node_concept)}_{owner}_{item}"
    return f"{normalize_predicate_name(node_concept)}_e"


def wmnode_to_predicate_atoms(node: Any) -> List[PredicateFact]:
    """Convert a legacy mixed-format WMNode into unary/binary predicate atoms.

    This is the compatibility layer between the earlier record-like WMEs and the
    newer canonical logic representation. The target representation keeps all
    predicates arity <= 2 and reifies events/states as IDs.
    """
    if node is None:
        return []

    concept = getattr(node, "concept", None)
    if concept is None:
        return []

    attrs = getattr(node, "to_dict", lambda: {})()
    if not isinstance(attrs, Mapping):
        attrs = {}

    if not attrs:
        return [PredicateFact(normalize_predicate_name(concept), tuple())]

    event_id = _event_id_for(str(concept), attrs)
    atoms: List[PredicateFact] = []
    concept_pred = normalize_predicate_name(concept)
    atoms.append(PredicateFact(concept_pred, (event_id,)))

    # Emit the canonical binary relations that make the object usable in a
    # classical predicate-logic rule system. We intentionally avoid long n-ary
    # predicates such as Has(owner, item, qty), and instead reify the event/state.
    for key, value in attrs.items():
        canon_key = normalize_predicate_name(key)
        if key in {"id", "provenance_id", "activation", "layer", "source", "phase", "kind"}:
            continue
        if key in {"owner", "item", "goal", "label", "status", "reason", "event_kind", "unit", "source_id", "target_id", "relation"}:
            atoms.append(PredicateFact(canon_key, (event_id, _coerce_arg(value))))
            continue
        if key in {"qty", "value", "before_qty", "after_qty", "delta", "factor", "pct", "price", "price_per_unit", "before", "after", "result", "total", "remaining", "event_order"}:
            atoms.append(PredicateFact(canon_key, (event_id, _coerce_arg(value))))
            continue
        atoms.append(PredicateFact(canon_key, (event_id, _coerce_arg(value))))

    return atoms


def wm_to_predicate_atoms(wm: Any) -> List[PredicateFact]:
    """Flatten a whole working memory into a canonical predicate-logic fact set."""
    atoms: List[PredicateFact] = []
    facts = getattr(wm, "facts", {}).keys()
    for node in facts:
        atoms.extend(wmnode_to_predicate_atoms(node))
    return atoms


def canonical_rule_template_family(name: str, **kwargs: Any) -> dict:
    """Return a compact rule template in a learnable DSL shape."""
    template = {
        "name": name,
        "arity_limit": 2,
        "predicates": [],
        "parameters": dict(kwargs),
    }
    return template
