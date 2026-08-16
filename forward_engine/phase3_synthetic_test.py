from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from ontology import (
    ComparisonEvent,
    CreationEvent,
    LossEvent,
    NeutralEvent,
    TransferEvent,
    resolve_event_class,
)


@dataclass(frozen=True)
class Example:
    text: str
    verb: str
    subject: str
    obj: str
    qty: int
    expected: str


TRAIN_VERB_CLASSES = {
    "transfer": {"give", "hand", "pass", "send"},
    "creation": {"buy", "find", "pick", "bake", "create", "make", "get", "receive"},
    "loss": {"eat", "lose", "break", "consume", "drop", "spend", "discard"},
    "comparison": {"have", "double", "twice", "triple", "half"},
}

EXAMPLES: List[Example] = [
    Example("John donated 3 apples to Mary.", "donated", "john", "mary", 3, "transfer"),
    Example("John gifted 2 apples to Mary.", "gifted", "john", "mary", 2, "transfer"),
    Example("Mary purchased 4 apples.", "purchased", "mary", "", 4, "creation"),
    Example("Mary acquired 1 apple.", "acquired", "mary", "", 1, "creation"),
    Example("John shattered 2 apples.", "shattered", "john", "", 2, "loss"),
    Example("John destroyed 1 apple.", "destroyed", "john", "", 1, "loss"),
    Example("Mary has twice as many apples as John.", "twice", "mary", "john", 2, "comparison"),
    Example("Mary has half as many apples as John.", "half", "mary", "john", 2, "comparison"),
]


def baseline_classify(verb: str) -> str:
    lemma = verb.lower()
    for label, verbs in TRAIN_VERB_CLASSES.items():
        if lemma in verbs:
            return label
    return "neutral"


def ontology_classify(verb: str) -> str:
    cls = resolve_event_class(verb)
    if issubclass(cls, TransferEvent):
        return "transfer"
    if issubclass(cls, CreationEvent):
        return "creation"
    if issubclass(cls, LossEvent):
        return "loss"
    if issubclass(cls, ComparisonEvent):
        return "comparison"
    if issubclass(cls, NeutralEvent):
        return "neutral"
    return "neutral"


def apply_event(state: Dict[str, int], ex: Example, predicted: str) -> Dict[str, int]:
    next_state = dict(state)
    if predicted == "transfer":
        next_state[ex.subject] = next_state.get(ex.subject, 0) - ex.qty
        next_state[ex.obj] = next_state.get(ex.obj, 0) + ex.qty
    elif predicted == "creation":
        next_state[ex.subject] = next_state.get(ex.subject, 0) + ex.qty
    elif predicted == "loss":
        next_state[ex.subject] = next_state.get(ex.subject, 0) - ex.qty
    return next_state


def evaluate():
    baseline_hits = 0
    ontology_hits = 0
    baseline_state_hits = 0
    ontology_state_hits = 0
    total = len(EXAMPLES)
    class_rows: List[Tuple[str, str, str]] = []

    for ex in EXAMPLES:
        baseline = baseline_classify(ex.verb)
        ontology = ontology_classify(ex.verb)
        class_rows.append((ex.text, baseline, ontology))

        if baseline == ex.expected:
            baseline_hits += 1
        if ontology == ex.expected:
            ontology_hits += 1

        if ex.expected == "comparison":
            continue

        init_state = {ex.subject: 10, ex.obj: 0} if ex.obj else {ex.subject: 10}
        gold_state = apply_event(init_state, ex, ex.expected)
        baseline_state = apply_event(init_state, ex, baseline)
        ontology_state = apply_event(init_state, ex, ontology)
        baseline_state_hits += int(baseline_state == gold_state)
        ontology_state_hits += int(ontology_state == gold_state)

    state_total = sum(1 for ex in EXAMPLES if ex.expected != "comparison")
    print("Phase 3 synthetic ontology test")
    print(f"Examples: {total}")
    print(f"Class accuracy | baseline={baseline_hits / total:.3f} ontology={ontology_hits / total:.3f}")
    print(f"State accuracy | baseline={baseline_state_hits / max(1, state_total):.3f} ontology={ontology_state_hits / max(1, state_total):.3f}")
    print("\nPer-example:")
    for text, baseline, ontology in class_rows:
        print(f"- {text}")
        print(f"  baseline={baseline} ontology={ontology}")


if __name__ == "__main__":
    evaluate()
