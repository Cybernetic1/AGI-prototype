from __future__ import annotations

import collections
import collections.abc
import re
from functools import lru_cache
from typing import Dict, List, Tuple, Type

collections.Mapping = collections.abc.Mapping

from experta import Fact


class Event(Fact):
    pass


class TransferEvent(Event):
    pass


class CreationEvent(Event):
    pass


class LossEvent(Event):
    pass


class ComparisonEvent(Event):
    pass


# -------------------------------------------------------------------
# Lightweight GSM8K reasoning vocabulary used by the symbolic WM layer.
# These are intentionally compact but canonical, so the compiler can map
# text to a stable fact schema instead of ad hoc one-off predicates.
# -------------------------------------------------------------------
class Quantity(Fact):
    pass


class Item(Fact):
    pass


class Rate(Fact):
    pass


class Price(Fact):
    pass


class Has(Fact):
    pass


class Total(Fact):
    pass


class Remaining(Fact):
    pass


class NeutralEvent(Event):
    pass


_VERB_ALIASES: Dict[str, str] = {
    "gave": "give",
    "given": "give",
    "gives": "give",
    "handed": "hand",
    "passes": "pass",
    "sent": "send",
    "bought": "buy",
    "got": "get",
    "gets": "get",
    "received": "receive",
    "took": "take",
    "taken": "take",
    "picked": "pick",
    "baked": "bake",
    "created": "create",
    "made": "make",
    "found": "find",
    "lost": "lose",
    "loses": "lose",
    "broke": "break",
    "broken": "break",
    "consumed": "consume",
    "ate": "eat",
    "eaten": "eat",
    "dropped": "drop",
    "spent": "spend",
    "have": "have",
    "has": "have",
    "had": "have",
    "double": "double",
    "twice": "twice",
    "triple": "triple",
    "half": "half",
    "is": "be",
    "are": "be",
    "was": "be",
    "were": "be",
    "am": "be",
    "left": "leave",
    "needs": "need",
    "needed": "need",
    "does": "do",
    "did": "do",
    "done": "do",
    "went": "go",
    "goes": "go",
    "costs": "cost",
    "worked": "work",
    "drives": "drive",
    "drove": "drive",
    "lasted": "last",
    "kept": "keep",
    "holds": "hold",
    "contains": "contain",
    "contained": "contain",
    "weighed": "weigh",
    "averaged": "average",
    "borrowed": "borrow",
    "walked": "walk",
    "transported": "transport",
    "traveled": "travel",
    "wrote": "write",
    "brought": "bring",
    "carried": "carry",
    "donated": "donate",
    "gifted": "gift",
    "purchased": "purchase",
    "acquired": "acquire",
    "shattered": "shatter",
    "destroyed": "destroy",
}

_TRANSFER_VERBS = {
    "give", "hand", "pass", "send", "transfer", "deliver", "offer", "donate", "gift", "convey",
    # economic/transactional verbs often seen in GSM8K
    "pay", "charge", "sell", "receive", "grant", "provide", "transfer", "offer",
}
_CREATION_VERBS = {
    "buy", "find", "pick", "bake", "create", "make", "get", "receive", "acquire", "gain", "obtain", "purchase", "collect", "procure",
    # earning / saving / gaining resources
    "earn", "save", "gain",
}
_LOSS_VERBS = {
    "eat", "lose", "break", "consume", "drop", "spend", "sell", "discard", "remove", "throw", "shed", "destroy", "damage", "shatter", "ruin",
    # other depletion verbs
    "use", "consume", "pay",
}
_COMPARISON_VERBS = {
    "have", "double", "twice", "triple", "half", "more", "less", "fewer", "compare",
}

_WEAK_TRANSFER_VERBS = {"borrow", "lend", "bring", "transport", "carry"}
_WEAK_CREATION_VERBS = {"work", "drive", "walk", "bike", "plant", "write", "practice", "memorize", "mow", "shop"}
_WEAK_LOSS_VERBS = {"leave", "skip", "remove", "use"}
_WEAK_COMPARISON_VERBS = {"be", "need", "cost", "last", "keep", "want", "average", "hold", "weigh", "contain", "count", "plan", "decide", "consider"}


def _event_label_from_class(cls: Type[Event]) -> str:
    if issubclass(cls, TransferEvent):
        return "transfer"
    if issubclass(cls, CreationEvent):
        return "creation"
    if issubclass(cls, LossEvent):
        return "loss"
    if issubclass(cls, ComparisonEvent):
        return "comparison"
    return "neutral"


def classify_event_details(lemma: str) -> Dict[str, object]:
    """Defeasible lexical classifier with confidence for ILP labeling.

    Rules are priority-ordered: exact lexical matches outrank weak defaults,
    and weak defaults outrank WordNet backoff.
    """
    norm = normalize_verb(lemma)
    candidates: List[Tuple[int, float, Type[Event], str]] = []

    if norm in _TRANSFER_VERBS:
        candidates.append((100, 0.98, TransferEvent, "transfer_lexicon"))
    if norm in _CREATION_VERBS:
        candidates.append((95, 0.96, CreationEvent, "creation_lexicon"))
    if norm in _LOSS_VERBS:
        candidates.append((90, 0.95, LossEvent, "loss_lexicon"))
    if norm in _COMPARISON_VERBS:
        candidates.append((85, 0.95, ComparisonEvent, "comparison_lexicon"))

    if norm in _WEAK_TRANSFER_VERBS:
        candidates.append((60, 0.72, TransferEvent, "transfer_default"))
    if norm in _WEAK_CREATION_VERBS:
        candidates.append((55, 0.68, CreationEvent, "creation_default"))
    if norm in _WEAK_LOSS_VERBS:
        candidates.append((58, 0.7, LossEvent, "loss_default"))
    if norm in _WEAK_COMPARISON_VERBS:
        candidates.append((52, 0.65, ComparisonEvent, "comparison_default"))

    wordnet_cls = _wordnet_category(norm)
    if wordnet_cls is not None:
        candidates.append((40, 0.62, wordnet_cls, "wordnet_backoff"))

    if not candidates:
        return {
            "lemma": norm,
            "event_class": NeutralEvent,
            "label": "neutral",
            "confidence": 0.0,
            "reason": "no_match",
        }

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    _, confidence, event_class, reason = candidates[0]
    return {
        "lemma": norm,
        "event_class": event_class,
        "label": _event_label_from_class(event_class),
        "confidence": confidence,
        "reason": reason,
    }


def classify_event_label(lemma: str) -> Tuple[str, float, str]:
    details = classify_event_details(lemma)
    return str(details["label"]), float(details["confidence"]), str(details["reason"])


def sanitize_lemma(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "verb"


def normalize_verb(text: str) -> str:
    lemma = sanitize_lemma(text)
    if "_" in lemma:
        lemma = lemma.split("_")[0]
    mapped = _VERB_ALIASES.get(lemma)
    if mapped:
        return mapped

    # Lightweight lemmatization fallback for common GSM tokens.
    candidates: List[str] = []
    if lemma.endswith("ies") and len(lemma) > 3:
        candidates.append(lemma[:-3] + "y")
    if lemma.endswith("es") and len(lemma) > 2:
        candidates.append(lemma[:-2])
    if lemma.endswith("s") and len(lemma) > 1:
        candidates.append(lemma[:-1])
    if lemma.endswith("ed") and len(lemma) > 2:
        candidates.append(lemma[:-2])
    if lemma.endswith("ing") and len(lemma) > 3:
        candidates.append(lemma[:-3])

    known = (
        _TRANSFER_VERBS
        | _CREATION_VERBS
        | _LOSS_VERBS
        | _COMPARISON_VERBS
        | _WEAK_TRANSFER_VERBS
        | _WEAK_CREATION_VERBS
        | _WEAK_LOSS_VERBS
        | _WEAK_COMPARISON_VERBS
    )
    for cand in candidates:
        mapped = _VERB_ALIASES.get(cand, cand)
        if mapped in known:
            return mapped
    return lemma


def _wordnet_category(lemma: str):
    try:
        from nltk.corpus import wordnet as wn
    except Exception:
        return None

    try:
        synsets = wn.synsets(lemma, pos=wn.VERB)
    except Exception:
        return None

    hypernym_names = set()
    for syn in synsets:
        for hyper in syn.closure(lambda s: s.hypernyms()):
            hypernym_names.update(name.split(".")[0] for name in hyper.lemma_names())
        hypernym_names.update(name.split(".")[0] for syn in synsets for name in syn.lemma_names())

    if hypernym_names & _TRANSFER_VERBS:
        return TransferEvent
    if hypernym_names & _CREATION_VERBS:
        return CreationEvent
    if hypernym_names & _LOSS_VERBS:
        return LossEvent
    if hypernym_names & _COMPARISON_VERBS:
        return ComparisonEvent
    return None


def _base_event_class(lemma: str):
    details = classify_event_details(lemma)
    return details["event_class"]


@lru_cache(maxsize=None)
def resolve_event_class(lemma: str) -> Type[Event]:
    lemma = normalize_verb(lemma)
    base_cls = _base_event_class(lemma)
    class_name = f"{lemma.title().replace('_', '')}{base_cls.__name__}"
    return type(class_name, (base_cls,), {})


def resolve_event_fact(lemma: str, **kwargs):
    return resolve_event_class(lemma)(**kwargs)
