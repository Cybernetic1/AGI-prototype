from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, Type

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
}

_TRANSFER_VERBS = {
    "give", "hand", "pass", "send", "transfer", "deliver", "offer",
}
_CREATION_VERBS = {
    "buy", "find", "pick", "bake", "create", "make", "get", "receive", "acquire", "gain", "obtain",
}
_LOSS_VERBS = {
    "eat", "lose", "break", "consume", "drop", "spend", "sell", "discard", "remove", "throw", "shed",
}
_COMPARISON_VERBS = {
    "have", "double", "twice", "triple", "half", "more", "less", "fewer", "compare",
}


def sanitize_lemma(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "verb"


def normalize_verb(text: str) -> str:
    lemma = sanitize_lemma(text)
    return _VERB_ALIASES.get(lemma, lemma)


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
    lemma = normalize_verb(lemma)
    if lemma in _TRANSFER_VERBS:
        return TransferEvent
    if lemma in _CREATION_VERBS:
        return CreationEvent
    if lemma in _LOSS_VERBS:
        return LossEvent
    if lemma in _COMPARISON_VERBS:
        return ComparisonEvent

    wordnet_cls = _wordnet_category(lemma)
    if wordnet_cls is not None:
        return wordnet_cls
    return NeutralEvent


@lru_cache(maxsize=None)
def resolve_event_class(lemma: str) -> Type[Event]:
    lemma = normalize_verb(lemma)
    base_cls = _base_event_class(lemma)
    class_name = f"{lemma.title().replace('_', '')}{base_cls.__name__}"
    return type(class_name, (base_cls,), {})


def resolve_event_fact(lemma: str, **kwargs):
    return resolve_event_class(lemma)(**kwargs)
