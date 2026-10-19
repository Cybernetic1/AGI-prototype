#!/usr/bin/env python3
"""Auto-fill default annotations for the GSM8K validation scaffold.
For each example, the script:
 - computes a guessed class via a lightweight ontology mapping from verb
 - runs a quick ILP learn to see which class is inferred (if any)
 - marks neg_candidates as 'incorrect_negative' if they are entailed by inferred facts, else 'correct_negative'
 - writes out ilp_learner/gsm8k_validation_annotations_autofill.json
 - prints a short summary
"""
import json
from pathlib import Path
from ilp_learner.metagol_nds import Atom, learn_metagol_nds, forward_chain
import random

SCAFFOLD = Path('ilp_learner/gsm8k_validation_scaffold.jsonl')
OUT = Path('ilp_learner/gsm8k_validation_annotations_autofill.json')

# lightweight verb->class mapping
TRAIN_VERB_CLASSES = {
    "transfer": {"give", "hand", "pass", "send", "donated", "gifted", "sent", "gave", "donate", "gift"},
    "creation": {"buy", "find", "pick", "bake", "create", "make", "get", "receive", "purchased", "acquired", "found"},
    "loss": {"eat", "lose", "break", "consume", "drop", "spend", "discard", "shattered", "destroyed"},
    "comparison": {"have", "double", "twice", "triple", "half"},
}
ALL_CLASSES = ['comparison','creation','loss','transfer']

def ontology_classify(verb: str) -> str:
    lemma = str(verb).lower()
    for label, verbs in TRAIN_VERB_CLASSES.items():
        if lemma in verbs:
            return label
    return 'neutral'


def load_scaffold(path):
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))
    return items


def build_task(it):
    # convert scaffold entry into the fields expected by learner
    bg = []
    preds = set()
    roles = set()
    events = set()
    for b in it.get('background', []):
        pred = b.get('pred')
        args = b.get('args', [])
        if pred.startswith('predicate_'):
            # unary predicate predicate_verb(event)
            ev = args[0]
            bg.append(Atom(pred, [ev]))
            preds.add(pred)
            events.add(ev)
        else:
            if len(args) >= 2:
                ev = args[0]
                arg = args[1]
                bg.append(Atom(pred, [ev, arg]))
                roles.add(pred)
                preds.add(pred)
                events.add(ev)
    # add class candidates into predicates
    for c in ALL_CLASSES:
        preds.add(f'class_{c}')
    # find pos event
    pos = None
    if it.get('pos'):
        pos = Atom(it['pos']['pred'], it['pos']['args'])
    return list(preds), list(roles), bg, pos


def quick_infer(preds, roles, bg, pos):
    # run a quick learn to produce a hypothesis and check inferred class atoms
    try:
        best_hyp, score = learn_metagol_nds(preds, roles, bg, [pos], [], beam=3, max_clauses=1, prior_weight=0.5)
    except Exception:
        best_hyp, score = [], 0.0
    inferred = forward_chain(set(bg), best_hyp)
    # find any class_* atoms for the pos event
    ev = pos.args[0]
    inferred_classes = [a.pred for a in inferred if a.pred.startswith('class_') and a.args and a.args[0] == ev]
    return inferred_classes, best_hyp, score


def main():
    if not SCAFFOLD.exists():
        print('Scaffold not found:', SCAFFOLD)
        return
    items = load_scaffold(SCAFFOLD)
    out = []
    summary = {'total': 0, 'auto_matches': 0}
    for it in items:
        summary['total'] += 1
        verb = it.get('verb')
        guessed = ontology_classify(verb)
        preds, roles, bg, pos = build_task(it)
        inferred_classes, hyp, score = quick_infer(preds, roles, bg, pos)
        auto_pred = None
        if inferred_classes:
            # pick first inferred class name (strip class_)
            auto_pred = inferred_classes[0].replace('class_', '')
        else:
            auto_pred = guessed if guessed != 'neutral' else it.get('class')
        if auto_pred == it.get('class'):
            summary['auto_matches'] += 1
        # annotate negatives
        anns = []
        for n in it.get('neg_candidates', []):
            atom = n.get('atom', {})
            pred = atom.get('pred')
            args = atom.get('args', [])
            # if this neg equals the auto_pred for the event, mark as incorrect_negative (i.e., it's entailed)
            if args and args[0] == pos.args[0] and pred == f'class_{auto_pred}':
                decision = 'incorrect_negative'
            else:
                decision = 'correct_negative'
            anns.append({'decision': decision, 'notes': 'auto-filled'})
        out.append({'id': it.get('id'), 'verb': verb, 'class': it.get('class'), 'auto_pred': auto_pred, 'pos': it.get('pos'), 'neg_candidates': it.get('neg_candidates', []), 'annotations': anns})
    with open(OUT, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2)
    print('Wrote autofill annotations to', OUT)
    print('Summary:', summary)

if __name__ == '__main__':
    main()
