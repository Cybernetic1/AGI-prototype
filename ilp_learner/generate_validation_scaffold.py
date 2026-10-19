"""Generate a held-out GSM8K ILP validation scaffold for human review.
Selects a stratified sample of tasks from the LT export and emits a JSONL file with:
 - id, verb, class, background (list of simple atoms), pos (head atom),
 - neg_candidates: list of {type, atom, reason, needs_review}
 - review_instructions: short checklist for a human annotator

Write output to ilp_learner/gsm8k_validation_scaffold.jsonl
"""
import json
import random
from pathlib import Path
from collections import defaultdict

LT_PATH = Path('gsm8k-tests/lt_core/data/gsm8k_train_lt.jsonl')
OUT_PATH = Path('ilp_learner/gsm8k_validation_scaffold.jsonl')

random.seed(42)

# lightweight verb->class mapping (same as used in sweeps)
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

# read tasks
tasks = []
with LT_PATH.open() as f:
    for line in f:
        obj = json.loads(line)
        props = obj.get('input_props', [])
        # find query event
        q_events = [p for p in props if p.get('pred') == 'query']
        if q_events:
            q_ev = q_events[0]['args'][0]
        else:
            evs = [p for p in props if p.get('pred') == 'event']
            if not evs:
                continue
            q_ev = evs[0]['args'][0]
        verbs = [p for p in props if p.get('pred') == 'predicate' and p.get('args', [None])[0] == q_ev]
        if not verbs:
            continue
        verb_arg = verbs[0]['args'][1]
        if isinstance(verb_arg, str) and verb_arg.startswith('?'):
            continue
        verb = str(verb_arg)
        cls = ontology_classify(verb)
        if cls == 'neutral':
            continue
        # build background and meta
        background = []
        events = set()
        for p in props:
            pred = p.get('pred')
            args = p.get('args', [])
            if pred == 'predicate':
                ev = args[0]
                verb_a = args[1]
                if isinstance(verb_a, str) and verb_a.startswith('?'):
                    continue
                pred_name = f'predicate_{str(verb_a)}'
                background.append({'pred': pred_name, 'args': [ev]})
                events.add(ev)
            else:
                if len(args) >= 2:
                    ev = args[0]
                    arg = args[1]
                    if isinstance(arg, str) and arg.startswith('?'):
                        arg = arg[1:]
                    background.append({'pred': pred, 'args': [ev, arg]})
                    events.add(ev)
        target = f'class_{cls}'
        pos = {'pred': target, 'args': [q_ev]}
        tasks.append({'id': obj.get('id'), 'verb': verb, 'class': cls, 'background': background, 'pos': pos, 'events': list(events)})

print(f"Loaded {len(tasks)} candidate tasks")

# Stratified sample: try to get up to N_per_class per class
N_per_class = 12
by_class = defaultdict(list)
for t in tasks:
    by_class[t['class']].append(t)

sampled = []
for cls in ALL_CLASSES:
    pool = by_class.get(cls, [])
    if not pool:
        continue
    k = min(N_per_class, len(pool))
    sampled.extend(random.sample(pool, k))

# If fewer than target, fill from remaining
TARGET_TOTAL = 50
if len(sampled) < TARGET_TOTAL:
    remaining = [t for t in tasks if t not in sampled]
    to_add = min(TARGET_TOTAL - len(sampled), len(remaining))
    if to_add > 0:
        sampled.extend(random.sample(remaining, to_add))

print(f"Selected {len(sampled)} tasks for validation scaffold")

# helper to generate candidate negatives per task
def make_candidates(task, all_tasks):
    q_ev = task['pos']['args'][0]
    true_cls = task['class']
    cands = []
    # same-event wrong classes (label swap)
    for oc in ALL_CLASSES:
        if oc == true_cls:
            continue
        cands.append({'type': 'same_event_mislabel', 'atom': {'pred': f'class_{oc}', 'args': [q_ev]}, 'reason': 'different class on same event', 'needs_review': True})
    # role-corrupted: apply true class to another event in same example
    other_events = [e for e in task.get('events', []) if e != q_ev]
    if other_events:
        cands.append({'type': 'role_corrupt', 'atom': {'pred': f'class_{true_cls}', 'args': [other_events[0]]}, 'reason': 'true class applied to different event in same example', 'needs_review': True})
    # cross-task negatives: sample class from other tasks
    samp = 0
    for t2 in random.sample(all_tasks, min(len(all_tasks), 20)):
        if t2 is task:
            continue
        oc = t2['class']
        if oc == true_cls:
            continue
        cands.append({'type': 'cross_task', 'atom': {'pred': f'class_{oc}', 'args': [q_ev]}, 'reason': f'class from other example (id={t2.get("id")})', 'needs_review': True})
        samp += 1
        if samp >= 2:
            break
    return cands

# write scaffold JSONL with review instructions
review_instructions = (
    "For each example, review the positive head and the negative candidates.\n"
    "Mark each negative candidate as 'correct_negative' (i.e., not entailed by the background) or 'incorrect_negative' (i.e., actually entailed / plausible).\n"
    "If any negative seems ambiguous, leave a short comment in the 'notes' field.\n"
    "This scaffold helps build a small held-out validation set for precision/recall evaluation.\n"
)

with OUT_PATH.open('w') as out:
    for t in sampled:
        entry = {
            'id': t.get('id'),
            'verb': t.get('verb'),
            'class': t.get('class'),
            'background': t.get('background'),
            'pos': t.get('pos'),
            'neg_candidates': make_candidates(t, sampled),
            'review_instructions': review_instructions,
        }
        out.write(json.dumps(entry) + '\n')

print('Wrote scaffold to', OUT_PATH)
