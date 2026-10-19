"""Hyperparameter sweep with harder negatives and precision/recall metrics.
Negatives per task include:
 - class labels different from the true class applied to the same event (class_mislabel)
 - role-corrupted negatives: true class applied to a different event present in the background
 - cross-task negatives: class labels sampled from other tasks applied to this event

Saves summary to ilp_learner/hyperparam_sweep_hard_neg_results.json
"""
import json
import time
from pathlib import Path
from collections import defaultdict, Counter
import random
from ilp_learner.metagol_nds import Atom, learn_metagol_nds, forward_chain

# local lightweight ontology classifier mapping
TRAIN_VERB_CLASSES = {
    "transfer": {"give", "hand", "pass", "send", "donated", "gifted", "sent", "gave", "donate", "gift"},
    "creation": {"buy", "find", "pick", "bake", "create", "make", "get", "receive", "purchased", "acquired", "found"},
    "loss": {"eat", "lose", "break", "consume", "drop", "spend", "discard", "shattered", "destroyed"},
    "comparison": {"have", "double", "twice", "triple", "half"},
}

def ontology_classify(verb: str) -> str:
    lemma = str(verb).lower()
    for label, verbs in TRAIN_VERB_CLASSES.items():
        if lemma in verbs:
            return label
    return "neutral"

LT_PATH = Path('gsm8k-tests/lt_core/data/gsm8k_train_lt.jsonl')
OUT_PATH = Path('ilp_learner/hyperparam_sweep_hard_neg_results.json')

beams = [1,3,6,10]
max_clauses_list = [1,2,3]
prior_weights = [0.0, 0.5, 1.0]

random.seed(0)

# collect labeled tasks first
tasks = []
with LT_PATH.open() as f:
    for i, line in enumerate(f):
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
        predicates = set()
        roles = set()
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
                predicates.add(pred_name)
                background.append(Atom(pred_name, [ev]))
                events.add(ev)
            else:
                if len(args) >= 2:
                    ev = args[0]
                    arg = args[1]
                    if isinstance(arg, str) and arg.startswith('?'):
                        arg = arg[1:]
                    roles.add(pred)
                    background.append(Atom(pred, [ev, arg]))
                    events.add(ev)
        target = f'class_{cls}'
        predicates.add(target)
        pos = [Atom(target, [q_ev])]
        tasks.append({'id': obj.get('id'), 'verb': verb, 'class': cls, 'background': background, 'predicates': list(predicates), 'roles': list(roles), 'pos': pos, 'events': list(events)})

print(f"Collected {len(tasks)} labeled tasks from LT export")

# helper: create harder negatives for a task
# strategy: same-event wrong classes, role-corrupted (true class applied to other event in same task), cross-task class applied to this event
ALL_CLASSES = ['comparison','creation','loss','transfer']

def make_hard_negatives(task, all_tasks, n_same=2, n_cross=2):
    q_ev = task['pos'][0].args[0]
    true_cls = task['class']
    negs = []
    # same-event wrong classes
    for oc in ALL_CLASSES:
        if oc == true_cls:
            continue
        negs.append(Atom(f'class_{oc}', [q_ev]))
        if len(negs) >= n_same:
            break
    # role-corrupted: true class applied to a different event in same background (if available)
    other_events = [e for e in task.get('events', []) if e != q_ev]
    if other_events:
        negs.append(Atom(f'class_{true_cls}', [other_events[0]]))
    # cross-task: sample a few classes from other tasks and apply to this event
    sampled = 0
    for t2 in random.sample(all_tasks, min(len(all_tasks), 10)):
        if t2 is task:
            continue
        oc = t2['class']
        if oc == true_cls:
            continue
        negs.append(Atom(f'class_{oc}', [q_ev]))
        sampled += 1
        if sampled >= n_cross:
            break
    # deduplicate
    seen = set()
    uniq = []
    for n in negs:
        if (n.pred, n.args) not in seen:
            uniq.append(n)
            seen.add((n.pred, n.args))
    return uniq

results = []
start_all = time.time()
for beam in beams:
    for max_clauses in max_clauses_list:
        for prior_weight in prior_weights:
            t0 = time.time()
            total_tp = total_fp = total_pos = 0
            total_score = 0.0
            total_clauses = 0
            n = 0
            for tk in tasks:
                preds = tk['predicates']
                roles = tk['roles'] if tk['roles'] else ['agent','patient']
                bg = tk['background']
                pos = tk['pos']
                neg = make_hard_negatives(tk, tasks, n_same=2, n_cross=2)
                try:
                    best_hyp, score = learn_metagol_nds(preds, roles, bg, pos, neg, beam=beam, max_clauses=max_clauses, prior_weight=prior_weight)
                except Exception:
                    best_hyp, score = [], 0.0
                inferred = forward_chain(set(bg), best_hyp)
                tp = 1 if pos[0] in inferred else 0
                fp = sum(1 for nx in neg if nx in inferred)
                total_tp += tp
                total_fp += fp
                total_pos += 1
                total_score += float(score)
                total_clauses += len(best_hyp)
                n += 1
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            recall = total_tp / max(1, total_pos)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            dur = time.time() - t0
            summary = {
                'beam': beam,
                'max_clauses': max_clauses,
                'prior_weight': prior_weight,
                'n_tasks': n,
                'total_tp': total_tp,
                'total_fp': total_fp,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'avg_score': total_score / max(1,n),
                'avg_num_clauses': total_clauses / max(1,n),
                'duration_sec': dur,
            }
            print(f"beam={beam} max_clauses={max_clauses} prior={prior_weight} prec={precision:.3f} rec={recall:.3f} f1={f1:.3f} dur={dur:.2f}s")
            results.append(summary)
end_all = time.time()
with OUT_PATH.open('w') as f:
    json.dump({'total_duration_sec': end_all-start_all, 'sweep_results': results}, f, indent=2)
print('Saved sweep results to', OUT_PATH)
