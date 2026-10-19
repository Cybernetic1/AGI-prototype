"""Hyperparameter sweep for ILP on GSM8K LT export with synthetic negatives.
Generates simple negative examples per task and sweeps beam, max_clauses, prior_weight.
Saves summary to ilp_learner/hyperparam_sweep_with_neg_results.json
"""
import json
import time
from pathlib import Path
from collections import defaultdict
from ilp_learner.metagol_nds import Atom, learn_metagol_nds, forward_chain
# Local lightweight ontology classifier (copied mapping from forward_engine.phase3_synthetic_test)
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
OUT_PATH = Path('ilp_learner/hyperparam_sweep_with_neg_results.json')

beams = [1,3,6,10]
max_clauses_list = [1,2,3]
prior_weights = [0.0, 0.5, 1.0]

# helper: create simple negatives for a task
def make_negatives(task, n=2):
    # task: dict with keys id, verb, class, background, predicates, roles, pos
    negs = []
    q_ev = task['pos'][0].args[0]
    true_cls = task['class']
    other_classes = ['comparison','creation','loss','transfer']
    # pick up to n other classes as negatives for same event
    cnt = 0
    for oc in other_classes:
        if oc == true_cls:
            continue
        negs.append(Atom(f'class_{oc}', [q_ev]))
        cnt += 1
        if cnt >= n:
            break
    # as an extra negative, take a predicate head wrongly assigned to this event
    # pick a predicate name from predicates but not one appearing in background for q_ev
    try:
        preds = task.get('predicates', [])
        for p in preds:
            if not p.startswith('class_'):
                negs.append(Atom(p, [q_ev]))
                break
    except Exception:
        pass
    return negs

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
            else:
                if len(args) >= 2:
                    ev = args[0]
                    arg = args[1]
                    if isinstance(arg, str) and arg.startswith('?'):
                        arg = arg[1:]
                    roles.add(pred)
                    background.append(Atom(pred, [ev, arg]))
        target = f'class_{cls}'
        predicates.add(target)
        pos = [Atom(target, [q_ev])]
        tasks.append({'id': obj.get('id'), 'verb': verb, 'class': cls, 'background': background, 'predicates': list(predicates), 'roles': list(roles), 'pos': pos})

print(f"Collected {len(tasks)} labeled tasks from LT export")

results = []
start_all = time.time()
for beam in beams:
    for max_clauses in max_clauses_list:
        for prior_weight in prior_weights:
            t0 = time.time()
            metrics = defaultdict(float)
            n = 0
            for tk in tasks:
                preds = tk['predicates']
                roles = tk['roles'] if tk['roles'] else ['agent','patient']
                bg = tk['background']
                pos = tk['pos']
                neg = make_negatives(tk, n=2)
                try:
                    best_hyp, score = learn_metagol_nds(preds, roles, bg, pos, neg, beam=beam, max_clauses=max_clauses, prior_weight=prior_weight)
                except Exception:
                    best_hyp, score = [], 0.0
                inferred = forward_chain(set(bg), best_hyp)
                # coverage: pos in inferred
                covered = 1.0 if pos[0] in inferred else 0.0
                # count false positives (negatives inferred)
                false_pos = 0
                for nx in neg:
                    if nx in inferred:
                        false_pos += 1
                metrics['covered'] += covered
                metrics['false_pos'] += false_pos
                metrics['score'] += float(score)
                metrics['num_clauses'] += len(best_hyp)
                n += 1
            dur = time.time() - t0
            summary = {
                'beam': beam,
                'max_clauses': max_clauses,
                'prior_weight': prior_weight,
                'n_tasks': n,
                'covered': metrics['covered'],
                'coverage_rate': metrics['covered']/max(1,n),
                'false_pos_total': metrics['false_pos'],
                'avg_false_pos_per_task': metrics['false_pos']/max(1,n),
                'avg_score': metrics['score']/max(1,n),
                'avg_num_clauses': metrics['num_clauses']/max(1,n),
                'duration_sec': dur,
            }
            print(f"beam={beam} max_clauses={max_clauses} prior={prior_weight} coverage={summary['coverage_rate']:.3f} avg_fp={summary['avg_false_pos_per_task']:.3f} avg_score={summary['avg_score']:.3f} dur={dur:.2f}s")
            results.append(summary)
end_all = time.time()
with OUT_PATH.open('w') as f:
    json.dump({'total_duration_sec': end_all-start_all, 'sweep_results': results}, f, indent=2)
print('Saved sweep results to', OUT_PATH)
