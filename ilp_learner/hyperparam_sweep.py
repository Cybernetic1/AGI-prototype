"""Hyperparameter sweep for ILP on GSM8K LT export.
Sweeps beam, max_clauses, prior_weight. Saves summary to ilp_learner/hyperparam_sweep_results.json
"""
import json
import time
from pathlib import Path
from collections import defaultdict
from ilp_learner.metagol_nds import Atom, learn_metagol_nds, forward_chain
from forward_engine.phase3_synthetic_test import ontology_classify

LT_PATH = Path('gsm8k-tests/lt_core/data/gsm8k_train_lt.jsonl')
OUT_PATH = Path('ilp_learner/hyperparam_sweep_results.json')

beams = [1,3,6,10]
max_clauses_list = [1,2,3]
prior_weights = [0.0, 0.5, 1.0]

# collect all labeled tasks first
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
                try:
                    best_hyp, score = learn_metagol_nds(preds, roles, bg, pos, [], beam=beam, max_clauses=max_clauses, prior_weight=prior_weight)
                except Exception:
                    best_hyp, score = [], 0.0
                inferred = forward_chain(set(bg), best_hyp)
                # coverage: pos in inferred
                covered = 1.0 if pos[0] in inferred else 0.0
                metrics['covered'] += covered
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
                'avg_score': metrics['score']/max(1,n),
                'avg_num_clauses': metrics['num_clauses']/max(1,n),
                'duration_sec': dur,
            }
            print(f"beam={beam} max_clauses={max_clauses} prior={prior_weight} coverage={summary['coverage_rate']:.3f} avg_score={summary['avg_score']:.3f} dur={dur:.2f}s")
            results.append(summary)
end_all = time.time()
with OUT_PATH.open('w') as f:
    json.dump({'total_duration_sec': end_all-start_all, 'sweep_results': results}, f, indent=2)
print('Saved sweep results to', OUT_PATH)
