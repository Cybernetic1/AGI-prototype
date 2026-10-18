"""Run ILP sweep over beam sizes and max_clauses for the demo task.
Outputs results to stdout and saves JSON to ilp_learner/experiment_results.json
"""
import json
from ilp_learner.metagol_nds import (
    Atom,
    learn_metagol_nds,
    embedding_prior,
    forward_chain,
)

# demo domain (same as run_verbose_demo)
predicates = ['predicate_love', 'predicate_admire', 'predicate_help']
roles = ['agent', 'patient']
background = [
    Atom('predicate_love', ['e1']),
    Atom('agent', ['e1', 'p1']),
    Atom('patient', ['e1', 'p2']),
    Atom('predicate_admire', ['e2']),
    Atom('agent', ['e2', 'p3']),
    Atom('patient', ['e2', 'p4']),
]
pos = [Atom('predicate_help', ['e1'])]
neg = [Atom('predicate_help', ['e2'])]

beams = [1, 3, 6, 10]
max_clauses_list = [1, 2, 3]

results = []
for beam in beams:
    for max_clauses in max_clauses_list:
        best_hyp, score = learn_metagol_nds(predicates, roles, background, pos, neg, beam=beam, max_clauses=max_clauses)
        # Serialize clause list
        clauses_str = [str(c) for c in best_hyp]
        # Infer with forward chaining
        inferred = forward_chain(set(background), best_hyp)
        inferred_str = [str(a) for a in sorted(inferred, key=lambda x: (x.pred, x.args))]
        res = {
            'beam': beam,
            'max_clauses': max_clauses,
            'score': float(score),
            'num_clauses': len(best_hyp),
            'clauses': clauses_str,
            'inferred': inferred_str,
        }
        results.append(res)
        print(f"beam={beam} max_clauses={max_clauses} score={score:.4f} clauses={clauses_str}")

# save results
with open('ilp_learner/experiment_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nSaved results to ilp_learner/experiment_results.json')
