"""Run a verbose ILP demo that prints Lorentz prior values for candidate clauses.
"""
from ilp_learner.metagol_nds import (
    Atom,
    instantiate_metarule_role_transfer,
    embedding_prior,
    learn_metagol_nds,
)


def main():
    predicates = ['predicate_love', 'predicate_admire', 'predicate_help']
    roles = ['agent', 'patient']

    print('Checking embedding state and priors for candidate clauses...')
    candidates = []
    for p in predicates:
        for q in predicates:
            for r in roles:
                prior = embedding_prior(q, [p, r])
                candidates.append((prior, p, q, r))
    candidates.sort(reverse=True, key=lambda x: x[0])
    print('\nTop candidate priors:')
    for prior, p, q, r in candidates[:20]:
        print(f'  prior={prior:.6f}  head={q}  body=[{p},{r}]')

    # Run the learner and print final hypothesis
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

    best_hyp, score = learn_metagol_nds(predicates, roles, background, pos, neg, beam=6, max_clauses=2)
    print('\nLearned hypothesis score:', score)
    print('Learned clauses:')
    for c in best_hyp:
        print('  ', c)


if __name__ == '__main__':
    main()
