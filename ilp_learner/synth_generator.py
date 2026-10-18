"""Synthetic data generator for neo-Davidsonian ILP demos.
Produces small background, positives and negatives suitable for the learner.
Moved to ilp_learner and adapted to import Atom from metagol_nds.
"""
try:
    from ilp_learner.metagol_nds import Atom
except Exception:
    # fallback local Atom definition
    class Atom:
        def __init__(self, pred, args):
            self.pred = pred
            self.args = tuple(args)
        def __repr__(self):
            return f"{self.pred}({', '.join(map(str,self.args))})"


def sample_domain():
    predicates = ['predicate_love','predicate_help','predicate_admire']
    roles = ['agent','patient']

    # small background with 3 events and persons
    background = [
        Atom('predicate_love', ['e1']), Atom('agent', ['e1','alice']), Atom('patient', ['e1','bob']),
        Atom('predicate_admire', ['e2']), Atom('agent', ['e2','carol']), Atom('patient', ['e2','dave']),
        Atom('predicate_help', ['e3']), Atom('agent', ['e3','erin']), Atom('patient', ['e3','frank']),
    ]
    # positives: want to learn a rule that predicts predicate_help(e1) because love+agent->help in this toy domain
    pos = [Atom('predicate_help', ['e1'])]
    neg = [Atom('predicate_help', ['e2']), Atom('predicate_help', ['e3'])]
    return predicates, roles, background, pos, neg

if __name__ == '__main__':
    preds, roles, bg, pos, neg = sample_domain()
    print('Background:')
    for b in bg: print(' ', b)
    print('Pos:', pos)
    print('Neg:', neg)
