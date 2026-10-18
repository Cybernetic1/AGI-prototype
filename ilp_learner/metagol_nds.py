"""Metagol-like ILP (neo-Davidsonian) — moved from forward-engine.
Run as a script to see a small synthetic demo.
"""
from collections import defaultdict
from difflib import SequenceMatcher
import itertools

# --- Data structures ---
class Atom:
    def __init__(self, pred, args):
        self.pred = pred
        self.args = tuple(args)
    def __repr__(self):
        return f"{self.pred}({', '.join(map(str,self.args))})"
    def __eq__(self, other):
        return isinstance(other, Atom) and self.pred==other.pred and self.args==other.args
    def __hash__(self):
        return hash((self.pred, self.args))

class Clause:
    def __init__(self, head, body):
        self.head = head  # Atom (may contain variables like 'E','X')
        self.body = body  # list[Atom]
    def __repr__(self):
        return f"{self.head} :- {', '.join(map(str,self.body))}"

# --- Simple variable helpers ---
E, X, Y = 'E','X','Y'  # canonical nDS event and role vars

# --- Metarules (nDS-focused) ---
# Each metarule returns templates for head & body using placeholders 'P' and 'Q' for predicate names
# 1) role-transfer: if predicate(e,P) & role(e,X) -> predicate(e,Q)
# 2) role-to-role: if role1(e,X) & role2(e,Y) -> predicate(e,Q)

def instantiate_metarule_role_transfer(p_pred,q_pred, role_pred):
    head = Atom(q_pred, [E])
    body = [Atom(p_pred, [E]), Atom(role_pred, [E, X])]
    return Clause(head, body)

# --- Forward-chaining entailment (simple) ---
# facts: set of ground Atoms (no variables) as strings

def forward_chain(facts, clauses, max_steps=5):
    facts = set(facts)
    for _ in range(max_steps):
        added = False
        for cl in clauses:
            # attempt to match body (only supporting patterns of form pred(E) and role(E,arg))
            # naive matching: find all bindings that satisfy every body atom from current facts
            candidates = match_clause_body(facts, cl.body)
            for subst in candidates:
                head_inst = substitute(cl.head, subst)
                if head_inst not in facts:
                    facts.add(head_inst)
                    added = True
        if not added:
            break
    return facts

# find substitutions (mapping from vars to constants) that make all body atoms present in facts
def match_clause_body(facts, body_atoms):
    # each fact is Atom with concrete args; body atoms may contain variables 'E','X','Y'
    # We'll perform a simple search over facts for each body atom and accumulate consistent bindings.
    candidates = [dict()]
    for atom in body_atoms:
        new_cands = []
        for f in facts:
            if f.pred != atom.pred or len(f.args) != len(atom.args):
                continue
            # check compatibility
            for c in candidates:
                ok = True
                c2 = dict(c)
                for var, val in zip(atom.args, f.args):
                    if is_var(var):
                        if var in c2 and c2[var] != val:
                            ok = False; break
                        c2[var] = val
                    else:
                        if var != val:
                            ok = False; break
                if ok:
                    new_cands.append(c2)
        candidates = new_cands
        if not candidates:
            return []
    return candidates

def is_var(x):
    return isinstance(x, str) and x in {E,X,Y}

def substitute(atom, subst):
    args = [subst.get(a, a) for a in atom.args]
    return Atom(atom.pred, args)

# --- Coverage scoring ---
def score_hypothesis(clauses, pos_examples, neg_examples, background):
    covered = 0
    for ex in pos_examples:
        facts = set(background)
        facts.update([])
        facts = forward_chain(facts, clauses)
        if ex in facts:
            covered += 1
    false_pos = 0
    for ex in neg_examples:
        facts = set(background)
        facts = forward_chain(facts, clauses)
        if ex in facts:
            false_pos += 1
    # simple score: positives - penalty * false positives
    return covered - 2 * false_pos

# --- Simple embedding/prior heuristic ---
# use string similarity to prefer heads that are close to body preds

def embedding_prior(head_pred, body_preds):
    # average sequence-similarity between head and body preds
    s = 0.0
    for b in body_preds:
        s += SequenceMatcher(None, head_pred, b).ratio()
    return s / max(1, len(body_preds))

# --- Beam search learner ---

def learn_metagol_nds(predicates, roles, background, pos_examples, neg_examples, beam=5, max_clauses=3):
    # initial empty hypothesis
    beam_hypos = [([], 0.0)]  # list of (clauses_list, score)
    for step in range(max_clauses):
        candidates = []
        for hy, _ in beam_hypos:
            # expand by adding one clause instantiated from metarule templates
            for p_pred in predicates:
                for q_pred in predicates:
                    for role_pred in roles:
                        clause = instantiate_metarule_role_transfer(p_pred, q_pred, role_pred)
                        new_hyp = hy + [clause]
                        sc = score_hypothesis(new_hyp, pos_examples, neg_examples, background)
                        prior = embedding_prior(q_pred, [p_pred, role_pred])
                        total = sc + 0.5 * prior
                        candidates.append((new_hyp, total))
        # keep top beam
        candidates.sort(key=lambda x: x[1], reverse=True)
        beam_hypos = candidates[:beam]
        # quick early exit if perfect coverage
        if beam_hypos and beam_hypos[0][1] > len(pos_examples) * 0.9:
            break
    best = beam_hypos[0] if beam_hypos else ([], 0.0)
    return best

# --- Demo runner ---
if __name__ == '__main__':
    # simple domain: predicate(e, love), predicate(e, admire), agent(e, A), patient(e, B), male(A), female(B)
    predicates = ['predicate_love', 'predicate_admire', 'predicate_help']
    roles = ['agent', 'patient']

    # background facts (ground atoms)
    # events: e1,e2; persons: p1,p2
    background = [
        Atom('predicate_love', ['e1']),
        Atom('agent', ['e1', 'p1']),
        Atom('patient', ['e1', 'p2']),
        Atom('predicate_admire', ['e2']),
        Atom('agent', ['e2', 'p3']),
        Atom('patient', ['e2', 'p4']),
    ]

    # positive example: infer predicate_help(e1) (pretend predicate_help is correlated with love+agent)
    pos = [Atom('predicate_help', ['e1'])]
    neg = [Atom('predicate_help', ['e2'])]

    best_hyp, score = learn_metagol_nds(predicates, roles, background, pos, neg, beam=6, max_clauses=2)
    print("Best score:", score)
    print("Learned clauses:")
    for c in best_hyp:
        print('  ', c)
    # show forward-chaining result
    inferred = forward_chain(set(background), best_hyp)
    print("Inferred facts:")
    for f in sorted(inferred, key=lambda x: (x.pred, x.args)):
        print('  ', f)
