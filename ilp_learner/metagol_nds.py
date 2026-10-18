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

# --- Embedding / Lorentz prior adapter ---
# Prefer candidates whose predicate embeddings are close in Lorentz space.
import os
try:
    import torch
    from logic_transformer.lorentz_math import project_to_hyperboloid, lorentz_distance
except Exception:
    torch = None

_EMB_STATE = None

def _load_aligned_embeddings(emb_path='logic_transformer/checkpoints/aligned_hyperbolic_embeddings_full.pt', vocab_source='logic_transformer/checkpoints/aligned_hyperbolic_embeddings_full_vocab.json'):
    """Attempt to load aligned embeddings and vocabulary mapping from checkpoints by default.
    Returns a dict with keys 'weights' (Tensor), 'projected' (Tensor) and 'vocab' (dict token->idx) on success, else None.
    """
    import json
    global _EMB_STATE
    if _EMB_STATE is not None:
        return _EMB_STATE
    if torch is None:
        _EMB_STATE = None
        return None
    try:
        if not os.path.exists(emb_path):
            _EMB_STATE = None
            return None
        emb_state = torch.load(emb_path, map_location='cpu')
        # emb_state is likely a state_dict like {'weight': Tensor}
        weights = None
        if isinstance(emb_state, dict):
            for k in ('weight', 'embeddings.weight', 'emb.weight'):
                if k in emb_state:
                    weights = emb_state[k]
                    break
            if weights is None:
                # fallback to first tensor-like value
                for v in emb_state.values():
                    if hasattr(v, 'ndim'):
                        weights = v
                        break
        else:
            weights = emb_state
        if weights is None:
            _EMB_STATE = None
            return None
        # load vocab json if present
        vocab = None
        try:
            if os.path.exists(vocab_source):
                with open(vocab_source, 'r', encoding='utf-8') as fh:
                    vocab = json.load(fh)
        except Exception:
            vocab = None
        # project weights onto hyperboloid if possible
        projected = None
        try:
            projected = project_to_hyperboloid(weights)
        except Exception:
            projected = weights
        _EMB_STATE = { 'weights': weights, 'projected': projected, 'vocab': vocab }
        return _EMB_STATE
    except Exception:
        _EMB_STATE = None
        return None

import hashlib
import numpy as _np
from pathlib import Path

def _synthetic_lorentz(pred, dim=16):
    """Deterministic synthetic vector generator on the Lorentz hyperboloid.
    Uses SHA256 of predicate name to seed a pseudo-random vector, then projects.
    """
    # create reproducible pseudo-random vector from hash
    h = hashlib.sha256(pred.encode('utf-8')).digest()
    arr = _np.frombuffer(h, dtype=_np.uint8).astype(_np.float32)
    # expand or truncate to dim-1 spatial dims
    spatial = _np.zeros(dim-1, dtype=_np.float32)
    spatial[:min(len(arr), dim-1)] = arr[:min(len(arr), dim-1)] / 255.0
    # ensure not all zeros
    if _np.allclose(spatial, 0.0):
        spatial[0] = 1e-3
    # compute time-like coordinate
    t = _np.sqrt(1.0 + _np.sum(spatial**2) + 1e-6)
    vec = _np.concatenate([_np.array([t], dtype=_np.float32), spatial])
    if torch is not None:
        return torch.from_numpy(vec)
    else:
        return vec

def embedding_prior(head_pred, body_preds, emb_path='/tmp/aligned_hyperbolic_embeddings.pt', vocab_source='/tmp/pot_lt_pairs.jsonl'):
    """Return a prior score (higher is better) for head_pred given body_preds.
    Tries to use pre-aligned embeddings; falls back to deterministic synthetic Lorentz vectors.
    """
    state = _load_aligned_embeddings(emb_path, vocab_source)
    head_emb = None
    body_embs = []
    # helper to get embedding for a predicate name
    def get_emb(pred):
        # try aligned weights/vocab if available
        if state and state.get('vocab'):
            # find a vocab token starting with pred + '('
            for tok, idx in state['vocab'].items():
                if tok.startswith(pred + '('):
                    try:
                        vec = state['projected'][idx]
                        return vec
                    except Exception:
                        try:
                            return state['weights'][idx]
                        except Exception:
                            break
        # fallback synthetic
        return _synthetic_lorentz(pred, dim=state['projected'].size(1) if (state and state.get('projected') is not None and hasattr(state['projected'], 'size')) else 16)

    try:
        head_emb = get_emb(head_pred)
        for b in body_preds:
            body_embs.append(get_emb(b))
        if torch is not None and 'lorentz_distance' in globals():
            # compute mean Lorentz distance (smaller is closer) and convert to similarity
            dists = []
            for be in body_embs:
                try:
                    d = lorentz_distance(head_emb.unsqueeze(0), be.unsqueeze(0)).item()
                except Exception:
                    # fallback numeric
                    d = float('inf')
                dists.append(d)
            mean_d = sum(dists) / max(1, len(dists))
            # convert to prior in [0, 1] (arbitrary): higher when closer
            return 1.0 / (1.0 + mean_d)
    except Exception:
        pass
    # final fallback: string similarity as before
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
