"""Metagol-like ILP (neo-Davidsonian) — moved from forward-engine.
Run as a script to see a small synthetic demo.
"""
from collections import defaultdict
import copy
from difflib import SequenceMatcher
import itertools
import os

AGI_TRANSFER_METARULES_ENV = "AGI_ENABLE_TRANSFER_METARULES"
AGI_TRANSFER_FAMILIES = (
    "part_whole_balance",
    "temporal_causal_chain",
    "unit_compatibility",
)

_TOGGLE_TRUE_VALUES = {"1", "true", "yes", "on"}


def resolve_agi_transfer_toggle(include_agi_transfer_families=None):
    if include_agi_transfer_families is not None:
        return bool(include_agi_transfer_families)
    raw = str(os.getenv(AGI_TRANSFER_METARULES_ENV, "")).strip().lower()
    return raw in _TOGGLE_TRUE_VALUES


UNIT_DIMENSIONS = {
    "second": "time",
    "seconds": "time",
    "minute": "time",
    "minutes": "time",
    "hour": "time",
    "hours": "time",
    "day": "time",
    "days": "time",
    "week": "time",
    "weeks": "time",
    "meter": "distance",
    "meters": "distance",
    "centimeter": "distance",
    "centimeters": "distance",
    "kilometer": "distance",
    "kilometers": "distance",
    "mile": "distance",
    "miles": "distance",
    "gram": "mass",
    "grams": "mass",
    "kilogram": "mass",
    "kilograms": "mass",
    "pound": "mass",
    "pounds": "mass",
    "liter": "volume",
    "liters": "volume",
    "milliliter": "volume",
    "milliliters": "volume",
    "dozen": "count",
    "item": "count",
    "items": "count",
    "piece": "count",
    "pieces": "count",
    "usd": "currency",
    "dollar": "currency",
    "dollars": "currency",
}

PART_TYPE_HINTS = ("part", "component", "piece", "subset", "segment")
WHOLE_TYPE_HINTS = ("whole", "total", "group", "set", "collection")
TEMPORAL_TYPE_HINTS = ("before", "after", "later", "earlier", "time", "step", "stage")
CAUSAL_TYPE_HINTS = ("cause", "delta", "effect", "impact", "change", "trigger")

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


def instantiate_metarule_role_pair(q_pred, role_a, role_b):
    head = Atom(q_pred, [E])
    body = [Atom(role_a, [E, X]), Atom(role_b, [E, Y])]
    return Clause(head, body)


SEMANTIC_BINDING_ROLES = {
    "agent",
    "patient",
    "recipient",
    "owner",
    "item",
    "name",
    "relation",
    "source",
    "target",
}


def constrain_roles_for_search(roles, action_constraint="none"):
    role_list = list(dict.fromkeys(roles or []))
    if str(action_constraint or "none").lower() != "semantic_binding":
        return role_list
    constrained = [r for r in role_list if str(r).lower() in SEMANTIC_BINDING_ROLES]
    return constrained if constrained else role_list


def semantic_binding_bonus(body_preds, action_constraint="none", binding_bias=0.0):
    if str(action_constraint or "none").lower() != "semantic_binding":
        return 0.0
    if not body_preds:
        return 0.0
    hits = sum(1 for pred in body_preds if str(pred).lower() in SEMANTIC_BINDING_ROLES)
    ratio = float(hits) / float(len(body_preds))
    return float(binding_bias) * ratio


SEMANTIC_ROLE_PAIR_HINTS = {
    tuple(sorted(("agent", "patient"))),
    tuple(sorted(("agent", "recipient"))),
    tuple(sorted(("owner", "item"))),
    tuple(sorted(("source", "target"))),
}


def semantic_role_pair_bonus(body_preds, metarule_type="", pair_bias=0.0):
    if float(pair_bias) == 0.0:
        return 0.0
    if str(metarule_type) != "role_pair":
        return 0.0
    if len(body_preds) < 2:
        return 0.0
    key = tuple(sorted((str(body_preds[0]).lower(), str(body_preds[1]).lower())))
    if key in SEMANTIC_ROLE_PAIR_HINTS:
        return float(pair_bias)
    return 0.0


def preferred_role_pair_bonus(body_preds, metarule_type="", preferred_role_pairs=None, pair_match_bias=0.0, pair_miss_penalty=0.0):
    if str(metarule_type) != "role_pair":
        return 0.0
    if len(body_preds) < 2:
        return 0.0
    pairs = {tuple(str(a).lower() for a in pair) for pair in (preferred_role_pairs or []) if len(pair) >= 2}
    if not pairs:
        return 0.0
    key = (str(body_preds[0]).lower(), str(body_preds[1]).lower())
    if key in pairs:
        return float(pair_match_bias)
    return float(pair_miss_penalty)


def head_preference_bonus(head_pred, preferred_heads=None, class_head_bias=0.0, nonclass_head_penalty=0.0):
    pred = str(head_pred or "")
    preferred = set(preferred_heads or [])
    bonus = 0.0
    if preferred and pred in preferred:
        bonus += float(class_head_bias)
    if pred and not pred.startswith("class_"):
        bonus += float(nonclass_head_penalty)
    return bonus


def self_reference_penalty(head_pred, body_preds=None, penalty=0.0):
    if float(penalty) == 0.0:
        return 0.0
    if str(head_pred or "") in {str(p) for p in (body_preds or [])}:
        return float(penalty)
    return 0.0


def event_local_role_bonus(body_preds, preferred_local_roles=None, local_role_bias=0.0):
    if float(local_role_bias) == 0.0:
        return 0.0
    role_set = {str(r).lower() for r in (preferred_local_roles or [])}
    if not role_set:
        return 0.0
    hits = sum(1 for pred in (body_preds or []) if str(pred).lower() in role_set)
    if hits <= 0:
        return 0.0
    ratio = float(hits) / max(1.0, float(len(body_preds or [])))
    return float(local_role_bias) * ratio


def body_predicate_alignment_bonus(
    body_preds,
    preferred_body_preds=None,
    body_predicate_bias=0.0,
    body_predicate_miss_penalty=0.0,
):
    preferred = {str(p).lower() for p in (preferred_body_preds or [])}
    if not preferred:
        return 0.0
    body = {str(p).lower() for p in (body_preds or [])}
    if body.intersection(preferred):
        return float(body_predicate_bias)
    return float(body_predicate_miss_penalty)


def clause_economy_penalty(hyp_size, clause_cost=0.0):
    if float(clause_cost) == 0.0:
        return 0.0
    return float(clause_cost) * max(0.0, float(hyp_size) - 1.0)


def required_role_coverage_bonus(
    body_preds,
    required_roles=None,
    required_role_bias=0.0,
    missing_required_role_penalty=0.0,
):
    roles = {str(r).lower() for r in (required_roles or [])}
    if not roles:
        return 0.0
    body = {str(p).lower() for p in (body_preds or [])}
    missing = roles.difference(body)
    if not missing:
        return float(required_role_bias)
    return float(missing_required_role_penalty) * float(len(missing))


def _clause_head_and_body_preds(clause):
    try:
        head_pred = str(getattr(getattr(clause, "head", None), "pred", ""))
        body = list(getattr(clause, "body", []) or [])
        body_preds = [str(getattr(atom, "pred", "")) for atom in body]
        return head_pred, body_preds
    except Exception:
        return "", []


def mirrored_role_pair_penalty(existing_hyp, candidate_clause, candidate_type="", penalty=0.0):
    if float(penalty) == 0.0 or str(candidate_type) != "role_pair":
        return 0.0
    cand_head, cand_body = _clause_head_and_body_preds(candidate_clause)
    if len(cand_body) < 2:
        return 0.0
    rev = [cand_body[1], cand_body[0]]
    for cl in existing_hyp or []:
        head, body = _clause_head_and_body_preds(cl)
        if head == cand_head and body[:2] == rev:
            return float(penalty)
    return 0.0


def repeated_head_penalty(existing_hyp, candidate_clause, penalty=0.0):
    if float(penalty) == 0.0:
        return 0.0
    cand_head, _ = _clause_head_and_body_preds(candidate_clause)
    for cl in existing_hyp or []:
        head, _ = _clause_head_and_body_preds(cl)
        if head == cand_head:
            return float(penalty)
    return 0.0


def _infer_semantic_type(value):
    if isinstance(value, (int, float)):
        return "number"
    token = str(value).strip().lower()
    if not token:
        return "unknown"
    dimension = UNIT_DIMENSIONS.get(token)
    if dimension:
        return f"unit:{dimension}"
    if any(hint in token for hint in WHOLE_TYPE_HINTS):
        return "whole_entity"
    if any(hint in token for hint in PART_TYPE_HINTS):
        return "part_entity"
    if any(hint in token for hint in TEMPORAL_TYPE_HINTS):
        return "temporal_marker"
    if any(hint in token for hint in CAUSAL_TYPE_HINTS):
        return "causal_marker"
    return "entity"


def _check_part_whole(tokens):
    has_part = any(any(hint in tok for hint in PART_TYPE_HINTS) for tok in tokens)
    has_whole = any(any(hint in tok for hint in WHOLE_TYPE_HINTS) for tok in tokens)
    return has_part and has_whole


def _check_temporal_causal(tokens):
    has_temporal = any(any(hint in tok for hint in TEMPORAL_TYPE_HINTS) for tok in tokens)
    has_causal = any(any(hint in tok for hint in CAUSAL_TYPE_HINTS) for tok in tokens)
    return has_temporal and has_causal


def _check_unit_compatibility(tokens):
    units = [UNIT_DIMENSIONS[tok] for tok in tokens if tok in UNIT_DIMENSIONS]
    if len(units) < 2:
        return False
    return len(set(units)) == 1


def evaluate_candidate_constraints(candidate, examples):
    """Evaluate optional typed constraints for a metarule candidate.

    Returns `(is_valid, report)` where `report` carries `score` and any violations.
    """
    constraints = candidate.get("constraints") or {}
    if not constraints:
        return True, {"score": 1.0, "violations": [], "checks": []}
    if not examples:
        return True, {"score": 1.0, "violations": [], "checks": []}

    body_preds = {
        str(pred).strip().lower()
        for pred, _ in (candidate.get("body") or [])
        if isinstance(pred, str)
    }
    required_body_preds = {
        str(pred).strip().lower()
        for pred in (constraints.get("required_body_predicates") or [])
        if str(pred).strip()
    }
    missing_body = sorted(required_body_preds.difference(body_preds))
    violations = []
    if missing_body:
        violations.append(f"missing_body_predicates:{','.join(missing_body)}")

    checks = [str(item).strip().lower() for item in (constraints.get("compatibility_checks") or []) if str(item).strip()]
    required_type_counts = {
        str(k).strip().lower(): int(v)
        for k, v in (constraints.get("required_type_counts") or {}).items()
    }

    pass_count = 0
    for pred_name, args, _ in examples:
        tokens = [str(pred_name).strip().lower()]
        tokens.extend(str(arg).strip().lower() for arg in (args or []))

        inferred = defaultdict(int)
        for arg in (args or []):
            ty = _infer_semantic_type(arg)
            inferred[ty] += 1
            if ty.startswith("unit:"):
                inferred["unit"] += 1

        type_ok = True
        for ty, needed in required_type_counts.items():
            if inferred.get(ty, 0) < max(0, needed):
                type_ok = False
                violations.append(f"type:{ty}:need={needed}:have={inferred.get(ty, 0)}")

        compat_ok = True
        if "part_whole" in checks:
            compat_ok = compat_ok and _check_part_whole(tokens)
        if "temporal_causal" in checks:
            compat_ok = compat_ok and _check_temporal_causal(tokens)
        if "unit_compatibility" in checks:
            compat_ok = compat_ok and _check_unit_compatibility(tokens)
        if not compat_ok:
            violations.append(f"compatibility:{candidate.get('family', 'unknown')}:{pred_name}")

        if type_ok and compat_ok:
            pass_count += 1

    score = float(pass_count) / float(max(1, len(examples)))
    is_valid = (not missing_body) and score >= 1.0
    report = {
        "score": score,
        "violations": sorted(set(violations)),
        "checks": checks,
    }
    return is_valid, report


def generate_clause_candidates(
    predicates,
    roles,
    existing_hyp=None,
    action_constraint="none",
    allowed_role_pairs=None,
    allowed_metarule_types=None,
):
    """Generate candidate clause actions from the metarule action space."""
    existing = {str(c) for c in (existing_hyp or [])}
    candidates = []
    active_roles = constrain_roles_for_search(roles, action_constraint=action_constraint)
    allowed_pairs = {
        tuple(str(x).lower() for x in pair)
        for pair in (allowed_role_pairs or [])
        if len(pair) >= 2
    }
    allowed_types = {str(t) for t in (allowed_metarule_types or [])}
    allow_transfer = not allowed_types or "role_transfer" in allowed_types
    allow_pair = not allowed_types or "role_pair" in allowed_types
    if allow_transfer:
        for p_pred in predicates:
            for q_pred in predicates:
                if p_pred == q_pred:
                    continue
                for role_pred in active_roles:
                    clause = instantiate_metarule_role_transfer(p_pred, q_pred, role_pred)
                    is_dup = str(clause) in existing
                    if is_dup:
                        continue
                    candidates.append(
                        {
                            "clause": clause,
                            "metarule_type": "role_transfer",
                            "head_pred": q_pred,
                            "body_preds": [p_pred, role_pred],
                            "duplicate": is_dup,
                        }
                    )
    if allow_pair and len(active_roles) >= 2:
        for q_pred in predicates:
            for role_a in active_roles:
                for role_b in active_roles:
                    if role_a == role_b:
                        continue
                    if allowed_pairs and (str(role_a).lower(), str(role_b).lower()) not in allowed_pairs:
                        continue
                    clause = instantiate_metarule_role_pair(q_pred, role_a, role_b)
                    is_dup = str(clause) in existing
                    if is_dup:
                        continue
                    candidates.append(
                        {
                            "clause": clause,
                            "metarule_type": "role_pair",
                            "head_pred": q_pred,
                            "body_preds": [role_a, role_b],
                            "duplicate": is_dup,
                        }
                    )
    return candidates

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
try:
    import torch
    from logic_transformer.lorentz_math import project_to_hyperboloid, lorentz_distance
except Exception:
    torch = None

_EMB_STATE = None

def _load_aligned_embeddings(emb_path='logic_transformer/checkpoints/aligned_hyperbolic_embeddings_full.pt', vocab_source='logic_transformer/checkpoints/aligned_hyperbolic_embeddings_full_vocab.json'):
    """Attempt to load aligned embeddings and vocabulary mapping.
    Behavior:
      - If logic_transformer/local_checkpoints/{basename(emb_path)} exists, prefer it (local-only artifacts).
      - Otherwise fall back to emb_path.
    Returns a dict with keys 'weights' (Tensor), 'projected' (Tensor) and 'vocab' (dict token->idx) on success, else None.
    """
    import json
    from pathlib import Path
    global _EMB_STATE
    if _EMB_STATE is not None:
        return _EMB_STATE
    if torch is None:
        _EMB_STATE = None
        return None
    try:
        # prefer local_checkpoints if present
        local_dir = Path('logic_transformer') / 'local_checkpoints'
        emb_path_p = Path(emb_path)
        if local_dir.exists():
            local_emb = local_dir / emb_path_p.name
            local_vocab = local_dir / Path(vocab_source).name
            if local_emb.exists():
                emb_path_use = str(local_emb)
                vocab_use = str(local_vocab) if local_vocab.exists() else vocab_source
            else:
                emb_path_use = emb_path
                vocab_use = vocab_source
        else:
            emb_path_use = emb_path
            vocab_use = vocab_source

        if not os.path.exists(emb_path_use):
            _EMB_STATE = None
            return None

        emb_state = torch.load(emb_path_use, map_location='cpu')
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
            if os.path.exists(vocab_use):
                with open(vocab_use, 'r', encoding='utf-8') as fh:
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

# --- Semantic metarule grammar ---
# These seed meta-rules are intentionally compact. They are not the final solver;
# they provide the reusable abstractions that later ILP learning can generalize over.

class MetaRule:
    def __init__(self, family, name, head, body, calc, target, description="", constraints=None):
        self.family = family
        self.name = name
        self.head = head
        self.body = body
        self.calc = calc
        self.target = target
        self.description = description
        self.constraints = constraints or {}

    def instantiate(self, binding=None):
        head = self.head
        body = self.body
        if binding:
            def repl(val):
                return binding.get(val, val) if isinstance(val, str) and val.startswith("?") else val
            head = (repl(head[0]), [repl(v) for v in head[1]])
            body = [(p, [repl(v) for v in args]) for p, args in body]
        return {
            "family": self.family,
            "name": self.name,
            "head": head,
            "body": body,
            "calc": self.calc,
            "target": self.target,
            "description": self.description,
            "constraints": copy.deepcopy(self.constraints),
        }


class MetaRuleGrammar:
    @staticmethod
    def seeded_rules(include_agi_transfer_families=None):
        include_transfer = resolve_agi_transfer_toggle(include_agi_transfer_families)
        rules = [
            MetaRule(
                "ratio_compose",
                "ratio_compose",
                ("quantity", ["?owner", "?item", "?ans"]),
                [("ratio", ["?owner", "?ref_owner", "?item", "?factor"]), ("quantity", ["?ref_owner", "?item", "?base"])],
                "base * factor",
                "?ans",
                "same-item ratio composition from a reference owner",
            ),
            MetaRule(
                "remaining_quantity",
                "remaining_quantity",
                ("remaining_quantity", ["?owner", "?item", "?ans"]),
                [("quantity", ["?owner", "?item", "?base"]), ("percent", ["?owner", "?item", "?pct"])],
                "base * (1 - pct)",
                "?ans",
                "percent-of-remaining / remaining-quantity reduction",
            ),
            MetaRule(
                "state_transition",
                "state_transition",
                ("state", ["?owner", "?item", "?ans"]),
                [("state_transition", ["?owner", "?item", "?before", "?delta", "?after"])],
                "before + delta",
                "?ans",
                "same-owner/item state updates use before + delta",
            ),
            MetaRule(
                "latest_valid_state",
                "latest_valid_state",
                ("final_state", ["?owner", "?item", "?ans"]),
                [("state", ["?owner", "?item", "?later"]), ("state", ["?owner", "?item", "?earlier"])],
                "later",
                "?ans",
                "latest valid state wins over earlier temporally stale candidates",
            ),
            # Core numeric primitives below are compact anchors for broader GSM coverage.
            MetaRule(
                "rate_product",
                "rate_times_duration",
                ("rate_times_duration", ["?owner", "?item", "?ans"]),
                [("rate", ["?item", "?rate"]), ("quantity", ["?owner", "?duration", "?steps"])],
                "rate * steps",
                "?ans",
                "rate × duration/quantity pattern",
            ),
            MetaRule(
                "sum_total",
                "combined_total",
                ("combined_total", ["?owner", "?item", "?ans"]),
                [("quantity", ["?owner", "?item", "?left"]), ("quantity", ["?other", "?item", "?right"])],
                "left + right",
                "?ans",
                "sum of two quantities for same item",
            ),
            MetaRule(
                "unit_convert",
                "unit_convert",
                ("unit_convert", ["?value", "?from_unit", "?to_unit", "?ans"]),
                [("unitconvert", ["?value", "?from_unit", "?to_unit", "?result"])],
                "result",
                "?ans",
                "canonical unit conversion bridge",
            ),
            MetaRule(
                "percent_of_total",
                "percent_of_total",
                ("percent_of_total", ["?owner", "?item", "?ans"]),
                [("quantity", ["?owner", "?item", "?base"]), ("percent", ["?owner", "?item", "?pct"])],
                "base * pct",
                "?ans",
                "percent of base quantity",
            ),
            MetaRule(
                "ratio_of_total",
                "ratio_of_total",
                ("ratio_of_total", ["?owner", "?item", "?ans"]),
                [("quantity", ["?owner", "?item", "?part"]), ("quantity", ["?other", "?item", "?rest"])],
                "part / (part + rest)",
                "?ans",
                "ratio over a two-part total",
            ),
            MetaRule(
                "fraction_of_total",
                "fraction_of_total",
                ("fraction_of_total", ["?owner", "?item", "?ans"]),
                [("quantity", ["?owner", "?item", "?part"]), ("quantity", ["?group", "?item", "?total"])],
                "part / total",
                "?ans",
                "fraction of a known total",
            ),
            MetaRule(
                "percent_growth",
                "percent_growth",
                ("percent_growth", ["?owner", "?item", "?ans"]),
                [("quantity", ["?owner", "?item", "?base"]), ("percent", ["?owner", "?item", "?pct"])],
                "base * (1 + pct)",
                "?ans",
                "single growth step from a base quantity",
            ),
            MetaRule(
                "percent_sequence",
                "percent_sequence",
                ("percent_sequence", ["?owner", "?item", "?ans"]),
                [("quantity", ["?owner", "?item", "?base"]), ("percent", ["?owner", "?item", "?pct1"]), ("percent", ["?owner", "?item", "?pct2"])],
                "base * (1 + pct1) * (1 + pct2)",
                "?ans",
                "two-step percent sequence approximation",
            ),
            MetaRule(
                "average_speed",
                "average_speed",
                ("average_speed", ["?trip", "?ans"]),
                [("quantity", ["?trip", "distance", "?dist"]), ("quantity", ["?trip", "time", "?dur"])],
                "dist / dur",
                "?ans",
                "distance over total time",
            ),
            MetaRule(
                "simple_interest",
                "simple_interest",
                ("simple_interest", ["?bank", "?deposit", "?ans"]),
                [("quantity", ["?bank", "principal", "?principal"]), ("percent", ["?bank", "principal", "?rate"]), ("quantity", ["?bank", "years", "?years"])],
                "principal * rate * years",
                "?ans",
                "simple interest: P * r * t",
            ),
            MetaRule(
                "profit_cost",
                "profit_cost",
                ("profit_cost", ["?seller", "?product", "?ans"]),
                [("quantity", ["?seller", "sale_price", "?sale"]), ("percent", ["?seller", "sale_price", "?pct"])],
                "sale / (1 + pct)",
                "?ans",
                "recover original cost from sale price and profit percent",
            ),
        ]
        if include_transfer:
            rules.extend(
                [
                    MetaRule(
                        "part_whole_balance",
                        "part_whole_balance",
                        ("part_whole_balance", ["?part_entity", "?whole_entity", "?ans"]),
                        [
                            ("part_quantity", ["?part_entity", "?whole_entity", "?item", "?part_value"]),
                            ("quantity", ["?whole_entity", "?item", "?remaining_value"]),
                        ],
                        "part_value + remaining_value",
                        "?ans",
                        "part-whole aggregation with typed part/whole roles",
                        constraints={
                            "required_type_counts": {"part_entity": 1, "whole_entity": 1, "number": 1},
                            "required_body_predicates": ["part_quantity", "quantity"],
                            "compatibility_checks": ["part_whole"],
                        },
                    ),
                    MetaRule(
                        "temporal_causal_chain",
                        "temporal_causal_chain",
                        ("temporal_causal_chain", ["?owner", "?item", "?ans"]),
                        [
                            ("state", ["?owner", "?item", "?before"]),
                            ("causal_delta", ["?owner", "?item", "?cause", "?delta", "?after_marker"]),
                        ],
                        "before + delta",
                        "?ans",
                        "temporal-causal update from prior state and causal delta",
                        constraints={
                            "required_type_counts": {"temporal_marker": 1, "causal_marker": 1, "number": 1},
                            "required_body_predicates": ["state", "causal_delta"],
                            "compatibility_checks": ["temporal_causal"],
                        },
                    ),
                    MetaRule(
                        "unit_compatibility",
                        "unit_compatibility",
                        ("unit_compatibility", ["?value", "?from_unit", "?to_unit", "?ans"]),
                        [("unit_rate", ["?value", "?from_unit", "?to_unit", "?factor"])],
                        "value * factor",
                        "?ans",
                        "unit-aware conversion constrained by shared dimensions",
                        constraints={
                            "required_type_counts": {"unit": 2, "number": 1},
                            "required_body_predicates": ["unit_rate"],
                            "compatibility_checks": ["unit_compatibility"],
                        },
                    ),
                ]
            )
        return rules

    @staticmethod
    def infer_family(examples, include_agi_transfer_families=None):
        include_transfer = resolve_agi_transfer_toggle(include_agi_transfer_families)
        names = []
        for pred, args, expected in examples:
            if isinstance(pred, str):
                names.append(pred.lower())
            for arg in args or []:
                if isinstance(arg, str):
                    names.append(arg.lower())
        text = " ".join(names)
        if include_transfer:
            if "part_whole_balance" in text or ("part" in text and ("whole" in text or "total" in text)):
                return "part_whole_balance"
            if "temporal_causal_chain" in text or (("before" in text or "after" in text) and ("cause" in text or "delta" in text)):
                return "temporal_causal_chain"
            if "unit_compatibility" in text or ("unit_rate" in text and "convert" in text):
                return "unit_compatibility"
        if "rate" in text or "hour" in text or "duration" in text or "speed" in text:
            return "rate_product"
        if "ratio_of_total" in text:
            return "ratio_of_total"
        if "fraction_of_total" in text:
            return "fraction_of_total"
        if "percent_growth" in text:
            return "percent_growth"
        if "percent_sequence" in text:
            return "percent_sequence"
        if "average_speed" in text:
            return "average_speed"
        if "simple_interest" in text:
            return "simple_interest"
        if "profit_cost" in text:
            return "profit_cost"
        if "combined" in text or "total" in text or "sum" in text or "altogether" in text:
            return "sum_total"
        if "unit_convert" in text or "convert" in text or "dozen" in text:
            return "unit_convert"
        if "percent_of_total" in text:
            return "percent_of_total"
        if "ratio" in text or "times" in text:
            return "ratio_compose"
        if "remaining" in text or "percent" in text or "discount" in text:
            return "remaining_quantity"
        if "state" in text or "buy" in text or "donat" in text or "spoil" in text:
            return "latest_valid_state"
        return "state_transition"


class MetaRuleLearner:
    def __init__(self, grammar=None, include_agi_transfer_families=None):
        self.include_agi_transfer_families = resolve_agi_transfer_toggle(include_agi_transfer_families)
        self.grammar = grammar or MetaRuleGrammar.seeded_rules(
            include_agi_transfer_families=self.include_agi_transfer_families
        )

    def _rule_score(self, rule, examples):
        family = rule.family
        names = []
        for pred, args, _ in examples:
            names.append(str(pred).lower())
            for arg in args or []:
                if isinstance(arg, str):
                    names.append(arg.lower())
        text = " ".join(names)
        if family == "rate_product" and "rate" not in text and "hour" not in text and "duration" not in text:
            return 0.0
        if family == "ratio_of_total" and "ratio_of_total" not in text:
            return 0.0
        if family == "fraction_of_total" and "fraction_of_total" not in text:
            return 0.0
        if family == "percent_growth" and "percent_growth" not in text:
            return 0.0
        if family == "percent_sequence" and "percent_sequence" not in text:
            return 0.0
        if family == "average_speed" and "average_speed" not in text:
            return 0.0
        if family == "simple_interest" and "simple_interest" not in text:
            return 0.0
        if family == "profit_cost" and "profit_cost" not in text:
            return 0.0
        if family == "sum_total" and "total" not in text and "combined" not in text and "sum" not in text and "altogether" not in text:
            return 0.0
        if family == "unit_convert" and "convert" not in text and "dozen" not in text and "unit" not in text:
            return 0.0
        if family == "percent_of_total" and "percent" not in text:
            return 0.0
        if family == "ratio_compose" and "ratio" not in text and "times" not in text:
            return 0.0
        if family == "remaining_quantity" and "percent" not in text and "remaining" not in text and "discount" not in text:
            return 0.0
        if family == "latest_valid_state" and "buy" not in text and "spoil" not in text and "donat" not in text and "state" not in text:
            return 0.0
        if family == "state_transition" and "state" not in text and "buy" not in text and "give" not in text:
            return 0.0
        if family == "part_whole_balance" and ("part" not in text or ("whole" not in text and "total" not in text)):
            return 0.0
        if family == "temporal_causal_chain" and (("before" not in text and "after" not in text) or ("cause" not in text and "delta" not in text)):
            return 0.0
        if family == "unit_compatibility" and "unit" not in text and "convert" not in text and "hour" not in text and "minute" not in text:
            return 0.0
        constraint_ok, report = evaluate_candidate_constraints(rule.instantiate(), examples)
        if not constraint_ok:
            return 0.0
        return float(report.get("score", 1.0))

    def _instantiate_family_examples(self, family):
        if family == "rate_product":
            return [
                ("rate_times_duration", ["alice", "hours", 5.0], 50.0),
                ("rate_times_duration", ["bob", "hours", 3.0], 30.0),
            ]
        if family == "ratio_of_total":
            return [
                ("ratio_of_total", ["alice", "marbles", 0.4], 0.4),
                ("ratio_of_total", ["bob", "items", 0.25], 0.25),
            ]
        if family == "fraction_of_total":
            return [
                ("fraction_of_total", ["alice", "students", 0.4], 0.4),
                ("fraction_of_total", ["bob", "tickets", 0.5], 0.5),
            ]
        if family == "percent_growth":
            return [
                ("percent_growth", ["town", "residents", 600.0], 600.0),
                ("percent_growth", ["box", "items", 120.0], 120.0),
            ]
        if family == "percent_sequence":
            return [
                ("percent_sequence", ["town", "residents", 540.0], 540.0),
                ("percent_sequence", ["price", "jacket", 70.4], 70.4),
            ]
        if family == "average_speed":
            return [
                ("average_speed", ["trip", 60.0], 60.0),
                ("average_speed", ["route", 45.0], 45.0),
            ]
        if family == "simple_interest":
            return [
                ("simple_interest", ["bank", "deposit", 30.0], 30.0),
                ("simple_interest", ["bank", "deposit", 20.0], 20.0),
            ]
        if family == "profit_cost":
            return [
                ("profit_cost", ["seller", "product", 125.0], 125.0),
                ("profit_cost", ["vendor", "item", 80.0], 80.0),
            ]
        if family == "sum_total":
            return [
                ("combined_total", ["alice", "apples", 20.0], 20.0),
                ("combined_total", ["bob", "books", 14.0], 14.0),
            ]
        if family == "unit_convert":
            return [
                ("unit_convert", [3.0, "dozen", "egg", 36.0], 36.0),
                ("unit_convert", [2.0, "week", "day", 14.0], 14.0),
            ]
        if family == "percent_of_total":
            return [
                ("percent_of_total", ["alice", "apples", 10.0], 10.0),
                ("percent_of_total", ["bob", "books", 6.0], 6.0),
            ]
        if family == "ratio_compose":
            return [
                ("ratio_compose", ["alice", "bob", "apples", 2.0], 20.0),
                ("ratio_compose", ["carol", "dave", "apples", 1.5], 15.0),
            ]
        if family == "remaining_quantity":
            return [
                ("remaining_quantity", ["alice", "apples", 15.0], 15.0),
                ("remaining_quantity", ["alice", "apples", 12.0], 12.0),
            ]
        if family == "latest_valid_state":
            return [
                ("final_state", ["alice", "apples", 20.0], 20.0),
                ("final_state", ["alice", "apples", 17.0], 17.0),
            ]
        if family == "part_whole_balance":
            return [
                ("part_whole_balance", ["part_crate", "whole_crate", 30.0], 30.0),
                ("part_whole_balance", ["part_batch", "whole_batch", 18.0], 18.0),
            ]
        if family == "temporal_causal_chain":
            return [
                ("temporal_causal_chain", ["before_stage", "after_stage", "cause_addition", 14.0], 14.0),
                ("temporal_causal_chain", ["before_state", "after_state", "cause_loss", 9.0], 9.0),
            ]
        if family == "unit_compatibility":
            return [
                ("unit_compatibility", [3.0, "hour", "minute", 180.0], 180.0),
                ("unit_compatibility", [2.0, "day", "hour", 48.0], 48.0),
            ]
        return [
            ("state", ["alice", "apples", 8.0], 8.0),
            ("state", ["alice", "apples", 11.0], 11.0),
        ]

    def fit(self, examples):
        family = MetaRuleGrammar.infer_family(
            examples,
            include_agi_transfer_families=self.include_agi_transfer_families,
        )
        ranked = []
        for rule in self.grammar:
            if rule.family != family:
                continue
            score = self._rule_score(rule, examples)
            ranked.append((score, rule))
        if not ranked:
            self._instantiate_family_examples(
                MetaRuleGrammar.infer_family(
                    examples,
                    include_agi_transfer_families=self.include_agi_transfer_families,
                )
            )
            for rule in self.grammar:
                ranked.append((1.0, rule))
            return [rule for _, rule in sorted(ranked, key=lambda x: x[0], reverse=True)][:3]
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [rule for _, rule in ranked[:3]]

    def propose(self, examples):
        return [rule.instantiate() for rule in self.fit(examples)]


def generate_synthetic_family_examples(include_agi_transfer_families=None):
    """Generate graph-grounded synthetic NL examples for the seeded metarule families.

    The goal here is to keep the training set close to the semantic invariants we already
    understand instead of learning from broad, noisy raw NL. Each family is a compact rule
    pattern with multiple paraphrastic variants.
    """
    include_transfer = resolve_agi_transfer_toggle(include_agi_transfer_families)
    examples = {
        "rate_product": [
            ("rate_times_duration", ["seller", "hours", 5.0], 90.0),
            ("rate_times_duration", ["worker", "hours", 4.0], 72.0),
            ("value_per_unit", ["vendor", "hours", 2.0], 24.0),
            ("rate_times_duration", ["driver", "hours", 3.0], 60.0),
        ],
        "ratio_of_total": [
            ("ratio_of_total", ["unknown", "marbles", 0.4], 0.4),
            ("ratio_of_total", ["unknown", "balls", 0.25], 0.25),
            ("ratio_of_total", ["unknown", "apples", 0.5], 0.5),
            ("ratio_of_total", ["unknown", "cards", 0.6], 0.6),
        ],
        "fraction_of_total": [
            ("fraction_of_total", ["unknown", "students", 0.4], 0.4),
            ("fraction_of_total", ["unknown", "tickets", 0.5], 0.5),
            ("fraction_of_total", ["unknown", "books", 0.2], 0.2),
            ("fraction_of_total", ["unknown", "coins", 0.75], 0.75),
        ],
        "percent_growth": [
            ("percent_growth", ["town", "residents", 600.0], 600.0),
            ("percent_growth", ["school", "students", 240.0], 240.0),
            ("percent_growth", ["inventory", "items", 75.0], 75.0),
            ("percent_growth", ["box", "marbles", 30.0], 30.0),
        ],
        "percent_sequence": [
            ("percent_sequence", ["town", "residents", 540.0], 540.0),
            ("percent_sequence", ["jacket", "price", 70.4], 70.4),
            ("percent_sequence", ["fund", "balance", 110.0], 110.0),
            ("percent_sequence", ["store", "stock", 95.0], 95.0),
        ],
        "average_speed": [
            ("average_speed", ["trip", 60.0], 60.0),
            ("average_speed", ["route", 45.0], 45.0),
            ("average_speed", ["drive", 30.0], 30.0),
            ("average_speed", ["run", 12.0], 12.0),
        ],
        "simple_interest": [
            ("simple_interest", ["bank", "deposit", 30.0], 30.0),
            ("simple_interest", ["bank", "savings", 20.0], 20.0),
            ("simple_interest", ["credit_union", "fund", 18.0], 18.0),
            ("simple_interest", ["lender", "loan", 15.0], 15.0),
        ],
        "profit_cost": [
            ("profit_cost", ["seller", "product", 125.0], 125.0),
            ("profit_cost", ["vendor", "item", 80.0], 80.0),
            ("profit_cost", ["shop", "good", 50.0], 50.0),
            ("profit_cost", ["market", "bundle", 40.0], 40.0),
        ],
        "sum_total": [
            ("combined_total", ["alice", "apples", 20.0], 20.0),
            ("combined_total", ["alice", "books", 14.0], 14.0),
            ("combined_total", ["team", "points", 55.0], 55.0),
            ("combined_total", ["store", "items", 18.0], 18.0),
        ],
        "unit_convert": [
            ("unit_convert", [3.0, "dozen", "egg", 36.0], 36.0),
            ("unit_convert", [2.0, "week", "day", 14.0], 14.0),
            ("unit_convert", [4.0, "hour", "minute", 240.0], 240.0),
            ("unit_convert", [5.0, "meter", "centimeter", 500.0], 500.0),
        ],
        "percent_of_total": [
            ("percent_of_total", ["box", "marble", 10.0], 10.0),
            ("percent_of_total", ["class", "student", 12.0], 12.0),
            ("percent_of_total", ["town", "resident", 50.0], 50.0),
            ("percent_of_total", ["inventory", "item", 8.0], 8.0),
        ],
        "ratio_compose": [
            ("ratio_compose", ["alice", "bob", "apples", 2.0], 20.0),
            ("ratio_compose", ["charlie", "dana", "apples", 3.0], 30.0),
            ("ratio_compose", ["erin", "frank", "oranges", 0.5], 5.0),
            ("ratio_compose", ["grace", "heidi", "bananas", 1.5], 15.0),
        ],
        "remaining_quantity": [
            ("remaining_quantity", ["alice", "apples", 0.75], 15.0),
            ("remaining_quantity", ["alice", "apples", 0.5], 10.0),
            ("remaining_quantity", ["bob", "books", 0.8], 24.0),
            ("remaining_quantity", ["charlie", "tickets", 0.2], 6.0),
        ],
        "latest_valid_state": [
            ("final_state", ["alice", "apples", 20.0], 20.0),
            ("final_state", ["alice", "apples", 17.0], 17.0),
            ("final_state", ["bob", "mangoes", 13.0], 13.0),
            ("final_state", ["carol", "pencils", 9.0], 9.0),
        ],
        "state_transition": [
            ("state", ["alice", "apples", 8.0], 8.0),
            ("state", ["alice", "apples", 11.0], 11.0),
            ("state", ["bob", "books", 14.0], 14.0),
            ("state", ["carol", "pens", 7.0], 7.0),
        ],
    }
    if include_transfer:
        examples.update(
            {
                "part_whole_balance": [
                    ("part_whole_balance", ["part_carton", "whole_carton", 36.0], 36.0),
                    ("part_whole_balance", ["part_crate", "whole_crate", 18.0], 18.0),
                    ("part_whole_balance", ["part_bundle", "whole_bundle", 12.0], 12.0),
                    ("part_whole_balance", ["part_segment", "whole_segment", 24.0], 24.0),
                ],
                "temporal_causal_chain": [
                    ("temporal_causal_chain", ["before_stage", "after_stage", "cause_addition", 14.0], 14.0),
                    ("temporal_causal_chain", ["before_shift", "after_shift", "cause_loss", 9.0], 9.0),
                    ("temporal_causal_chain", ["before_step", "after_step", "cause_gain", 22.0], 22.0),
                    ("temporal_causal_chain", ["before_phase", "after_phase", "cause_drop", 16.0], 16.0),
                ],
                "unit_compatibility": [
                    ("unit_compatibility", [3.0, "hour", "minute", 180.0], 180.0),
                    ("unit_compatibility", [2.0, "day", "hour", 48.0], 48.0),
                    ("unit_compatibility", [5.0, "meter", "centimeter", 500.0], 500.0),
                    ("unit_compatibility", [4.0, "week", "day", 28.0], 28.0),
                ],
            }
        )
    return examples


def generate_synthetic_training_set(include_agi_transfer_families=None):
    dataset = []
    for family_examples in generate_synthetic_family_examples(
        include_agi_transfer_families=include_agi_transfer_families
    ).values():
        dataset.extend(family_examples)
    return dataset


# --- Beam search learner ---

def _learn_metagol_beam(
    predicates,
    roles,
    background,
    pos_examples,
    neg_examples,
    beam=5,
    max_clauses=3,
    prior_weight=0.5,
    action_constraint="none",
    binding_bias=0.0,
    preferred_heads=None,
    class_head_bias=0.0,
    nonclass_head_penalty=0.0,
    pair_bias=0.0,
    preferred_role_pairs=None,
    pair_match_bias=0.0,
    pair_miss_penalty=0.0,
    self_ref_penalty=0.0,
    allowed_role_pairs=None,
    allowed_metarule_types=None,
    preferred_local_roles=None,
    local_role_bias=0.0,
    preferred_body_preds=None,
    body_predicate_bias=0.0,
    body_predicate_miss_penalty=0.0,
    clause_cost=0.0,
    mirrored_pair_penalty_value=0.0,
    repeated_head_penalty_value=0.0,
    required_roles=None,
    required_role_bias=0.0,
    missing_required_role_penalty=0.0,
):
    # initial empty hypothesis
    beam_hypos = [([], 0.0)]  # list of (clauses_list, score)
    for step in range(max_clauses):
        candidates = []
        for hy, _ in beam_hypos:
            # expand by adding one clause instantiated from metarule templates
            for cand in generate_clause_candidates(
                predicates,
                roles,
                existing_hyp=hy,
                action_constraint=action_constraint,
                allowed_role_pairs=allowed_role_pairs,
                allowed_metarule_types=allowed_metarule_types,
            ):
                clause = cand["clause"]
                new_hyp = hy + [clause]
                sc = score_hypothesis(new_hyp, pos_examples, neg_examples, background)
                prior = embedding_prior(cand["head_pred"], cand["body_preds"])
                bind_bonus = semantic_binding_bonus(
                    cand["body_preds"],
                    action_constraint=action_constraint,
                    binding_bias=binding_bias,
                )
                pair_bonus = semantic_role_pair_bonus(
                    cand["body_preds"],
                    metarule_type=cand.get("metarule_type", ""),
                    pair_bias=pair_bias,
                )
                class_pair_bonus = preferred_role_pair_bonus(
                    cand["body_preds"],
                    metarule_type=cand.get("metarule_type", ""),
                    preferred_role_pairs=preferred_role_pairs,
                    pair_match_bias=pair_match_bias,
                    pair_miss_penalty=pair_miss_penalty,
                )
                head_bonus = head_preference_bonus(
                    cand.get("head_pred", ""),
                    preferred_heads=preferred_heads,
                    class_head_bias=class_head_bias,
                    nonclass_head_penalty=nonclass_head_penalty,
                )
                ref_penalty = self_reference_penalty(
                    cand.get("head_pred", ""),
                    cand.get("body_preds", []),
                    penalty=self_ref_penalty,
                )
                local_bonus = event_local_role_bonus(
                    cand.get("body_preds", []),
                    preferred_local_roles=preferred_local_roles,
                    local_role_bias=local_role_bias,
                )
                body_bonus = body_predicate_alignment_bonus(
                    cand.get("body_preds", []),
                    preferred_body_preds=preferred_body_preds,
                    body_predicate_bias=body_predicate_bias,
                    body_predicate_miss_penalty=body_predicate_miss_penalty,
                )
                compact_pen = clause_economy_penalty(len(new_hyp), clause_cost=clause_cost)
                role_coverage = required_role_coverage_bonus(
                    cand.get("body_preds", []),
                    required_roles=required_roles,
                    required_role_bias=required_role_bias,
                    missing_required_role_penalty=missing_required_role_penalty,
                )
                mirror_pen = mirrored_role_pair_penalty(
                    hy,
                    clause,
                    candidate_type=cand.get("metarule_type", ""),
                    penalty=mirrored_pair_penalty_value,
                )
                repeat_pen = repeated_head_penalty(
                    hy,
                    clause,
                    penalty=repeated_head_penalty_value,
                )
                total = sc + prior_weight * prior + bind_bonus + pair_bonus + class_pair_bonus + head_bonus + ref_penalty + local_bonus + body_bonus + compact_pen + mirror_pen + repeat_pen + role_coverage
                candidates.append((new_hyp, total))
        # keep top beam
        candidates.sort(key=lambda x: x[1], reverse=True)
        beam_hypos = candidates[:beam]
        # quick early exit if perfect coverage
        if beam_hypos and beam_hypos[0][1] > len(pos_examples) * 0.9:
            break
    best = beam_hypos[0] if beam_hypos else ([], 0.0)
    return best


def learn_metagol_nds(
    predicates,
    roles,
    background,
    pos_examples,
    neg_examples,
    beam=5,
    max_clauses=3,
    prior_weight=0.5,
    *,
    search_strategy="beam",
    reward_profile="balanced",
    value_model=None,
    value_learning_rate=0.02,
    value_discount=0.9,
    online_value_updates=True,
    action_constraint="none",
    binding_bias=0.0,
    preferred_heads=None,
    class_head_bias=0.0,
    nonclass_head_penalty=0.0,
    pair_bias=0.0,
    preferred_role_pairs=None,
    pair_match_bias=0.0,
    pair_miss_penalty=0.0,
    self_ref_penalty=0.0,
    allowed_role_pairs=None,
    allowed_metarule_types=None,
    preferred_local_roles=None,
    local_role_bias=0.0,
    preferred_body_preds=None,
    body_predicate_bias=0.0,
    body_predicate_miss_penalty=0.0,
    clause_cost=0.0,
    mirrored_pair_penalty_value=0.0,
    repeated_head_penalty_value=0.0,
    required_roles=None,
    required_role_bias=0.0,
    missing_required_role_penalty=0.0,
    trace_out=None,
):
    """Learn a clause set for the target using either beam or value-guided search.

    `trace_out` can be an empty list supplied by caller; this function appends
    per-step diagnostics when strategy supports it.
    """
    strategy = str(search_strategy or "beam").lower()
    if strategy == "beam":
        return _learn_metagol_beam(
            predicates,
            roles,
            background,
            pos_examples,
            neg_examples,
            beam=beam,
            max_clauses=max_clauses,
            prior_weight=prior_weight,
            action_constraint=action_constraint,
            binding_bias=binding_bias,
            preferred_heads=preferred_heads,
            class_head_bias=class_head_bias,
            nonclass_head_penalty=nonclass_head_penalty,
            pair_bias=pair_bias,
            preferred_role_pairs=preferred_role_pairs,
            pair_match_bias=pair_match_bias,
            pair_miss_penalty=pair_miss_penalty,
            self_ref_penalty=self_ref_penalty,
            allowed_role_pairs=allowed_role_pairs,
            allowed_metarule_types=allowed_metarule_types,
            preferred_local_roles=preferred_local_roles,
            local_role_bias=local_role_bias,
            preferred_body_preds=preferred_body_preds,
            body_predicate_bias=body_predicate_bias,
            body_predicate_miss_penalty=body_predicate_miss_penalty,
            clause_cost=clause_cost,
            mirrored_pair_penalty_value=mirrored_pair_penalty_value,
            repeated_head_penalty_value=repeated_head_penalty_value,
            required_roles=required_roles,
            required_role_bias=required_role_bias,
            missing_required_role_penalty=missing_required_role_penalty,
        )

    if strategy != "value_guided":
        raise ValueError(f"Unknown search_strategy: {search_strategy}")

    # Lazy import avoids circular imports when value-guided module imports this file.
    from ilp_learner.value_guided_search import learn_with_value_guided_search

    best_hyp, best_score, trace = learn_with_value_guided_search(
        predicates=predicates,
        roles=roles,
        background=background,
        pos_examples=pos_examples,
        neg_examples=neg_examples,
        beam=beam,
        max_clauses=max_clauses,
        prior_weight=prior_weight,
        reward_profile=reward_profile,
        value_model=value_model,
        value_learning_rate=value_learning_rate,
        value_discount=value_discount,
        online_value_updates=online_value_updates,
        action_constraint=action_constraint,
        binding_bias=binding_bias,
        preferred_heads=preferred_heads,
        class_head_bias=class_head_bias,
        nonclass_head_penalty=nonclass_head_penalty,
        pair_bias=pair_bias,
        preferred_role_pairs=preferred_role_pairs,
        pair_match_bias=pair_match_bias,
        pair_miss_penalty=pair_miss_penalty,
        self_ref_penalty=self_ref_penalty,
        allowed_role_pairs=allowed_role_pairs,
        allowed_metarule_types=allowed_metarule_types,
        preferred_local_roles=preferred_local_roles,
        local_role_bias=local_role_bias,
        preferred_body_preds=preferred_body_preds,
        body_predicate_bias=body_predicate_bias,
        body_predicate_miss_penalty=body_predicate_miss_penalty,
        clause_cost=clause_cost,
        mirrored_pair_penalty_value=mirrored_pair_penalty_value,
        repeated_head_penalty_value=repeated_head_penalty_value,
        required_roles=required_roles,
        required_role_bias=required_role_bias,
        missing_required_role_penalty=missing_required_role_penalty,
    )
    if isinstance(trace_out, list):
        trace_out.extend(trace)
    return best_hyp, best_score


def demo_meta_rule_learning():
    learner = MetaRuleLearner()
    examples = [
        ("ratio_compose", ["alice", "bob", "apples", 2.0], 20.0),
        ("ratio_compose", ["carol", "dave", "apples", 3.0], 30.0),
    ]
    learned = learner.propose(examples)
    print("[Meta rule learner] learned candidates:")
    for rule in learned:
        print(rule)


# --- Demo runner ---
if __name__ == '__main__':
    demo_meta_rule_learning()
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
