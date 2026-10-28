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
    def __init__(self, family, name, head, body, calc, target, description=""):
        self.family = family
        self.name = name
        self.head = head
        self.body = body
        self.calc = calc
        self.target = target
        self.description = description

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
        }


class MetaRuleGrammar:
    @staticmethod
    def seeded_rules():
        return [
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

    @staticmethod
    def infer_family(examples):
        names = []
        for pred, args, expected in examples:
            if isinstance(pred, str):
                names.append(pred.lower())
        text = " ".join(names)
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
    def __init__(self, grammar=None):
        self.grammar = grammar or MetaRuleGrammar.seeded_rules()

    def _rule_score(self, rule, examples):
        family = rule.family
        text = " ".join(str(pred).lower() for pred, _, _ in examples)
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
        return 1.0

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
        return [
            ("state", ["alice", "apples", 8.0], 8.0),
            ("state", ["alice", "apples", 11.0], 11.0),
        ]

    def fit(self, examples):
        family = MetaRuleGrammar.infer_family(examples)
        ranked = []
        for rule in self.grammar:
            if rule.family != family:
                continue
            score = self._rule_score(rule, examples)
            ranked.append((score, rule))
        if not ranked:
            synthetic = self._instantiate_family_examples(MetaRuleGrammar.infer_family(examples))
            for rule in self.grammar:
                ranked.append((1.0, rule))
            return [rule for _, rule in sorted(ranked, key=lambda x: x[0], reverse=True)][:3]
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [rule for _, rule in ranked[:3]]

    def propose(self, examples):
        return [rule.instantiate() for rule in self.fit(examples)]


def generate_synthetic_family_examples():
    """Generate graph-grounded synthetic NL examples for the seeded metarule families.

    The goal here is to keep the training set close to the semantic invariants we already
    understand instead of learning from broad, noisy raw NL. Each family is a compact rule
    pattern with multiple paraphrastic variants.
    """
    return {
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


def generate_synthetic_training_set():
    dataset = []
    for family_examples in generate_synthetic_family_examples().values():
        dataset.extend(family_examples)
    return dataset


# --- Beam search learner ---

def _learn_metagol_beam(predicates, roles, background, pos_examples, neg_examples, beam=5, max_clauses=3, prior_weight=0.5):
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
                        total = sc + prior_weight * prior
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
