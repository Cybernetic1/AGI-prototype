"""Generate ILP tasks from GSM8K LT exports and run learner per task.
For each example, we:
 - determine query event (from 'query' prop or '?e1')
 - build background atoms: predicate_VERB(e), role(e, arg_const)
 - set class label from ontology_classify(VERB) -> class_transfer/creation/loss/comparison/neutral
 - run learn_metagol_nds to try to learn a mapping to the class label
 - save results to ilp_learner/gsm8k_ilp_results.json
"""
import json
from pathlib import Path
from ilp_learner.metagol_nds import Atom, learn_metagol_nds, forward_chain
from forward_engine.phase3_synthetic_test import ontology_classify_details

DATA_CANDIDATES = [
    Path('gsm8k-tests/lt_core/data/gsm8k_train_lt.jsonl'),
    Path('logic_transformer/data/gsm8k_train_lt.jsonl'),
    Path('gsm8k-tests-backup/lt_core/data/gsm8k_train_lt.jsonl'),
]
IN_PATH = next((p for p in DATA_CANDIDATES if p.exists()), DATA_CANDIDATES[0])
OUT_PATH = Path('ilp_learner/gsm8k_ilp_results.json')
GUIDANCE_PATH = Path('ilp_learner/gsm8k_ilp_uncertain_for_review.json')
MAX_EXAMPLES = 30
MIN_CLASS_CONFIDENCE = 0.70

results = []
uncertain_for_review = []
skip_counts = {"neutral": 0, "low_confidence": 0}

def var_to_const(v):
    # convert '?x3' -> 'x3'
    if isinstance(v, str) and v.startswith('?'):
        return v[1:]
    return str(v)

with IN_PATH.open() as f:
    for i, line in enumerate(f):
        if i >= MAX_EXAMPLES:
            break
        obj = json.loads(line)
        props = obj.get('input_props', [])
        # find query event
        q_events = [p for p in props if p.get('pred') == 'query']
        if q_events:
            q_ev = q_events[0]['args'][0]
        else:
            # pick first event in props
            evs = [p for p in props if p.get('pred') == 'event']
            if not evs:
                continue
            q_ev = evs[0]['args'][0]
        # collect verbs on this event
        verbs = [p for p in props if p.get('pred') == 'predicate' and p.get('args', [None])[0] == q_ev]
        if not verbs:
            # skip if no predicate verb on query event
            continue
        verb_arg = verbs[0]['args'][1]
        if isinstance(verb_arg, str) and verb_arg.startswith('?'):
            # unknown verb
            continue
        verb = str(verb_arg)
        info = ontology_classify_details(verb)
        cls = str(info["label"])
        conf = float(info["confidence"])
        reason = str(info["reason"])
        if cls == 'neutral':
            # skip unlabeled
            skip_counts["neutral"] += 1
            continue
        if conf < MIN_CLASS_CONFIDENCE:
            # Ask-human guidance queue: keep uncertain labels for manual review.
            uncertain_for_review.append(
                {
                    "index": i,
                    "id": obj.get("id"),
                    "verb": verb,
                    "proposed_class": cls,
                    "confidence": conf,
                    "reason": reason,
                }
            )
            skip_counts["low_confidence"] += 1
            continue
        # build background
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
                # role or other; treat as role name with event and arg const
                if len(args) >= 2:
                    ev = args[0]
                    arg = var_to_const(args[1])
                    roles.add(pred)
                    background.append(Atom(pred, [ev, arg]))
        # add class label as target head predicate
        target = f'class_{cls}'
        predicates.add(target)
        pos = [Atom(target, [q_ev])]
        neg = []
        # ensure roles list is non-empty; fallback to common ones
        if not roles:
            roles = {'agent', 'patient'}
        roles = list(roles)
        predicates = list(predicates)
        # run learner
        try:
            best_hyp, score = learn_metagol_nds(predicates, roles, background, pos, neg, beam=6, max_clauses=2)
        except Exception as e:
            best_hyp, score = [], 0.0
        inferred = forward_chain(set(background), best_hyp)
        res = {
            'index': i,
            'id': obj.get('id'),
            'verb': verb,
            'class': cls,
            'class_confidence': conf,
            'class_reason': reason,
            'score': float(score),
            'num_clauses': len(best_hyp),
            'clauses': [str(c) for c in best_hyp],
            'inferred': [str(a) for a in sorted(inferred, key=lambda x: (x.pred, x.args))]
        }
        print(f"[{i}] id={obj.get('id')} verb={verb} class={cls} score={score:.4f} clauses={res['clauses']}")
        results.append(res)

with OUT_PATH.open('w') as f:
    json.dump(results, f, indent=2)
with GUIDANCE_PATH.open("w") as f:
    json.dump(
        {
            "min_class_confidence": MIN_CLASS_CONFIDENCE,
            "skip_counts": skip_counts,
            "uncertain_for_review": uncertain_for_review,
        },
        f,
        indent=2,
    )
print('Saved GSM8K ILP results to', OUT_PATH)
print('Saved uncertain-label review queue to', GUIDANCE_PATH)
