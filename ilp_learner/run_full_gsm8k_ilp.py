"""Run ILP benchmark over the full GSM8K LT train set and time the run.
Saves results to ilp_learner/gsm8k_ilp_results_full.json
"""
import json
import time
from pathlib import Path
from ilp_learner.metagol_nds import Atom, learn_metagol_nds, forward_chain
from forward_engine.phase3_synthetic_test import ontology_classify_details

DATA_CANDIDATES = [
    Path('gsm8k-tests/lt_core/data/gsm8k_train_lt.jsonl'),
    Path('logic_transformer/data/gsm8k_train_lt.jsonl'),
    Path('gsm8k-tests-backup/lt_core/data/gsm8k_train_lt.jsonl'),
]
IN_PATH = next((p for p in DATA_CANDIDATES if p.exists()), DATA_CANDIDATES[0])
OUT_PATH = Path('ilp_learner/gsm8k_ilp_results_full.json')
GUIDANCE_PATH = Path('ilp_learner/gsm8k_ilp_uncertain_full.json')
MIN_CLASS_CONFIDENCE = 0.70

results = []
uncertain_for_review = []
count = 0
processed = 0
skip_neutral = 0
skip_low_confidence = 0
start = time.time()
with IN_PATH.open() as f:
    for i, line in enumerate(f):
        count += 1
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
        # collect verbs on this event
        verbs = [p for p in props if p.get('pred') == 'predicate' and p.get('args', [None])[0] == q_ev]
        if not verbs:
            continue
        verb_arg = verbs[0]['args'][1]
        if isinstance(verb_arg, str) and verb_arg.startswith('?'):
            continue
        verb = str(verb_arg)
        info = ontology_classify_details(verb)
        cls = str(info["label"])
        conf = float(info["confidence"])
        reason = str(info["reason"])
        if cls == 'neutral':
            skip_neutral += 1
            continue
        if conf < MIN_CLASS_CONFIDENCE:
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
            skip_low_confidence += 1
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
        neg = []
        if not roles:
            roles = {'agent', 'patient'}
        roles = list(roles)
        predicates = list(predicates)
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
        results.append(res)
        processed += 1
        if processed % 100 == 0:
            print(f"Processed {processed} labeled examples (total scanned {count})...")
end = time.time()
duration = end - start
with OUT_PATH.open('w') as f:
    json.dump(
        {
            'duration_sec': duration,
            'scanned': count,
            'processed': processed,
            'min_class_confidence': MIN_CLASS_CONFIDENCE,
            'skip_neutral': skip_neutral,
            'skip_low_confidence': skip_low_confidence,
            'uncertain_review_count': len(uncertain_for_review),
            'results': results,
        },
        f,
        indent=2,
    )
with GUIDANCE_PATH.open("w") as f:
    json.dump(
        {
            "min_class_confidence": MIN_CLASS_CONFIDENCE,
            "uncertain_for_review": uncertain_for_review,
        },
        f,
        indent=2,
    )
print(f"Done. Scanned {count} examples, processed {processed} labeled examples in {duration:.2f}s. Results saved to {OUT_PATH}")
print(f"Uncertain-label review queue saved to {GUIDANCE_PATH}")
