"""Generate ILP tasks from GSM8K LT exports and run learner per task."""
import argparse
import json
from pathlib import Path

from ilp_learner.metagol_nds import Atom, forward_chain, learn_metagol_nds
from ilp_learner.value_model import LinearValueModel
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

def parse_args():
    parser = argparse.ArgumentParser(description="Run ILP learner on a GSM8K LT sample.")
    parser.add_argument("--max-examples", type=int, default=MAX_EXAMPLES)
    parser.add_argument("--beam", type=int, default=6)
    parser.add_argument("--max-clauses", type=int, default=2)
    parser.add_argument("--prior-weight", type=float, default=0.5)
    parser.add_argument("--search-strategy", choices=["beam", "value_guided"], default="beam")
    parser.add_argument("--reward-profile", choices=["conservative", "balanced", "aggressive"], default="balanced")
    parser.add_argument("--value-model-path", default="", help="Optional JSON model file for value-guided scoring.")
    parser.add_argument("--trace-limit", type=int, default=8, help="Max trace rows stored per example.")
    return parser.parse_args()

def var_to_const(v):
    # convert '?x3' -> 'x3'
    if isinstance(v, str) and v.startswith('?'):
        return v[1:]
    return str(v)

def main():
    args = parse_args()
    results = []
    uncertain_for_review = []
    skip_counts = {"neutral": 0, "low_confidence": 0}
    value_model = LinearValueModel.load(args.value_model_path) if args.value_model_path else None

    with IN_PATH.open() as f:
        for i, line in enumerate(f):
            if i >= args.max_examples:
                break
            obj = json.loads(line)
            props = obj.get("input_props", [])
            q_events = [p for p in props if p.get("pred") == "query"]
            if q_events:
                q_ev = q_events[0]["args"][0]
            else:
                evs = [p for p in props if p.get("pred") == "event"]
                if not evs:
                    continue
                q_ev = evs[0]["args"][0]
            verbs = [p for p in props if p.get("pred") == "predicate" and p.get("args", [None])[0] == q_ev]
            if not verbs:
                continue
            verb_arg = verbs[0]["args"][1]
            if isinstance(verb_arg, str) and verb_arg.startswith("?"):
                continue
            verb = str(verb_arg)
            info = ontology_classify_details(verb)
            cls = str(info["label"])
            conf = float(info["confidence"])
            reason = str(info["reason"])
            if cls == "neutral":
                skip_counts["neutral"] += 1
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
                skip_counts["low_confidence"] += 1
                continue

            background = []
            predicates = set()
            roles = set()
            for p in props:
                pred = p.get("pred")
                args_p = p.get("args", [])
                if pred == "predicate":
                    ev = args_p[0]
                    verb_a = args_p[1]
                    if isinstance(verb_a, str) and verb_a.startswith("?"):
                        continue
                    pred_name = f"predicate_{str(verb_a)}"
                    predicates.add(pred_name)
                    background.append(Atom(pred_name, [ev]))
                elif len(args_p) >= 2:
                    ev = args_p[0]
                    arg = var_to_const(args_p[1])
                    roles.add(pred)
                    background.append(Atom(pred, [ev, arg]))

            target = f"class_{cls}"
            predicates.add(target)
            pos = [Atom(target, [q_ev])]
            neg = []
            if not roles:
                roles = {"agent", "patient"}
            roles = list(roles)
            predicates = list(predicates)
            trace = []
            try:
                best_hyp, score = learn_metagol_nds(
                    predicates,
                    roles,
                    background,
                    pos,
                    neg,
                    beam=args.beam,
                    max_clauses=args.max_clauses,
                    prior_weight=args.prior_weight,
                    search_strategy=args.search_strategy,
                    reward_profile=args.reward_profile,
                    value_model=value_model,
                    trace_out=trace,
                )
            except Exception:
                best_hyp, score = [], 0.0
                trace = []
            inferred = forward_chain(set(background), best_hyp)
            res = {
                "index": i,
                "id": obj.get("id"),
                "verb": verb,
                "class": cls,
                "class_confidence": conf,
                "class_reason": reason,
                "search_strategy": args.search_strategy,
                "reward_profile": args.reward_profile,
                "score": float(score),
                "num_clauses": len(best_hyp),
                "clauses": [str(c) for c in best_hyp],
                "inferred": [str(a) for a in sorted(inferred, key=lambda x: (x.pred, x.args))],
            }
            if trace:
                res["search_trace"] = trace[: max(0, int(args.trace_limit))]
            print(f"[{i}] id={obj.get('id')} verb={verb} class={cls} score={score:.4f} clauses={res['clauses']}")
            results.append(res)

    with OUT_PATH.open("w") as f:
        json.dump(results, f, indent=2)
    with GUIDANCE_PATH.open("w") as f:
        json.dump(
            {
                "min_class_confidence": MIN_CLASS_CONFIDENCE,
                "search_strategy": args.search_strategy,
                "reward_profile": args.reward_profile,
                "skip_counts": skip_counts,
                "uncertain_for_review": uncertain_for_review,
            },
            f,
            indent=2,
        )
    print("Saved GSM8K ILP results to", OUT_PATH)
    print("Saved uncertain-label review queue to", GUIDANCE_PATH)


if __name__ == "__main__":
    main()
