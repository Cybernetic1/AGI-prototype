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
CLASS_LABELS = ("transfer", "creation", "loss", "comparison")

def parse_args():
    parser = argparse.ArgumentParser(description="Run ILP learner on a GSM8K LT sample.")
    parser.add_argument("--max-examples", type=int, default=MAX_EXAMPLES)
    parser.add_argument("--start-index", type=int, default=0, help="Skip rows before this dataset index.")
    parser.add_argument("--beam", type=int, default=6)
    parser.add_argument("--max-clauses", type=int, default=2)
    parser.add_argument("--prior-weight", type=float, default=0.5)
    parser.add_argument("--search-strategy", choices=["beam", "value_guided"], default="beam")
    parser.add_argument("--reward-profile", choices=["conservative", "balanced", "aggressive"], default="balanced")
    parser.add_argument("--value-model-path", default="", help="Optional JSON model file for value-guided scoring.")
    parser.add_argument("--save-value-model-path", default="", help="Optional output path to persist updated value model.")
    parser.add_argument("--value-lr", type=float, default=0.02, help="Learning rate for online value updates.")
    parser.add_argument("--value-discount", type=float, default=0.9, help="Discount factor for online value updates.")
    parser.add_argument("--no-online-value-updates", action="store_true", help="Disable online value-model updates.")
    parser.add_argument("--downstream-reward", action="store_true", help="Use downstream numeric solve signal to update value model.")
    parser.add_argument("--downstream-max-cycles", type=int, default=2, help="Cycles for downstream benchmark_single_problem checks.")
    parser.add_argument("--downstream-reward-weight", type=float, default=1.0, help="Scale for downstream reward contribution.")
    parser.add_argument("--downstream-unsolved-penalty", type=float, default=-1.0, help="Reward value when downstream solve fails.")
    parser.add_argument("--downstream-reward-mode", choices=["binary", "distance"], default="distance", help="How to compute downstream reward.")
    parser.add_argument("--downstream-distance-scale", type=float, default=1.0, help="Scale factor for distance-based reward normalization.")
    parser.add_argument("--downstream-class-source", choices=["ontology", "ilp"], default="ilp", help="Class label source for downstream benchmark.")
    parser.add_argument("--class-competition", action="store_true", help="Include all class_* heads in predicate search space.")
    parser.add_argument("--class-supervision", choices=["strict", "none"], default="strict", help="Whether to supervise ILP target class during search.")
    parser.add_argument("--trace-limit", type=int, default=8, help="Max trace rows stored per example.")
    return parser.parse_args()

def var_to_const(v):
    # convert '?x3' -> 'x3'
    if isinstance(v, str) and v.startswith('?'):
        return v[1:]
    return str(v)


def _downstream_reward_value(bench, args):
    solved = bool(bench.get("solved"))
    if solved:
        return float(args.downstream_reward_weight)
    if args.downstream_reward_mode == "binary":
        return float(args.downstream_unsolved_penalty) * float(args.downstream_reward_weight)
    expected = bench.get("expected")
    belief_values = bench.get("belief_values") or []
    if expected is None or not belief_values:
        return float(args.downstream_unsolved_penalty) * float(args.downstream_reward_weight)
    errs = [abs(float(v) - float(expected)) for v in belief_values]
    min_err = min(errs) if errs else None
    if min_err is None:
        return float(args.downstream_unsolved_penalty) * float(args.downstream_reward_weight)
    norm = max(1.0, abs(float(expected))) * max(1e-6, float(args.downstream_distance_scale))
    closeness = max(0.0, 1.0 - (min_err / norm))
    # map closeness [0,1] to reward [-1,1]
    shaped = (2.0 * closeness) - 1.0
    return shaped * float(args.downstream_reward_weight)


def _extract_ilp_class(best_hyp):
    for clause in best_hyp or []:
        try:
            head_pred = getattr(getattr(clause, "head", None), "pred", "")
        except Exception:
            head_pred = ""
        if isinstance(head_pred, str) and head_pred.startswith("class_"):
            return head_pred[len("class_") :]
    return None

def main():
    args = parse_args()
    results = []
    uncertain_for_review = []
    skip_counts = {"neutral": 0, "low_confidence": 0}
    value_model = None
    if args.search_strategy == "value_guided":
        value_model = LinearValueModel.load(args.value_model_path) if args.value_model_path else LinearValueModel.seeded()

    with IN_PATH.open() as f:
        for i, line in enumerate(f):
            if i < args.start_index:
                continue
            if i >= args.start_index + args.max_examples:
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
            if args.class_competition:
                for c in CLASS_LABELS:
                    predicates.add(f"class_{c}")
            else:
                predicates.add(target)
            pos = [Atom(target, [q_ev])] if args.class_supervision == "strict" else []
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
                    value_learning_rate=args.value_lr,
                    value_discount=args.value_discount,
                    online_value_updates=not args.no_online_value_updates,
                    trace_out=trace,
                )
            except Exception:
                best_hyp, score = [], 0.0
                trace = []
            inferred = forward_chain(set(background), best_hyp)
            downstream_solved = None
            downstream_reward_value = None
            if args.downstream_reward:
                from cognitive_loop import benchmark_single_problem

                question = obj.get("text") or obj.get("question") or ""
                downstream_class = cls
                ilp_class = _extract_ilp_class(best_hyp)
                if args.downstream_class_source == "ilp" and ilp_class:
                    downstream_class = ilp_class
                bench = benchmark_single_problem(
                    question,
                    obj.get("answer"),
                    max_cycles=args.downstream_max_cycles,
                    quiet=True,
                    input_props=props,
                    class_label=downstream_class,
                    use_ltm_recall=False,
                    auto_accept_recall=False,
                    write_ltm_episode=False,
                )
                downstream_solved = bool(bench.get("solved"))
                downstream_reward_value = _downstream_reward_value(bench, args)
                if args.search_strategy == "value_guided" and value_model is not None and trace:
                    for step in trace:
                        feats = step.get("features")
                        if not feats:
                            continue
                        td_target = float(step.get("td_target", 0.0))
                        value_model.update(feats, td_target + float(downstream_reward_value), lr=args.value_lr)

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
            if downstream_solved is not None:
                res["downstream_solved"] = downstream_solved
            if downstream_reward_value is not None:
                res["downstream_reward_value"] = float(downstream_reward_value)
            if args.downstream_reward:
                res["downstream_class_source"] = args.downstream_class_source
                res["downstream_class_used"] = downstream_class
                res["ilp_predicted_class"] = ilp_class
            if trace:
                # Keep trace compact in output; do not persist full feature vectors.
                trimmed = []
                for step in trace[: max(0, int(args.trace_limit))]:
                    step_copy = dict(step)
                    step_copy.pop("features", None)
                    trimmed.append(step_copy)
                res["search_trace"] = trimmed
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
    if args.search_strategy == "value_guided" and args.save_value_model_path and value_model is not None:
        value_model.save(args.save_value_model_path)
        print("Saved value model to", args.save_value_model_path)


if __name__ == "__main__":
    main()
