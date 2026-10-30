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
ROLE_ALIASES = {
    "beneficiary": "recipient",
    "payee": "recipient",
    "receiver": "recipient",
    "recepient": "recipient",
    "subject": "agent",
    "object": "patient",
}
TRANSFER_OUTFLOW_VERBS = {
    "pay",
    "paid",
    "spend",
    "spent",
    "sell",
    "sold",
    "give",
    "gave",
    "donate",
    "donated",
}
TRANSFER_INFLOW_VERBS = {
    "receive",
    "received",
    "collect",
    "collected",
    "refund",
    "refunded",
}
VERB_ROLE_PRIORS = {
    "pay": [("agent", "recipient")],
    "paid": [("agent", "recipient")],
    "spend": [("agent", "recipient")],
    "spent": [("agent", "recipient")],
    "sell": [("agent", "recipient")],
    "sold": [("agent", "recipient")],
    "give": [("agent", "recipient")],
    "gave": [("agent", "recipient")],
    "donate": [("agent", "recipient")],
    "donated": [("agent", "recipient")],
    "receive": [("recipient", "agent")],
    "received": [("recipient", "agent")],
    "collect": [("recipient", "agent")],
    "collected": [("recipient", "agent")],
    "refund": [("recipient", "agent")],
    "refunded": [("recipient", "agent")],
}


def _canonical_role_name(pred):
    name = str(pred or "").strip().lower()
    return ROLE_ALIASES.get(name, name)


def _transfer_flow_mode_from_verb(verb):
    v = str(verb or "").strip().lower()
    if v in TRANSFER_OUTFLOW_VERBS:
        return "outflow"
    if v in TRANSFER_INFLOW_VERBS:
        return "inflow"
    return "unknown"


def _preferred_role_pairs_for_class(class_label, observed_roles=None, verb=None):
    label = str(class_label or "").lower()
    observed = {str(r).lower() for r in (observed_roles or [])}
    verb_key = str(verb or "").strip().lower()
    verb_prior_pairs = VERB_ROLE_PRIORS.get(verb_key, [])
    flow = _transfer_flow_mode_from_verb(verb)
    if label == "transfer":
        if verb_prior_pairs:
            return [tuple(pair) for pair in verb_prior_pairs]
        if "recipient" in observed:
            if flow == "outflow":
                return [("agent", "recipient")]
            if flow == "inflow":
                return [("recipient", "agent")]
            return [("agent", "recipient"), ("recipient", "agent")]
        if flow == "outflow":
            return [("agent", "patient")]
        if flow == "inflow":
            return [("patient", "agent")]
        return [("agent", "patient"), ("patient", "agent")]
    if label in {"creation", "loss", "comparison"}:
        return [("agent", "patient"), ("patient", "agent")]
    return []


def _augment_transfer_recipient_roles(
    background,
    roles,
    cls,
    *,
    transfer_confidence=1.0,
    verb=None,
    min_confidence=0.8,
    require_transfer_cue=True,
):
    if str(cls).lower() != "transfer":
        return 0
    if float(transfer_confidence) < float(min_confidence):
        return 0
    flow = _transfer_flow_mode_from_verb(verb)
    if require_transfer_cue and flow == "unknown":
        return 0
    role_set = set(roles)
    if "recipient" in role_set:
        return 0
    by_event = {}
    for atom in list(background):
        if atom.pred not in {"agent", "patient", "recipient"} or len(atom.args) < 2:
            continue
        ev = atom.args[0]
        by_event.setdefault(ev, {}).setdefault(atom.pred, set()).add(atom.args[1])
    recovered = 0
    for ev, payload in by_event.items():
        if payload.get("recipient"):
            continue
        for arg in payload.get("patient", set()):
            background.append(Atom("recipient", [ev, arg]))
            role_set.add("recipient")
            recovered += 1
    roles.clear()
    roles.update(role_set)
    return recovered


def _query_event_role_set(props, query_event):
    roles = set()
    for p in props:
        pred = _canonical_role_name(p.get("pred"))
        args_p = p.get("args", [])
        if len(args_p) < 2 or args_p[0] != query_event:
            continue
        if pred in {"predicate", "event", "query", "question"}:
            continue
        roles.add(pred)
    return roles

def parse_args():
    parser = argparse.ArgumentParser(description="Run ILP learner on a GSM8K LT sample.")
    parser.add_argument("--max-examples", type=int, default=MAX_EXAMPLES)
    parser.add_argument("--start-index", type=int, default=0, help="Skip rows before this dataset index.")
    parser.add_argument("--beam", type=int, default=6)
    parser.add_argument("--max-clauses", type=int, default=2)
    parser.add_argument("--prior-weight", type=float, default=0.5)
    parser.add_argument("--search-strategy", choices=["beam", "value_guided"], default="beam")
    parser.add_argument("--action-constraint", choices=["none", "semantic_binding"], default="none", help="Restrict ILP action space to semantic binding roles.")
    parser.add_argument("--binding-bias", type=float, default=0.25, help="Heuristic bonus for semantic binding actions when constrained.")
    parser.add_argument("--class-head-bias", type=float, default=0.45, help="Heuristic bonus when candidate head matches target class predicate.")
    parser.add_argument("--nonclass-head-penalty", type=float, default=-0.15, help="Penalty for non class_* heads in class-prediction tasks.")
    parser.add_argument("--pair-bias", type=float, default=0.20, help="Bonus for semantically plausible role-pair metarule bodies.")
    parser.add_argument("--class-pair-bias", type=float, default=0.30, help="Additional bonus for role-pair bodies preferred by ontology class.")
    parser.add_argument("--class-pair-miss-penalty", type=float, default=-0.10, help="Penalty when role-pair body does not match class-preferred pair hints.")
    parser.add_argument("--strict-transfer-polarity", action="store_true", help="Apply stronger mismatch penalty for transfer directional role pairs.")
    parser.add_argument("--self-reference-penalty", type=float, default=-0.6, help="Penalty for clauses whose body reuses the head predicate.")
    parser.add_argument("--local-role-bias", type=float, default=0.20, help="Bonus for clauses using query-event-local semantic roles.")
    parser.add_argument("--body-predicate-bias", type=float, default=0.20, help="Bonus when clause body uses preferred query verb predicate.")
    parser.add_argument("--body-predicate-miss-penalty", type=float, default=-0.05, help="Penalty when preferred query verb predicate is absent from clause body.")
    parser.add_argument("--clause-cost", type=float, default=-0.10, help="Per-extra-clause penalty to keep hypotheses compact.")
    parser.add_argument("--mirrored-pair-penalty", type=float, default=-0.25, help="Penalty when adding the mirrored role-pair of an existing clause.")
    parser.add_argument("--repeated-head-penalty", type=float, default=-0.15, help="Penalty when adding another clause with the same head predicate.")
    parser.add_argument("--required-role-bias", type=float, default=0.30, help="Bonus when all class-required roles are present in a clause body.")
    parser.add_argument("--missing-required-role-penalty", type=float, default=-0.20, help="Penalty per missing required role in clause body.")
    parser.add_argument("--confidence-scaled-heuristics", action="store_true", help="Scale selected heuristic strengths by ontology class confidence.")
    parser.add_argument("--class-head-only", action="store_true", help="Restrict candidate heads to class_* predicates only.")
    parser.add_argument("--strict-class-pairs", action="store_true", help="Use only the top preferred role pair for each class.")
    parser.add_argument("--role-pair-only", action="store_true", help="Disable role-transfer metarules and use role-pair metarules only.")
    parser.add_argument("--recipient-synth-min-confidence", type=float, default=0.80, help="Minimum transfer class confidence for recipient synthesis.")
    parser.add_argument("--recipient-synth-require-cue", action="store_true", help="Require known transfer verb cue (inflow/outflow) before recipient synthesis.")
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
                    role_pred = _canonical_role_name(pred)
                    roles.add(role_pred)
                    background.append(Atom(role_pred, [ev, arg]))
            query_local_roles = _query_event_role_set(props, q_ev)
            recovered_recipient_count = _augment_transfer_recipient_roles(
                background,
                roles,
                cls,
                transfer_confidence=conf,
                verb=verb,
                min_confidence=args.recipient_synth_min_confidence,
                require_transfer_cue=args.recipient_synth_require_cue,
            )

            target = f"class_{cls}"
            preferred_heads = [target]
            if args.class_competition:
                for c in CLASS_LABELS:
                    predicates.add(f"class_{c}")
            else:
                predicates.add(target)
            if args.class_head_only:
                predicates = {p for p in predicates if str(p).startswith("class_")}
                predicates.add(target)
            if args.strict_transfer_polarity and cls == "transfer":
                predicates = {p for p in predicates if str(p).startswith("class_")}
                predicates.add(target)
            pos = [Atom(target, [q_ev])] if args.class_supervision == "strict" else []
            neg = []
            if not roles:
                roles = {"agent", "patient"}
            preferred_role_pairs = _preferred_role_pairs_for_class(cls, roles, verb=verb)
            effective_pair_miss_penalty = float(args.class_pair_miss_penalty)
            effective_nonclass_head_penalty = float(args.nonclass_head_penalty)
            effective_self_ref_penalty = float(args.self_reference_penalty)
            effective_clause_cost = float(args.clause_cost)
            effective_mirrored_pair_penalty = 0.0
            effective_repeated_head_penalty = 0.0
            effective_required_role_bias = float(args.required_role_bias)
            effective_missing_required_role_penalty = float(args.missing_required_role_penalty)
            allowed_role_pairs = None
            allowed_metarule_types = ["role_pair"] if args.role_pair_only else None
            if args.strict_transfer_polarity and cls == "transfer" and len(preferred_role_pairs) == 1:
                effective_pair_miss_penalty = min(effective_pair_miss_penalty, -1.5)
                effective_nonclass_head_penalty = min(effective_nonclass_head_penalty, -2.5)
                effective_self_ref_penalty = min(effective_self_ref_penalty, -2.0)
                effective_clause_cost = min(effective_clause_cost, -0.25)
                effective_mirrored_pair_penalty = min(float(args.mirrored_pair_penalty), -0.8)
                effective_repeated_head_penalty = min(float(args.repeated_head_penalty), -0.5)
                allowed_role_pairs = list(preferred_role_pairs)
                allowed_metarule_types = ["role_pair"]
            elif args.strict_class_pairs and preferred_role_pairs:
                allowed_role_pairs = [preferred_role_pairs[0]]
                effective_repeated_head_penalty = min(float(args.repeated_head_penalty), -0.35)
                effective_mirrored_pair_penalty = min(float(args.mirrored_pair_penalty), -0.2)
            if args.confidence_scaled_heuristics:
                scale = max(0.0, min(1.0, float(conf)))
                effective_required_role_bias *= scale
                effective_missing_required_role_penalty *= scale
            roles = list(roles)
            predicates = list(predicates)
            preferred_body_preds = [f"predicate_{verb}"] if f"predicate_{verb}" in predicates else []
            required_roles = set()
            if preferred_role_pairs:
                required_roles.update(str(x) for x in preferred_role_pairs[0])
            if not required_roles:
                required_roles = {"agent", "patient"}
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
                    action_constraint=args.action_constraint,
                    binding_bias=args.binding_bias,
                    preferred_heads=preferred_heads,
                    class_head_bias=args.class_head_bias,
                    nonclass_head_penalty=effective_nonclass_head_penalty,
                    pair_bias=args.pair_bias,
                    preferred_role_pairs=preferred_role_pairs,
                    pair_match_bias=args.class_pair_bias,
                    pair_miss_penalty=effective_pair_miss_penalty,
                    self_ref_penalty=effective_self_ref_penalty,
                    allowed_role_pairs=allowed_role_pairs,
                    allowed_metarule_types=allowed_metarule_types,
                    preferred_local_roles=query_local_roles,
                    local_role_bias=args.local_role_bias,
                    preferred_body_preds=preferred_body_preds,
                    body_predicate_bias=args.body_predicate_bias,
                    body_predicate_miss_penalty=args.body_predicate_miss_penalty,
                    clause_cost=effective_clause_cost,
                    mirrored_pair_penalty_value=effective_mirrored_pair_penalty,
                    repeated_head_penalty_value=effective_repeated_head_penalty,
                    required_roles=sorted(required_roles),
                    required_role_bias=effective_required_role_bias,
                    missing_required_role_penalty=effective_missing_required_role_penalty,
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
                "recovered_recipient_count": int(recovered_recipient_count),
                "query_local_roles": sorted(str(r) for r in query_local_roles),
                "transfer_flow_mode": _transfer_flow_mode_from_verb(verb) if cls == "transfer" else "na",
                "search_strategy": args.search_strategy,
                "action_constraint": args.action_constraint,
                "binding_bias": float(args.binding_bias),
                "class_head_bias": float(args.class_head_bias),
                "nonclass_head_penalty": float(effective_nonclass_head_penalty),
                "pair_bias": float(args.pair_bias),
                "class_pair_bias": float(args.class_pair_bias),
                "class_pair_miss_penalty": float(effective_pair_miss_penalty),
                "strict_transfer_polarity": bool(args.strict_transfer_polarity),
                "class_head_only": bool(args.class_head_only),
                "strict_class_pairs": bool(args.strict_class_pairs),
                "role_pair_only": bool(args.role_pair_only),
                "self_reference_penalty": float(effective_self_ref_penalty),
                "local_role_bias": float(args.local_role_bias),
                "body_predicate_bias": float(args.body_predicate_bias),
                "body_predicate_miss_penalty": float(args.body_predicate_miss_penalty),
                "clause_cost": float(effective_clause_cost),
                "mirrored_pair_penalty": float(effective_mirrored_pair_penalty),
                "repeated_head_penalty": float(effective_repeated_head_penalty),
                "required_roles": sorted(required_roles),
                "required_role_bias": float(effective_required_role_bias),
                "missing_required_role_penalty": float(effective_missing_required_role_penalty),
                "confidence_scaled_heuristics": bool(args.confidence_scaled_heuristics),
                "recipient_synth_min_confidence": float(args.recipient_synth_min_confidence),
                "recipient_synth_require_cue": bool(args.recipient_synth_require_cue),
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
                "action_constraint": args.action_constraint,
                "binding_bias": float(args.binding_bias),
                "class_head_bias": float(args.class_head_bias),
                "nonclass_head_penalty": float(args.nonclass_head_penalty),
                "pair_bias": float(args.pair_bias),
                "class_pair_bias": float(args.class_pair_bias),
                "class_pair_miss_penalty": float(args.class_pair_miss_penalty),
                "strict_transfer_polarity": bool(args.strict_transfer_polarity),
                "self_reference_penalty": float(args.self_reference_penalty),
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
