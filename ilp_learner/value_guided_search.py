from __future__ import annotations

from math import sqrt

from ilp_learner.metagol_nds import (
    body_predicate_alignment_bonus,
    clause_economy_penalty,
    embedding_prior,
    event_local_role_bonus,
    generate_clause_candidates,
    head_preference_bonus,
    preferred_role_pair_bonus,
    required_role_coverage_bonus,
    mirrored_role_pair_penalty,
    repeated_head_penalty,
    self_reference_penalty,
    semantic_role_pair_bonus,
    semantic_binding_bonus,
    score_hypothesis,
)
from ilp_learner.value_model import LinearValueModel


def _reward_params(profile: str) -> dict[str, float]:
    name = str(profile or "balanced").lower()
    if name == "conservative":
        return {"delta": 0.20, "prior": 0.05, "duplicate": -0.25}
    if name == "aggressive":
        return {"delta": 0.90, "prior": 0.25, "duplicate": -0.60}
    return {"delta": 0.50, "prior": 0.12, "duplicate": -0.40}


def _feature_map(
    *,
    depth: int,
    max_depth: int,
    parent_score: float,
    candidate_score: float,
    prior: float,
    duplicate: bool,
    metarule_type: str,
) -> dict[str, float]:
    delta = candidate_score - parent_score
    return {
        "bias": 1.0,
        "depth_norm": float(depth) / max(1.0, float(max_depth)),
        "parent_score": float(parent_score),
        "candidate_score": float(candidate_score),
        "delta_score": float(delta),
        "prior": float(prior),
        "is_duplicate": 1.0 if duplicate else 0.0,
        "is_role_pair": 1.0 if metarule_type == "role_pair" else 0.0,
    }


def learn_with_value_guided_search(
    *,
    predicates,
    roles,
    background,
    pos_examples,
    neg_examples,
    beam: int = 5,
    max_clauses: int = 3,
    prior_weight: float = 0.5,
    reward_profile: str = "balanced",
    value_model: LinearValueModel | None = None,
    value_learning_rate: float = 0.02,
    value_discount: float = 0.9,
    online_value_updates: bool = True,
    action_constraint: str = "none",
    binding_bias: float = 0.0,
    preferred_heads=None,
    class_head_bias: float = 0.0,
    nonclass_head_penalty: float = 0.0,
    pair_bias: float = 0.0,
    preferred_role_pairs=None,
    pair_match_bias: float = 0.0,
    pair_miss_penalty: float = 0.0,
    self_ref_penalty: float = 0.0,
    allowed_role_pairs=None,
    allowed_metarule_types=None,
    preferred_local_roles=None,
    local_role_bias: float = 0.0,
    preferred_body_preds=None,
    body_predicate_bias: float = 0.0,
    body_predicate_miss_penalty: float = 0.0,
    clause_cost: float = 0.0,
    mirrored_pair_penalty_value: float = 0.0,
    repeated_head_penalty_value: float = 0.0,
    required_roles=None,
    required_role_bias: float = 0.0,
    missing_required_role_penalty: float = 0.0,
):
    """Value-guided clause search with lightweight MCTS-style visit bonus.

    This keeps the expansion graph discrete and compact while adding learned
    value guidance and exploration pressure beyond static beam ranking.
    """
    params = _reward_params(reward_profile)
    model = value_model or LinearValueModel.seeded()

    frontier = [{"hyp": [], "score": 0.0, "objective": 0.0}]
    action_visits: dict[tuple[int, str], int] = {}
    trace = []

    for depth in range(max_clauses):
        candidates = []
        parent_count = max(1, len(frontier))
        for node in frontier:
            hyp = node["hyp"]
            parent_score = float(node["score"])
            for spec in generate_clause_candidates(
                predicates,
                roles,
                existing_hyp=hyp,
                action_constraint=action_constraint,
                allowed_role_pairs=allowed_role_pairs,
                allowed_metarule_types=allowed_metarule_types,
            ):
                clause = spec["clause"]
                clause_str = str(clause)
                duplicate = bool(spec["duplicate"])
                new_hyp = hyp + [clause]
                score = score_hypothesis(new_hyp, pos_examples, neg_examples, background)
                prior = embedding_prior(spec["head_pred"], spec["body_preds"])
                features = _feature_map(
                    depth=depth + 1,
                    max_depth=max_clauses,
                    parent_score=parent_score,
                    candidate_score=score,
                    prior=prior,
                    duplicate=duplicate,
                    metarule_type=spec["metarule_type"],
                )
                value_est = model.score(features)
                shaped_reward = (
                    params["delta"] * (score - parent_score)
                    + params["prior"] * prior
                    + (params["duplicate"] if duplicate else 0.0)
                )
                bind_bonus = semantic_binding_bonus(
                    spec["body_preds"],
                    action_constraint=action_constraint,
                    binding_bias=binding_bias,
                )
                pair_bonus = semantic_role_pair_bonus(
                    spec["body_preds"],
                    metarule_type=spec.get("metarule_type", ""),
                    pair_bias=pair_bias,
                )
                class_pair_bonus = preferred_role_pair_bonus(
                    spec["body_preds"],
                    metarule_type=spec.get("metarule_type", ""),
                    preferred_role_pairs=preferred_role_pairs,
                    pair_match_bias=pair_match_bias,
                    pair_miss_penalty=pair_miss_penalty,
                )
                head_bonus = head_preference_bonus(
                    spec.get("head_pred", ""),
                    preferred_heads=preferred_heads,
                    class_head_bias=class_head_bias,
                    nonclass_head_penalty=nonclass_head_penalty,
                )
                ref_pen = self_reference_penalty(
                    spec.get("head_pred", ""),
                    spec.get("body_preds", []),
                    penalty=self_ref_penalty,
                )
                local_bonus = event_local_role_bonus(
                    spec.get("body_preds", []),
                    preferred_local_roles=preferred_local_roles,
                    local_role_bias=local_role_bias,
                )
                body_bonus = body_predicate_alignment_bonus(
                    spec.get("body_preds", []),
                    preferred_body_preds=preferred_body_preds,
                    body_predicate_bias=body_predicate_bias,
                    body_predicate_miss_penalty=body_predicate_miss_penalty,
                )
                compact_pen = clause_economy_penalty(len(new_hyp), clause_cost=clause_cost)
                mirror_pen = mirrored_role_pair_penalty(
                    hyp,
                    clause,
                    candidate_type=spec.get("metarule_type", ""),
                    penalty=mirrored_pair_penalty_value,
                )
                repeat_pen = repeated_head_penalty(
                    hyp,
                    clause,
                    penalty=repeated_head_penalty_value,
                )
                role_coverage = required_role_coverage_bonus(
                    spec.get("body_preds", []),
                    required_roles=required_roles,
                    required_role_bias=required_role_bias,
                    missing_required_role_penalty=missing_required_role_penalty,
                )
                shaped_reward += bind_bonus
                shaped_reward += pair_bonus
                shaped_reward += class_pair_bonus
                shaped_reward += head_bonus
                shaped_reward += ref_pen
                shaped_reward += local_bonus
                shaped_reward += body_bonus
                shaped_reward += compact_pen
                shaped_reward += mirror_pen
                shaped_reward += repeat_pen
                shaped_reward += role_coverage

                action_key = (depth, clause_str)
                visits = action_visits.get(action_key, 0)
                # Small exploration bonus (UCB-like) to avoid myopic collapse.
                exploration = 0.15 * sqrt(max(0.0, (1.0 + parent_count) / (1.0 + visits)))
                total = score + prior_weight * prior + value_est + shaped_reward + exploration
                candidates.append(
                    {
                        "hyp": new_hyp,
                        "score": float(score),
                        "total": float(total),
                        "prior": float(prior),
                        "value_est": float(value_est),
                        "shaped_reward": float(shaped_reward),
                        "binding_bonus": float(bind_bonus),
                        "pair_bonus": float(pair_bonus),
                        "class_pair_bonus": float(class_pair_bonus),
                        "head_bonus": float(head_bonus),
                        "self_ref_penalty": float(ref_pen),
                        "local_role_bonus": float(local_bonus),
                        "body_predicate_bonus": float(body_bonus),
                        "clause_economy_penalty": float(compact_pen),
                        "mirrored_pair_penalty": float(mirror_pen),
                        "repeated_head_penalty": float(repeat_pen),
                        "required_role_bonus": float(role_coverage),
                        "duplicate": duplicate,
                        "clause": clause_str,
                        "depth": depth + 1,
                        "features": features,
                        "metarule_type": spec["metarule_type"],
                    }
                )

        if not candidates:
            break
        candidates.sort(key=lambda c: c["total"], reverse=True)
        top = candidates[: max(1, int(beam))]
        for cand in top:
            action_key = (cand["depth"] - 1, cand["clause"])
            action_visits[action_key] = action_visits.get(action_key, 0) + 1
            if online_value_updates:
                # TD-style update: immediate shaped reward + discounted objective proxy.
                td_target = cand["shaped_reward"] + value_discount * cand["total"]
                model.update(cand["features"], td_target, lr=value_learning_rate)
            trace.append(
                {
                    "depth": cand["depth"],
                    "score": cand["score"],
                    "total": cand["total"],
                    "prior": cand["prior"],
                    "value_est": cand["value_est"],
                    "shaped_reward": cand["shaped_reward"],
                    "binding_bonus": cand["binding_bonus"],
                    "pair_bonus": cand["pair_bonus"],
                    "class_pair_bonus": cand["class_pair_bonus"],
                    "head_bonus": cand["head_bonus"],
                    "self_ref_penalty": cand["self_ref_penalty"],
                    "local_role_bonus": cand["local_role_bonus"],
                    "body_predicate_bonus": cand["body_predicate_bonus"],
                    "clause_economy_penalty": cand["clause_economy_penalty"],
                    "mirrored_pair_penalty": cand["mirrored_pair_penalty"],
                    "repeated_head_penalty": cand["repeated_head_penalty"],
                    "required_role_bonus": cand["required_role_bonus"],
                    "duplicate": cand["duplicate"],
                    "metarule_type": cand["metarule_type"],
                    "clause": cand["clause"],
                    "td_target": cand["shaped_reward"] + value_discount * cand["total"],
                    "features": dict(cand["features"]),
                }
            )
        frontier = [{"hyp": c["hyp"], "score": c["score"], "objective": c["total"]} for c in top]

        # Stop early if we already have very high positive-example coverage.
        if frontier and frontier[0]["score"] > len(pos_examples) * 0.9:
            break

    if not frontier:
        return [], 0.0, trace
    best = max(frontier, key=lambda x: x["objective"])
    return best["hyp"], float(best["objective"]), trace
