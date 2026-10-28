from __future__ import annotations

from math import sqrt

from ilp_learner.metagol_nds import (
    embedding_prior,
    instantiate_metarule_role_transfer,
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
    }


def _initial_model() -> LinearValueModel:
    # Hand-tuned initial weights stabilize early exploration.
    return LinearValueModel(
        weights={
            "bias": 0.02,
            "depth_norm": -0.02,
            "parent_score": 0.10,
            "candidate_score": 0.45,
            "delta_score": 0.80,
            "prior": 0.15,
            "is_duplicate": -0.50,
        },
        bias=0.0,
    )


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
):
    """Value-guided clause search with lightweight MCTS-style visit bonus.

    This keeps the expansion graph discrete and compact while adding learned
    value guidance and exploration pressure beyond static beam ranking.
    """
    params = _reward_params(reward_profile)
    model = value_model or _initial_model()

    frontier = [{"hyp": [], "score": 0.0, "objective": 0.0}]
    action_visits: dict[tuple[int, str], int] = {}
    trace = []

    for depth in range(max_clauses):
        candidates = []
        parent_count = max(1, len(frontier))
        for node in frontier:
            hyp = node["hyp"]
            parent_score = float(node["score"])
            seen_clause_strings = {str(c) for c in hyp}
            for p_pred in predicates:
                for q_pred in predicates:
                    for role_pred in roles:
                        clause = instantiate_metarule_role_transfer(p_pred, q_pred, role_pred)
                        clause_str = str(clause)
                        duplicate = clause_str in seen_clause_strings
                        new_hyp = hyp + [clause]
                        score = score_hypothesis(new_hyp, pos_examples, neg_examples, background)
                        prior = embedding_prior(q_pred, [p_pred, role_pred])
                        features = _feature_map(
                            depth=depth + 1,
                            max_depth=max_clauses,
                            parent_score=parent_score,
                            candidate_score=score,
                            prior=prior,
                            duplicate=duplicate,
                        )
                        value_est = model.score(features)
                        shaped_reward = (
                            params["delta"] * (score - parent_score)
                            + params["prior"] * prior
                            + (params["duplicate"] if duplicate else 0.0)
                        )

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
                                "duplicate": duplicate,
                                "clause": clause_str,
                                "depth": depth + 1,
                            }
                        )

        if not candidates:
            break
        candidates.sort(key=lambda c: c["total"], reverse=True)
        top = candidates[: max(1, int(beam))]
        for cand in top:
            action_key = (cand["depth"] - 1, cand["clause"])
            action_visits[action_key] = action_visits.get(action_key, 0) + 1
            trace.append(
                {
                    "depth": cand["depth"],
                    "score": cand["score"],
                    "total": cand["total"],
                    "prior": cand["prior"],
                    "value_est": cand["value_est"],
                    "shaped_reward": cand["shaped_reward"],
                    "duplicate": cand["duplicate"],
                    "clause": cand["clause"],
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
