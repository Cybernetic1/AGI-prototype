from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cognitive_loop import benchmark_template_generalization
from ilp_learner.metarule_loop import run_metarule_learning_cycle
from working_memory.working_memory import WorkingMemory


def register_candidates(*, top_n: int, activate: bool) -> list[dict]:
    wm = WorkingMemory(default_ttl=3)
    accepted = run_metarule_learning_cycle(top_n=top_n)
    status = "active" if activate else "approved"
    records = []
    for idx, cand in enumerate(accepted, start=1):
        family = str(cand.get("family") or "unknown")
        name = str(cand.get("name") or f"empirical_{family}_{idx}")
        ranking = cand.get("ranking") or {}
        rule_id = f"emp_{family}_{idx}_{name}"
        condition = f"metarule-family={family}; body={cand.get('body')}"
        effect = f"head={cand.get('head')}; calc={cand.get('calc')}; target={cand.get('target')}"
        evidence = {"ranking": ranking}
        rec = wm.register_empirical_rule(
            rule_id=rule_id,
            family=family,
            condition=condition,
            effect=effect,
            priority=int(40 + max(0, min(60, int(100 * float(ranking.get('score', 0.0)))))),
            status=status,
            evidence=evidence,
        )
        records.append(rec)
    return records


def run_ab(*, cycles: int, quiet: bool):
    baseline = benchmark_template_generalization(max_cycles=cycles, quiet=quiet, use_ltm_recall=False)
    recall_on = benchmark_template_generalization(
        max_cycles=cycles,
        quiet=quiet,
        use_ltm_recall=True,
        auto_accept_recall=True,
        write_ltm_episode=True,
    )
    return {
        "baseline": {"solved": baseline["solved"], "total": baseline["total"]},
        "recall": {"solved": recall_on["solved"], "total": recall_on["total"]},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run metarule synthesis/registration and A/B eval.")
    parser.add_argument("--top-n", type=int, default=2)
    parser.add_argument("--activate", action="store_true", help="Register metarules as active instead of approved.")
    parser.add_argument("--ab", action="store_true", help="Run A/B benchmark (baseline vs recall+writeback).")
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    recs = register_candidates(top_n=max(1, args.top_n), activate=args.activate)
    print(f"Registered {len(recs)} empirical metarules (status={'active' if args.activate else 'approved'}).")

    if args.ab:
        report = run_ab(cycles=max(1, args.cycles), quiet=args.quiet)
        print("A/B report:")
        print(f"  baseline: {report['baseline']['solved']} / {report['baseline']['total']}")
        print(f"  recall  : {report['recall']['solved']} / {report['recall']['total']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
