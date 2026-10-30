# AGI Bootstrap Milestone Review (v1)

- Generated: 2026-10-29T08:32:35.134082+00:00
- Scope: baseline + invariant + A/B + widening artifacts generated in this execution chain.

## Evidence sources
- Baseline: `rules/reports/agi_bootstrap_baseline_v1.json`
- Invariant: `rules/reports/agi_bootstrap_invariant_curriculum_v1.json`
- A/B: `rules/reports/agi_bootstrap_ab_efficiency_report.json`
- Widening: `rules/reports/agi_bootstrap_curriculum_widening_v1.json`

## What is achieved
- Widening regression passed with all gates green (18/18) in agi_bootstrap_curriculum_widening_v1.json.
- Hybrid profile preserved symbolic validity at 1.0 on baseline, invariant-core, A/B control+treatment, and every widening slice.
- A/B treatment preserved quality metrics: macro deltas are 0.0 for solve rate, symbolic-valid rate, and valid-solves/sample.

## What is not yet achieved
- 100x cost reduction is not met under symbolic-perfect hybrid comparison.
- A/B treatment did not reduce aggregate wall-time; macro delta is +0.097230 sec.
- ILP-heavy symbolic validity is not consistently preserved in widening (mixed-narrative-symbolic-compact=0.333333, widening-sentinel-regression=0.833333).

## Quantified gap to 100x target
- Baseline reference (hybrid control): `1.487491` sec/valid-solve.
- Best symbolic-perfect hybrid observed: `1.221010` sec/valid-solve (widening/adjacent-template-core/hybrid).
- Achieved reduction vs baseline (hybrid): `1.218x` (17.915%).
- 100x threshold from baseline: `0.014875` sec/valid-solve.
- Remaining gap (hybrid path): still `82.085x` above target; needs `98.782%` further reduction.
- Best symbolic-perfect any-profile checkpoint: `1.002778` sec/valid-solve (widening/adjacent-template-core/ilp-heavy), still `67.414x` above target.

## Immediate next actions
- Prioritize hybrid cost reduction experiments while keeping symbolic_valid_rate=1.0 and all widening gates green.
- Target profiles/slices where cost is highest under symbolic-perfect runs (e.g., invariant-core hybrid cost 1.800792 sec/valid-solve).
- Stabilize ilp-heavy symbolic validity on mixed narrative and widening sentinel slices before treating it as a reliable low-cost path.
- Re-run A/B and widening reports after each cost-focused change, then regenerate this milestone review with the command below.

## Reproducible command
```bash
python3 rules/generate_agi_bootstrap_milestone_review.py --baseline-report rules/reports/agi_bootstrap_baseline_v1.json --invariant-report rules/reports/agi_bootstrap_invariant_curriculum_v1.json --ab-report rules/reports/agi_bootstrap_ab_efficiency_report.json --widening-report rules/reports/agi_bootstrap_curriculum_widening_v1.json --output-json rules/reports/agi_bootstrap_milestone_review_v1.json --output-md rules/reports/agi_bootstrap_milestone_review_v1.md
```
