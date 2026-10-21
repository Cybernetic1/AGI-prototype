2026-10-22 final-state authority milestone
- Fixed the remaining ordering bug in the state machine: final-state transitions now use the global event order instead of restarting per sub-sequence, so the latest valid state outranks stale earlier candidates.
- Result: the compact semantic benchmark is green again at 14 / 14 active families, including the previously failing buy-more / spoilage final-state cluster.
- This is the current clean checkpoint for a controlled next step: keep the benchmark narrow and stable and only widen one family at a time.
- Do not create or activate a repo-local `.venv`; continue using the shared parent environment at `/home/yky/misc-programs/.venv`.

Next steps after the milestone
1) Keep the benchmark compact and stable; add only one new semantic family at a time.
2) Extend the state-order contract to a formal finality rule: same-owner/item latest valid event wins, earlier candidates are historical only.
3) Add a tiny regression suite covering the known final-state buy-more/spoilage/donation edge cases and rerun it on every change.
4) Once the compact suite stays green across multiple revisions, broaden to a modestly larger GSM8K-like slice without turning the loop into a noisy broad benchmark sweep.
5) Revisit ILP only as a local helper once the temporal finality contract is stable and the semantic graph remains consistent.

ILP reorganization note (2026-10-19)

I moved the minimal Metagol-style ILP scaffold from forward-engine/ to ilp_learner/ to keep ILP experimentation code grouped.

Files moved/added:
- ilp_learner/metagol_nds.py  (learner + demo)
- ilp_learner/synth_generator.py
- ilp_learner/run_demo.sh
- ilp_learner/README.md
- ilp_learner/__init__.py

Next steps:
- Replace the string-similarity prior in metagol_nds.embedding_prior with the repo's Lorentz hyperbolic prior and predicates' embeddings.
- Generate synthetic nDS datasets from pot-demo and run systematic ILP experiments.
- Integrate ILP learner into the LT/DLN evaluation harness when the prior and datasets are ready.

LT examples (what "LT" means)
- "LT" stands for Logic Transformer. LT exports are structured examples prepared to train or evaluate the Logic Transformer: they are produced by exporter/parsing scripts (e.g., pot-demo/export_lt_examples.py or the GSM8K preprocessor), not by the LT model itself.
- LT exports normalize natural-language + CoT or parsed logical forms into proposition tokens (for example, "predicate(?e1, buy).", "agent(?e1, ?x1)"). These tokens are the canonical input used by align_ontology, the ILP generator, and the LT training pipeline.

2026-10-19 update: align_ontology predicate extraction fix
- Problem: align_ontology was feeding outer functors (e.g. "predicate(?e1, buy)." -> "predicate") into ontology_classify, producing no labeled tokens and NaN contrastive loss.
- Fix: align_ontology now extracts inner verb tokens from "predicate(..., VERB)" and ignores non-verb functors (agent, patient, quantity, etc.). A short 50-epoch run on GSM8K produced non-NaN loss and saved aligned embeddings.

Recommended immediate next steps
1) Run a longer GSM8K alignment (200–500 epochs) using the fixed extractor to produce more stable embeddings for ILP priors.
2) Expand ontology mappings (VERB_ALIASES/_*_VERBS) or add a small heuristic clustering to increase coverage beyond the current ~85 labeled tokens.
3) Re-run ilp_learner/run_verbose_demo.py to observe prior scores and then run ILP experiments with the new GSM8K-aligned embeddings.
4) Decide whether to keep large .pt files in repo checkpoints or store them externally and reference by checksum (to control repo size).

I committed the fix and the short-run artifacts; tell me which of the recommended next steps to start and I'll proceed.

2026-10-19 local checkpoints and alignment
- Created logic_transformer/local_checkpoints/ and copied existing .pt artifacts there to keep local-only copies.
- Added local_checkpoints/ to .gitignore and stopped tracking large .pt files in logic_transformer/checkpoints/ (files remain in history).
- Ran a longer GSM8K alignment (300 epochs) with the predicate extractor fix and saved the long-run embedding to local_checkpoints.

2026-10-19 GSM8K ILP benchmark
- Added ilp_learner/generate_and_run_gsm8k_ilp.py which converts GSM8K LT examples into ILP tasks by:
  - extracting the query event and its predicate verb,
  - building background atoms (predicate_VERB(e), role(e,arg)) and a class label (class_transfer/creation/loss/comparison) via ontology_classify,
  - running the ILP learner to induce a clause mapping event predicates -> class label.
- Ran the generator on the first 30 examples (skipping neutral cases). Results saved to ilp_learner/gsm8k_ilp_results.json.

2026-10-19 GSM8K ILP full-run (timing)
- The current LT export file (gsm8k_train_lt.jsonl) contains 100 examples (not 1779). I ran the full-run script over the available file.
- Result: scanned 100 examples, processed 43 labeled examples, total runtime 8.97s (≈0.21s per processed example, ≈0.09s per scanned example).
- Output saved to ilp_learner/gsm8k_ilp_results_full.json.

Next: expand the GSM8K ILP benchmark (more examples), sweep hyperparameters (beam, prior weight), and aggregate coverage/FP metrics across tasks. #2 from earlier (expand ontology mappings) remains a planned follow-up.

2026-10-19 Hyperparameter sweep (GSM8K ILP)
- Ran hyperparameter sweep over beam=[1,3,6,10], max_clauses=[1,2,3], prior_weight=[0.0,0.5,1.0] on 1291 labeled GSM8K LT tasks.
- Sweep saved to ilp_learner/hyperparam_sweep_results.json.
- Observations:
  * Coverage (pos example inferred) was 1.00 for all parameter combinations — the learner covers the labeled target in every task.
  * Average score increases with prior_weight (expected: prior adds to score). Beam and max_clauses had negligible effect on coverage in this benchmark.
  * Sweep runtime: ~11–12s per combo; total run saved in results file.

Next diagnostic suggestions
- Add negative examples or a held-out validation set to measure false-positive rates and use them to tune prior_weight (coverage alone isn't informative here).
- Expand prior_weight grid (e.g., 0.0–2.0) and test on harder tasks where multiple candidate clauses compete.
- Improve ontology labeling or use clustering to create richer positive/negative pairs for align_ontology so priors better discriminate.

2026-10-20 decision: validation scaffold & next actions
- Keep current auto-labeled validation scaffold and autofill defaults. Use it as a held-out validation set for periodic precision/recall checks rather than as a training target.
- Rationale: autoregressive end-to-end training will supply abundant implicit negatives; scaffold is useful for targeted debugging and regression tests.

Recommended priority work before large-scale runs
1) End-to-end autoregressive ILP/DLN training on a faster machine (primary next run). Evaluate on the scaffold.
2) Improve modules most likely to affect GSM8K success: semantic parsing / LT encoder (better extraction of quantities, roles, and event structure), numeric grounding/scratchpad arithmetic (for multi-step algebra), and aligned embeddings quality (longer align_ontology training and vocab coverage).
3) Add targeted hard-negative generation and a small human-verified validation set (use the reviewer UI / CLI output already produced).
4) If evaluation shows low precision, tune prior_weight and expand metarules/search breadth.

Assessment: prospects for GSM8K
- The current ILP prototype is a useful symbolic scaffold but unlikely to solve GSM8K end-to-end with high accuracy without improvements in: (a) reliable semantic parsing to extract exact numeric facts, (b) explicit numeric reasoning / scratchpad integration, and (c) richer supervision or negatives to calibrate priors.
- With focused work on those modules plus end-to-end autoregressive training and larger LT exports, meaningful gains are achievable.

If you agree, schedule: run end-to-end training on faster machine (week-long or GPU instance), then evaluate on scaffold and iterate on numeric grounding and parser accuracy.

2026-10-21 status update: semantic binding fix
- Fixed the core ratio-binding bug: the base quantity was incorrectly attached to the subject rather than the reference entity.
- Ratio now follows the correct contract: `Ratio(owner=A, base_owner=B, item=apples, factor=2.0)` and `Quantity(owner=B, item=apples, qty=10)` -> result `A = 2 * B`.
- Verified on targeted cases: single-letter and multi-word ratio cases now solve correctly, including half/double/3x examples.
- Remaining next work is not broad arithmetic; it is semantic coverage and variable binding for deeper families: chained ratios, percent stacking, unit conversion, remaining-quantity ambiguity, and mixed state transitions.

Current priority queue
1) Improve semantic composer for ratio + percent + remaining quantity families in the compiler.
2) Add a tiny validation set covering failing families and keep these as regression tests.
3) Re-run focused benchmark families after each change to avoid drift.
4) Only when a benchmarked slice is stable and we have a clean diff should we create a GitHub commit.

GitHub commit timing recommendation
- Good time to commit: after the next focused benchmark pass stabilizes on 2–3 representative failure families and before you start a broader refactor or large dataset run.
- Not yet the best time: while the compiler is still mid-fix and the behavior is changing rapidly; noisy commits are less useful during active semantics churn.
- Best practice for this repo: commit a checkpoint after a meaningful, explainable milestone (for example: `ratio + percent + remaining quantity compiler fix + benchmark regression set`) rather than on every exploratory edit.

2026-10-21 validation pass: compact semantic regression set
- Added a compact benchmark in `cognitive_loop.py` covering the active semantic families: ratio, chained ratio, remaining quantity, sales total, rate-time, and percent-of-remaining cases.
- Verified on the current code: 8 / 8 families in the current compact benchmark solve. This gives a clean, stable slice for continued iteration.
- The remaining challenge is not arithmetic but broader semantic coverage: deeper mixed-state compositions, ambiguous entity ownership, and multi-constraint multi-step reasoning beyond the current curated patterns.
- The background cognitive loop is useful as a benchmark harness only when run against a stable, narrow validation set; it is not yet a broad GSM8K solver.

Current status (2026-10-21)
- Ratio binding fix is in place and verified.
- Percent-of-remaining and mixed ratio/state transitions were stabilized in the compiler.
- Compact family benchmark is green on the targeted slice.
- State-machine support was added and validated on the active mixed-state families.
- Current benchmark posture: 11 / 13 curated families solve with the state-trace enabled.
- Next project move: add a harder mixed-state benchmark and widen only after the current regression set remains stable.

State-machine validation note
- We have tested the state-machine path by checking that ordered `State` / `StateTransition` records are emitted for the mixed-state examples and by running the compact semantic benchmark.
- The target success families include remaining-quantity, percent-of-remaining, and multi-step ratio + give-away + percent transitions.
- Remaining open cases are the buy-more succession patterns, which still need stricter state-ordering or final-state selection logic before they are considered solved.

Immediate next actions
1) Add a harder mixed-state benchmark beyond the current 8-family set (for example, combined ratio + subtraction + remaining state + rate-time).
2) Keep the validation set narrow and stable while we widen one family at a time.
3) Only when the broader mixed-state slice remains stable should we consider a larger GSM8K sweep or a broader semantic proposer.
4) Commit at the next clean checkpoint after the harder benchmark is green, not during active semantic churn.

2026-10-21 mixed-state benchmark expansion
- Added a combined ratio + give-away + remaining-percent family to the compact benchmark to stress the next layer of semantic binding.
- The goal is to ensure the compiler preserves reference-owner difference, remaining-state tracking, and percent-of-remaining composition together in one example.
- If this pattern remains stable, the next step is a slightly broader mixed-state slice; if it fails, we tighten the state-transition contract before any broader GSM8K widening.

2026-10-21 variant widening
- Added a second mixed-state variant using a different percentage (20% instead of 25%) to confirm the state logic is not overfitted to a single hardcoded ratio/percent pair.
- This preserves the same semantic contract while widening the benchmark just enough to test generalization before broader GSM8K steps.

2026-10-21 achievements summary
- Fixed the core ratio-binding bug and verified reference-owner semantics: the base quantity now belongs to the referenced owner, not the subject.
- Stabilized the percent-of-remaining and remaining-quantity compiler paths and kept them green in the compact benchmark.
- Added a family-based semantic validation suite covering ratio, chained ratio, remaining quantity, sales total, rate-time, percent-of-remaining, mixed ratio/state transitions, and deeper multi-step transitions.
- Current benchmark result: 11 / 13 curated semantic families solved on the current code.
- The remaining failures are not arithmetic; they are concentrated in deeper multi-step state composition and final-state update tracking.
- Next structural move: explicit state-transition bookkeeping and ordered final-state updates in the semantic graph, before any broader GSM8K expansion.

2026-10-21 repo status update
- The ratio-binding fix and semantic benchmark pass are committed and pushed to the main branch.
- The next target is the remaining ambiguity family: percent-of-remaining and mixed state transitions in the compiler.
- The repo is still in the semantic grounding / variable-binding phase; broad GSM8K solvability remains a later milestone, not the current one.
- Mixed ratio + total and mixed remaining-state cases now solve in the compiler benchmark, and they have been promoted into the compact validation set.
- Current benchmark posture: the system is stable on a narrow but meaningful slice of GSM8K-like semantic families, and is now suitable for controlled widening rather than noisy broad sweeps.

2026-10-21 semantic widening milestone
- Added a final-state buy-more + spoilage variant to the semantic family benchmark and validated it against the working-memory state logic.
- The final-state ordering fix now suppresses stale earlier quantity candidates and prefers the latest authoritative state transition.
- Validation outcome: the compact benchmark now solves 14 / 14 active semantic families, including the previously failing buy-more / spoilage final-state cluster.
- This is the current clean checkpoint before any broader GSM8K-style expansion.

Next immediate step
- Keep the benchmark compact and stable, and only add one more semantic family at a time if we need new coverage.
- Do not broaden to a large GSM8K sweep until the family suite remains green across multiple revisions and the semantic graph contract is fully stable.

Shared-venv note (2026-10-21)
- A common parent-level venv is viable because both `AGI-prototype` and `AGI-experiments` live under `/home/yky/misc-programs` and already share overlapping scientific Python requirements (notably PyTorch, numpy, and related tooling).
- Recommended pattern: create a single parent env at `/home/yky/misc-programs/.venv` and symlink or point each repo at it only if the dependency versions are compatible across both projects.
- If one repo needs a diverging package set, prefer separate venvs; a shared env is best when both repos are intentionally aligned on the same stack.
- For this repo, do not create or activate a repo-local `.venv` under `AGI-prototype`; always use the parent env at `/home/yky/misc-programs/.venv`.
- The repo root should not contain a local `.venv` directory; if one appears, remove it and use the parent env instead.

Finality-status note (2026-10-21)
- Latest finality heuristic did not fully resolve the open final-state family: the fresh environment benchmark reports `12 / 14 families solved`.
- The remaining issue is still the final-state selection problem for deeper multi-step transitions, especially where stale state candidates survive alongside a later valid update.
- Current interpretation: the finality abstraction is directionally correct, but the rewrite still needs stronger causal/temporal ordering rather than a simple latest-transition preference.

Revision-layer note (2026-10-22)
- The next abstraction is explicit contradiction tracking plus preference ranking, not another ad hoc case patch.
- We now record `Contradiction` facts when same-owner/item state candidates disagree, and rank competing states by temporal order, source authority, and symbolic priority.
- This is the minimal AGM/TMS bridge needed before a larger truth-maintenance or revision policy is introduced.

Current implementation note (2026-10-22)
- We tightened the final-state authority in `working_memory/working_memory.py` so same-owner/item quantities prefer the latest valid ordered state and suppress stale intermediate values in answer selection.
- The fix is intentionally state-graph based rather than benchmark-specific: it makes `FinalState` / `BlockedState` / `Contradiction` explicit, which matches the non-monotonic reasoning direction described in the repo plan.
- Remaining validation should be done with a compact, stable benchmark slice only; broad GSM8K sweeps remain premature while the state-order/priority contract is still evolving.
