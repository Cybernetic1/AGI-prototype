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
