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