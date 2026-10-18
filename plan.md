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
