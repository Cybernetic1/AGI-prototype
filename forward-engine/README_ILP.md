Metagol-like ILP (neo-Davidsonian)

This folder contains a minimal Metagol-style ILP prototype restricted to neo-Davidsonian (event-centered) clauses.

Files:
- ilp_metagol_nds.py  : Learner + small demo runner
- ilp_synth_generator.py : Small synthetic domain generator (optional)
- run_ilp_demo.sh : Quick runner

Usage:
cd forward-engine
./run_ilp_demo.sh

Notes:
- This is a compact scaffold for experimentation. It uses simple forward-chaining and beam search.
- Extend metarules, clause instantiation, or replace the similarity prior with your hyperbolic alignment when integrating with System-1.
