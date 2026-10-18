Metagol-like ILP (neo-Davidsonian)

This package contains a minimal Metagol-style ILP prototype restricted to neo-Davidsonian (event-centered) clauses.

Files:
- metagol_nds.py  : Learner + small demo runner
- synth_generator.py : Small synthetic domain generator (optional)
- run_demo.sh : Quick runner

Usage:
cd AGI-prototype
./ilp_learner/run_demo.sh

Notes:
- This is a compact scaffold for experimentation. It uses simple forward-chaining and beam search.
- Extend metarules, clause instantiation, or replace the similarity prior with your hyperbolic alignment when integrating with System-1.
- The code was reorganized from forward-engine/ to ilp_learner/ on 2026-10-19 to keep ILP code grouped.
