# Copilot instructions for AGI-prototype

Purpose
- Short guide to help Copilot sessions understand how to run, test, and reason about this repository.

1) Build / install / run (per-submodule)
- Recommended: run from repository root so relative imports and sys.path patches work.
- Create and activate a venv then install requirements for the submodule you need.
  - logic_transformer (training/eval):
    - cd logic_transformer
    - python -m venv venv && source venv/bin/activate
    - pip install -r requirements.txt
    - Download data and run single steps: python download_gsm8k.py --config main --limit-train 200 --limit-test 50
    - Train: python train.py --data-path data/gsm8k/main_train.jsonl
    - Evaluate: python eval.py --data-path data/gsm8k/main_test.jsonl
    - Smoke demo: ./run_demo.sh
  - forward_engine (System 2):
    - cd forward_engine
    - pip install -r requirements.txt
    - python -m nltk.downloader wordnet omw-1.4
    - Quick run: python system2_poc.py
  - ilp_learner (System 3 / ILP):
    - ./ilp_learner/run_demo.sh

- Top-level orchestrator / demos
  - Run the synchronous orchestrator demo: python cognitive_loop.py
  - Run the integrated demo: python test_integrated_ilp.py
  - Run any script shown with if __name__ == "__main__" directly with python <script>.py to run a "single test" or demo.

2) Single-file / single-test guidance
- There is no pytest/ tox based test harness. To run a single demo/test run the script directly from repo root, e.g.:
  - python test_integrated_ilp.py
  - python cognitive_loop.py
- If a script expects model files (check logic_transformer load_dln paths) first run the dataset download + train pipeline or place model/vocab artifacts under logic_transformer/data and logic_transformer/models.

3) High-level architecture (big picture)
- Tri-system neurosymbolic architecture unified by a Shared Working Memory / Global Workspace (WM):
  - System 1 — Logic Transformer (DLN): neural, fuzzy/soft predictions, produces low-confidence WM facts (activation ~0.85). Implemented in logic_transformer/.
  - System 2 — Rete forward-chaining (Experta): deterministic pattern matching; consumes WM facts and emits Belief/Inventory/EventProcessed facts. Implemented in forward_engine/system2_poc.py.
  - System 3 — ProbLog / ILP-like backward-chaining and abduction: deep, goal-directed reasoning and online rule induction; implemented under ilp_learner/ and integrated in cognitive_loop.system3_step.
- Orchestrator: cognitive_loop.py runs a synchronous cognitive cycle: broadcast WM snapshot to all three systems (executed in parallel via ProcessPoolExecutor), merge outputs back into WM, apply TTL-based decay, repeat for cycles.

4) Key repository-specific conventions
- Working Memory and WMNode:
  - WM facts are keys in WorkingMemory.facts mapping to integer TTL values. Facts are WMNode instances created via WMNode.create(...).
  - WMNode fields use arg0/arg1... or named attributes; check to_dict() usage.
  - Each fact carries an activation/salience (float 0..1). System1 emits lower-activation guesses; System3 emits solved Belief with activation=1.0.
  - Facts that represent induced rules are created with concept name "Inducedrule" and specific fields (head_pred, head_args, body, calc, target).
- Naming and capitalization:
  - Node concept names are capitalized in WM (e.g., Inventory, Belief, Query, Inducedrule). Predicates in the KB are usually lowercase when declared from WM.
- Sys.path / import expectations:
  - Many top-level scripts append submodule dirs (forward_engine, logic_transformer, ilp_learner) into sys.path. Run from repo root to ensure these paths resolve correctly.
- Data & model artifacts:
  - logic_transformer expects model checkpoints and vocab files under logic_transformer/models and logic_transformer/data. If missing, run download scripts + train to create them.
- Parallelism & deterministic loop:
  - cognitive_loop enforces a synchronous, discrete cognitive cycle (ProcessPoolExecutor used to run three systems in parallel per cycle). Do not attempt to run subsystems as long-running daemons without adapting the orchestrator.
- Python compatibility:
  - forward_engine includes compatibility patches for newer Python versions (Experta related). Use the repo-provided scripts rather than relying on system-global Experta behavior.

5) Existing AI-assistant config discovery
- No repository-level Copilot/Claude/Cursor/Aider assistant rules were found (CLAUDE.md, .cursorrules, AGENTS.md, CONVENTIONS.md, .windsurfrules, .clinerules). If you want Copilot to follow additional local policies, add one of those files and update this document.

6) Quick checklist for Copilot sessions
- Start by reading cognitive_loop.py, working_memory/working_memory.py, forward_engine/system2_poc.py, and ilp_learner/abductive_ilp.py for the cross-cutting flow.
- If a requested task touches training/inference, verify presence of model artifacts under logic_transformer/models and logic_transformer/data; otherwise instruct user to run the download+train steps.
- Run scripts from repository root; prefer using per-submodule venvs and install only required packages.

---
Created by Copilot CLI scan on repository root. If you'd like additional focused instructions (for example, GPU training tips in logic_transformer, environment.yml/conda support, or adding a pytest harness), say which area to expand.