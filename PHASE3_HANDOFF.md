# Handoff to Phase 3: Ontology & Generalization

## Current State of the Prototype
We have successfully completed **Phase 2 (The GSM8K Neurosymbolic Bridge)**:
1. **System 1 (DLN):** Integrated the Logic Transformer (`DLNPointerDecoder`) into the cognitive loop. It translates parsed spaCy linguistic features into Neo-Davidsonian logic.
2. **System 2 (Rete Engine):** Created deterministic Experta axioms (`Transfer`, `Generation`, `Destruction`, `Multiplier`) to execute grade-school math.
3. **Working Memory:** GWT-style synchronous loop broadcasts variables and automatically decays (forgets) old facts via TTL.
4. **Overnight Training:** The DLN is currently training on the full 8,500 GSM8K dataset via `gsm8k-tests/lt_core/run_overnight.sh`. (The resulting weights will be saved locally to `models/dln_gsm8k_best.pt` and won't be tracked in git due to size).

## Getting Started on the New Computer
When you pull this repository on your new computer, your immediate next goal is **Phase 3: Ontology & Generalization**.

### The Problem to Solve
Right now, our System 2 Axioms rely on hard-coded verb lists. For example, our Destruction Axiom triggers on:
`lambda x: x in ["eat", "lose", "break", "consume"]`
If the test set says *"John dropped 3 apples"*, System 2 fails because "drop" isn't in the list.

### Next Steps for Phase 3
1. **WordNet/VerbNet Integration:** Replace the hardcoded Python lambda lists in `forward-engine/system2_poc.py` with an ontological hierarchy. 
2. **Experta Inheritance:** Leverage Experta's native Python class inheritance. You can dynamically define classes based on WordNet hypernyms (e.g., `class Drop(LossEvent): pass`), so the rule `@Rule(LossEvent())` naturally catches all of them.
3. **BFO (Basic Formal Ontology):** Optionally map these lexical entities into BFO classifications (Continuants vs. Occurrents) for stricter neurosymbolic grounding.

Good luck with Phase 3! The core neurosymbolic loop is fully functional and ready for this upper ontology.

## Phase 3 Progress Update
The prototype now has an `forward-engine/ontology.py` helper with event-class hierarchy support and optional WordNet-backed verb resolution. `forward-engine/system2_poc.py` now matches `TransferEvent`, `CreationEvent`, `LossEvent`, and `ComparisonEvent` facts directly, and `cognitive_loop.py` resolves predicate verbs into ontology event facts before handing them to System 2.

## Phase 3 Synthetic Test
Run `forward-engine/phase3_synthetic_test.py` to compare a hardcoded verb baseline against the ontology-backed resolver on held-out verbs. The script reports both event-class accuracy and inventory state accuracy, which makes it a small end-to-end check for whether the ontology helps System 1/System 2 cooperation.

Current benchmark result: the hardcoded baseline reaches 0.250 class accuracy / 0.000 state accuracy, while the ontology-backed resolver reaches 1.000 class accuracy / 1.000 state accuracy on the current 8-example held-out set.
