# AGI Development Plan for this Repository

This plan is intentionally concrete but flexible. The architecture is likely to evolve as we discover the actual bottlenecks on GSM8K, so the goal is to keep the overall direction stable while allowing each module to change as evidence demands.

## Direction

The project should evolve toward a symbolic-first AGI architecture:

- System 1: minimal semantic grounding layer
- System 2: forward-chaining inference and local rule learning
- System 3: backward-chaining / goal-directed reasoning
- ILP: shared rule induction and hypothesis generation
- LTM: long-term memory of rules, patterns, and solved examples
- Orchestrator: disciplined cognitive loop with logging and evaluation

The benchmark target is GSM8K, and the system should be shaped by bottleneck discovery rather than by a fixed ideology.

## Core principles

- System 1 should produce candidate facts, not solve the problem.
- System 2 should handle fast, structured inference over known facts and state transitions.
- System 3 should handle deeper search and goal-directed reasoning when System 2 is insufficient.
- ILP should be treated as a shared rule-discovery engine that can serve both symbolic systems.
- LTM should retain useful learned rules and previously solved patterns across runs.
- The cognitive loop should remain disciplined: snapshot, infer, merge, decay, log, evaluate.

## Recent implementation milestones (brief)

- **Canonical + defeasible Sys-2 runtime**
  - Added file-backed defeasible rules with strict (`->`), defeasible (`⟹`), and priority (`r1 > r2`) forms.
  - Defeasible resolver runs in the live loop and emits explicit meta-facts (`rule_applied`, `overrides`, `defeated`) for revision/accounting.

- **Rule lifecycle registry (ML-free governance)**
  - Added durable rule registry with status workflow: `proposed -> approved -> active -> retracted`.
  - Added provenance/mutability policy (`teacher` vs `empirical`; `locked` / `priority_only` / `fully_mutable`).
  - Added semantic validation gates for NL-injected rule families before activation.
  - Added strict teacher-LLM proposal ingestion (`rules/facts/counterfactuals`) with proposal-only runtime authority, audit/provenance logging, and low-volume caps.
  - Added A/B controls (`control` vs `treatment`) and explicit low-volume toggles for bootstrap ingestion runs.

- **WM↔LTM bridge with associative recall**
  - Added LTM store with episodic, semantic, and rule memory partitions.
  - WM can write episodes, retrieve cue-matched recall candidates, and run sleep-style consolidation/forgetting.
  - Forgetting uses retention/interestingness signals (reward, reuse, rule-signal, cue support) and archives weak episodes.

- **Recall accept/reject + simple rule accounting**
  - Associative recall is staged as candidates; each candidate can be explicitly accepted or rejected.
  - Per-rule accounting now tracks `applied`, `wins`, `defeats`, `recall_accepted`, and `recall_rejected`.
  - Defeasible outcomes are converted into accounting updates for later priority tuning.

- **Operational tools and validation**
  - CLI support to inspect Sys-2, defeasible KB, registry, and LTM summaries.
  - Focused validators exist for predicate grammar, defeasible behavior, lifecycle transitions, LTM flow, and accounting+recall decisions.

## Phase 1: Stabilize benchmark and instrumentation

### Objectives
- Get a reliable GSM8K benchmark harness.
- Log all problem traces and system outputs.
- Make failures interpretable.

### Deliverables
- Small benchmark runner for a fixed GSM8K subset.
- Per-problem structured logs:
  - input text
  - extracted propositions
  - current WM state
  - System 2 deductions
  - System 3 attempts
  - ILP proposals
  - final result
- Basic metrics report: solved / unsolved / failure class.

### Flexible rule
- Do not add architecture before measuring small benchmark slices.

## Phase 2: Reduce System 1 to a grounding layer

### Objectives
- Keep System 1 precise and narrow.
- Use it for semantic grounding instead of reasoning.
- Improve the conversion from raw text to symbolic candidate facts.

### Priority tasks
- Normalize numbers and units.
- Detect quantities, entities, relations, and time/state changes.
- Extract arithmetic operations from natural-language phrasing.
- Distinguish:
  - factual state
  - inferred state
  - query target
  - arithmetic operation
- Emit low-confidence candidate facts rather than authoritative conclusions.

### Flexible rule
- If System 1 starts acting like a full solver, shrink it back down and make it serve only as a grounding component.

## Phase 3: Strengthen System 2 as the fast forward engine

### Objectives
- Make System 2 the main engine for structured, known-rule reasoning.
- Solve the easy and semi-structured arithmetic cases directly.

### Priority tasks
- Define symbolic primitives for:
  - quantities
  - events
  - state facts
  - arithmetic operations
- Add forward rules for:
  - addition and subtraction
  - multiplication and division
  - inventory / ownership shifts
  - state transitions over time
  - comparison and remainder logic
- Add conflict detection when new facts contradict existing ones.
- Use confidence values to decide whether a fact is accepted or deferred.

### Flexible rule
- System 2 may learn simple local forward rules from recurring patterns, but it should remain efficient and deterministic.

## Phase 4: Strengthen System 3 as the goal-directed reasoner

### Objectives
- Build explicit goal decomposition and search capabilities.
- Use System 3 when forward reasoning is insufficient.

### Priority tasks
- Parse queries into subgoals, for example:
  - final amount
  - total cost
  - remaining stock
  - difference between quantities
  - ratio or fraction tasks when relevant
- Add recursive rule chaining to resolve subgoals.
- Determine when a query fails because of:
  - missing fact
  - missing rule
  - wrong entailment path
  - contradiction or ambiguity
- Use abduction when the missing information is likely learnable.

### Flexible rule
- System 3 should trigger ILP on demand when search stalls, rather than running a heavy learner continuously.

## Phase 5: Make ILP a shared rule-discovery engine

### Objectives
- Keep rule synthesis as a distinct learning component, not as a single system-specific hack.
- Let ILP propose new rules that can be used by either symbolic system.

### Priority tasks
- Define a compact rule schema with:
  - head predicate
  - body conditions
  - variables
  - arithmetic expression
  - confidence
  - scope / applicability
- Learn rules from solved examples, especially repeated patterns.
- Infer transformation rules and arithmetic derivations from prior successful traces.
- Store induced rules in a shared memory layer.

### Flexible rule
- System 2 may use ILP to create local forward rules.
- System 3 may use ILP to create deeper backward rules or missing goal-resolution patterns.
- The same ILP substrate can serve both, but the usage policy may differ.

## Phase 6: Build LTM as a real long-term memory layer

### Objectives
- Support reuse of rules, solved patterns, and contradictions across problem runs.

### Priority tasks
- Distinguish memory categories:
  - fact memory
  - rule memory
  - episodic memory
  - contradiction / revision memory
- Add retrieval by:
  - similarity to current query
  - predicate structure
  - variable pattern
  - prior success / confidence
- Add rule aging, pruning, and confidence calibration.

### Flexible rule
- LTM should be a service or component, not a single closed object. The design can adapt as the rule taxonomy becomes clearer.

## Phase 7: Rank bottlenecks and iterate by evidence

### Targeted bottlenecks to investigate in order
1. Parsing and proposition extraction
2. Quantity and state normalization
3. Arithmetic invariants and symbolic operations
4. Goal decomposition and subgoal search
5. Rule induction quality
6. Memory consolidation and retrieval quality

### Benchmarking approach
- Use GSM8K as the main benchmark.
- Evaluate the system along several axes:
  - exact answer accuracy
  - solved by direct forward reasoning
  - solved by deeper goal search
  - solved after ILP-induced rule learning
  - failure by parsing / grounding / reasoning / conflict

### Flexible rule
- We should not insist on a fixed architecture forever. The project should adapt to evidence from benchmark failures.

## Phase 8: Use background cognitive_loop only when the system is stable

### Use background runs when:
- the code is stable enough to interpret
- you want longer benchmark sweeps
- you want to accumulate rule-learning effects over time
- the trace output is already useful for debugging

### Avoid background runs when:
- the core rules are changing quickly
- the memory schema is in active flux
- results are hard to attribute to a specific change

### Practical recommendation
- During active development, prefer small, targeted runs with full logs.
- Once a stable benchmark slice exists, use background cognitive_loop for longer experiments.

## Expected trajectory

The likely path is:

1. System 1 becomes smaller and more precise
2. System 2 becomes the workhorse for structured inference
3. System 3 becomes the deeper planner and resolver
4. ILP + LTM become the learning and memory engine
5. The orchestrator becomes a disciplined self-improvement loop rather than a demo-only loop

This is the best high-level direction for a symbolic-first AGI prototype aimed at GSM8K and beyond.

## Current priority

The next work should focus on:

- benchmark harness + traces
- structured parsing and proposition extraction
- stronger System 2 rules for arithmetic and state changes
- System 3 goal decomposition and search

## Milestone: micro-step semantic layer is now in place

This milestone marks a conceptual and practical shift in the project:

- the cognitive loop is running end-to-end on real GSM8K examples
- the working-memory schema has been normalized around a compact symbolic vocabulary
- the compiler now emits finer-grained semantic facts such as:
  - Entity / Relation
  - Quantity / Has / ScaleFactor / Rate / Duration
  - Add / Subtract / Multiply / Divide / Percent / Ratio / UnitConvert
- System 2 can successfully evaluate representative arithmetic/state patterns in a micro-step, compositional way
- verified examples include:
  - Janet duck-egg profit: solved
  - half-that-much total quantity: solved
  - repeated weekly total: solved

### Benchmark snapshot

Representative cases solved by the current layer:
- Janet duck eggs: expected 18.0, solved=True
- Half-that-much total: expected 3.0, solved=True
- Weekly repeated total: expected 540.0, solved=True

### Interpretation

This milestone shows that the architecture is moving in the right direction: the system is no longer dependent on a small set of hardcoded problem templates. It is increasingly working from a stable, compositional symbolic substrate rather than from ad hoc GSM8K archetypes.

The main remaining bottleneck is not the orchestrator or the general engine. It is the semantic grounding layer: turning more GSM8K natural-language patterns into the explicit micro-step equation vocabulary that System 2 and 3 can reason about.

The next work is therefore to widen the micro-step language systematically, focusing on:
- percentage and discount semantics
- chained ratio relations
- rate × duration and unit-conversion tasks
- conditional / branch logic over repeated events
- equation solving for unknown quantities

This milestone is intentionally a checkpoint, not a completion marker. It reflects a materially better symbolic substrate for the next benchmark-driven iteration.
- ILP rule induction from solved traces
- LTM storage of induced rules and successful patterns

These steps are the best way to make real progress without over-investing in a large neural reasoner that is not yet the main bottleneck.
