Research report: sweep-based learning, rule seeding, and hybrid LT/ILP bootstrapping

Executive summary

The right long-term architecture is not “pure hand-coded rules versus pure ILP.” It is a staged hybrid: keep a small invariant layer fixed, let ILP search a narrow symbolic space seeded by local templates and priors, and widen the search only when the curriculum and validation signal justify it. In the current AGI prototype this is already the pattern: the semantic graph, state machine, finality contract, and owner/item binding are explicit invariants; ILP is used on top of that grounded substrate to learn local glue patterns, not to reconstruct the whole reasoning problem from raw text.

This matches Metagol-style ILP more closely than it does a full unconstrained learner: metarules constrain legal clause forms, and a hyperbolic/Lorentz prior biases search toward plausible predicate interactions. The main practical insight is that a sweep-based bootstrap works only when the search space is small and the predicates are already meaningfully canonicalized. Without that, ILP does not become “faster”; it becomes a noisy global learner with too much combinatorial freedom.

Classification

This is a technical deep-dive / conceptual design question. The answer is architectural, not a step-by-step usage guide. The main issue is how to bias symbolic rule discovery so a small, interpretable learner can bootstrap a larger reasoning system.

Research findings

1) The public ILP literature is consistent with a template-first design

The strongest public signal is the Metagol family of ILP systems. Metagol’s core idea is not “learn arbitrary logic from scratch,” but “learn from a small set of second-order metarules that define legal clause shapes.” These metarules are the structural prior. This is effectively a way to seed the learner with the forms of the rules it is allowed to synthesize.

This is exactly the kind of search scoping the AGI prototype needs. The current code already contains a small local learner, and the plan is to keep the invariant layer fixed while letting ILP work on local symbolic glue rather than the whole problem.

Evidence:
- metagol/metagol:README.md:51-123
- metagol/metagol:metagol.pl:28-66
- LogtalkDotOrg/logtalk3:ports/metagol/loader.lgt:29-38
- AGI-prototype/plan.md:1-40
- AGI-prototype/ilp_learner/metagol_nds.py:114-290

2) The AGI prototype already uses a layered scheme that matches “invariant layer + local glue” thinking

The repository’s architecture separates:
- a fixed semantic schema and working-memory graph,
- a state machine with explicit event ordering and final-state semantics,
- a learned local layer for rule induction and pattern synthesis.

This is the correct direction. The hardest reasoning problems are not arithmetic but semantic grounding, state ownership, provenance, and temporal finality. Those are not “rules to learn from zero.” They are invariants. Once they are fixed, ILP can search a much smaller rule set on top of them.

Evidence:
- AGI-prototype/cognitive_loop.py:15-61
- AGI-prototype/working_memory/semantic_schema.py:1-111
- AGI-prototype/working_memory/working_memory.py:62-181
- AGI-prototype/cognitive_loop.py:203-312
- AGI-prototype/plan.md:1-40

3) The real bootstrap pattern is a curriculum over rule complexity, not a single huge sweep

The practical strategy is:
- start with a tiny semantic family benchmark,
- keep the invariant layer stable,
- widen rule coverage one family at a time,
- only then broaden to a larger GSM8K-like slice.

This is a curriculum, not a brute-force broad benchmark sweep. It is consistent with how symbolic learners are built in practice: start narrow, add search bias, then widen.

Evidence:
- AGI-prototype/plan.md:1-12
- AGI-prototype/plan.md:64-87
- AGI-prototype/ROADMAP.md:19-41
- AGI-prototype/ilp_learner/hyperparam_sweep.py:1-115

4) Search bias is the critical ingredient

The project is already leaning toward the right set of levers:
- bounded clause count,
- beam width,
- prior weight,
- local templates / metarules,
- hyperbolic / Lorentz embedding prior,
- small canonical predicate vocabulary.

These are not random tricks; they are explicit forms of structural regularization. They reduce the search space enough that the learner can discover the right local rules before the global space becomes unmanageable.

Evidence:
- AGI-prototype/ilp_learner/metagol_nds.py:176-290
- AGI-prototype/ilp_learner/hyperparam_sweep.py:10-22
- AGI-prototype/ilp_learner/abductive_ilp.py:164-220

5) LT-side and ILP-side roles are complementary, not competing

The Logic Transformer side can act as a structured proposal generator, while the symbolic ILP stage acts as a constrained rule synthesizer. This is a classic complementary design:
- LT / System 1 proposes structured candidate facts or predicate-argument mappings with soft confidence,
- the semantic schema and working-memory graph enforce canonical structure,
- System 2 validates and reduces the graph,
- System 3 learns a small symbolic rule fragment when the invariant layer is stable enough.

The LT side should not be asked to invent arbitrary logic program clauses directly. Instead, it should propose predicates, relations, and local variable bindings that fit a constrained schema. ILP then turns those proposals into legal symbolic rules, with a small set of accepted templates and priors.

Evidence:
- AGI-prototype/cognitive_loop.py:84-136
- AGI-prototype/working_memory/semantic_schema.py:17-59
- AGI-prototype/ilp_learner/metagol_nds.py:77-120
- AGI-prototype/logic_transformer/logic_transformer.py:18-84

6) “Clamping” can help, but only within a fixed symbolic grammar

A useful pattern is to clamp or constrain a rule to a specific predicate family or semantic template, not to arbitrary free-form symbol generation. In other words:
- clamp the LT output to a known predicate vocabulary,
- clamp the rule head to a canonical output class,
- allow the learner to vary only in the argument bindings and local substructure.

This is a much more manageable search than a full unconstrained predicate synthesis problem. It is also why the repo’s canonical predicate vocabularies and fixed semantic kind tables are so important: they reduce the space from “arbitrary symbolic language” to “valid symbolic grammar.”

Evidence:
- AGI-prototype/cognitive_loop.py:15-61
- AGI-prototype/working_memory/semantic_schema.py:17-59
- AGI-prototype/ilp_learner/generate_and_run_gsm8k_ilp.py:21-74
- AGI-prototype/ilp_learner/hyperparam_sweep.py:36-78

7) Genetic algorithms are possible, but are not the cleanest early bootstrap mechanism here

A genetic algorithm can mutate and crossover rule bodies, but it is a weaker fit than ILP + metarules for a structured symbolic domain that already has a known grammar. It may be useful later as a second-stage search over candidate rule sets, especially for broader families. But in the short run, a beam search over metarules and local templates is more controlled and more interpretable.

The project should treat genetic search as an optional optimizer for a rule-set population, not as the primary induction mechanism during the bootstrap phase.

8) What this means for the current project

The realistic path is:
- keep the invariant layer explicit and small,
- bias ILP with local templates and hyperbolic priors,
- run narrow sweeps over semantic families and LT tasks instead of a giant broad benchmark,
- expand rule templates gradually as the vocabulary stabilizes,
- only then consider a larger GSM8K or GPU-backed run if the search-space bias is not enough.

This matches the repo’s current strategy and is consistent with the technical direction in the project plan.

Proposed practical plan

- Phase A: keep the invariant logic fixed and heavily tested.
- Phase B: add one or two code-generated local rule families via ILP over the grounded semantic graph.
- Phase C: run a focused sweep on 100-500 LT examples with beam and prior-weight variation.
- Phase D: add negative examples and held-out validation; increase search breadth only when precision is not just coverage.
- Phase E: consider GPU if the neural-side proposal model becomes the bottleneck rather than the symbolic learner.

 Confidence assessment

This research is moderately high confidence for the architectural recommendation and the Metagol-like bias pattern. The strongest source-backed conclusions are about rule seeding via metarules and template priors. The more speculative part is the exact LT clamping architecture, because public code search results were sparse for direct LT implementations. The local repo evidence is strong for the AGI prototype’s own design, but broader open-source LT implementations were not abundant in the indexed public search results.

References and footnotes

[^1]: metagol/metagol:README.md:51-123
[^2]: metagol/metagol:metagol.pl:28-66
[^3]: LogtalkDotOrg/logtalk3:ports/metagol/loader.lgt:29-38
[^4]: AGI-prototype/plan.md:1-40
[^5]: AGI-prototype/ROADMAP.md:19-41
[^6]: AGI-prototype/ilp_learner/metagol_nds.py:114-290
[^7]: AGI-prototype/ilp_learner/hyperparam_sweep.py:1-115
[^8]: AGI-prototype/ilp_learner/abductive_ilp.py:164-220
[^9]: AGI-prototype/cognitive_loop.py:15-61
[^10]: AGI-prototype/working_memory/semantic_schema.py:1-111
[^11]: AGI-prototype/working_memory/working_memory.py:62-181
[^12]: AGI-prototype/cognitive_loop.py:203-312
[^13]: AGI-prototype/logic_transformer/logic_transformer.py:18-84

Summary of the key answer

The right answer is: yes, sweep-based learning is the right long-run direction, but it must be seeded by an invariant symbolic substrate and a structured grammar. The best bootstrap is a local symbolic learner with metarules, templates, and priors, not a full unconstrained ILP or a raw LT rule generator. The long-term architecture is hybrid, and it is exactly how the AGI prototype is already converging.
