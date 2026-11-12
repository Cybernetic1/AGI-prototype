# Training Regimen: Semantic-AR First (v1)

## Scope
- Primary objective: semantic-AR reasoning quality and transfer.
- GSM/simple arithmetic role: diagnostic-only probe.
- Hardware target: RTX 5090 32GB.
- Corpus cap: 300 GB.

## Corpus plan (capped pack)
- **FineWeb (filtered/deduped): 45%**
- **C4 (cleaned subset): 30%**
- **RedPajama (quality-filtered subset): 20%**
- **Curated Common Crawl slices: 5%** (strict quality filtering only)

All sources should be deduplicated, language-filtered, and decontaminated against held-out diagnostic sets before training.

## Semantic-AR worked-example supervision
- Use `semantic_ar_worked_examples_v1.json` as structured curriculum.
- 180 traces total (30 per AGI category bucket):
  - physical/inventory
  - quantitative/arithmetic
  - temporal/event
  - causal/intervention
  - social/intent
  - epistemic/defeasible

## Training phases
1. **Phase 1: semantic-AR pretraining**
   - Objective mix: proposition retrieval, continuation ranking, hard-negative separation.
   - Emphasis: lexical-role alignment + structural consistency.

2. **Phase 2: rule-seeded joint training**
   - Inputs: semantic-AR tuples + seed-rule templates from worked traces.
   - Objective mix: semantic-AR + rule activation consistency + contradiction penalty.

3. **Phase 3: lightweight diagnostics**
   - Run simple arithmetic and GSM-style probes only as diagnostics.
   - No optimization for benchmark-specific style mimicry.

## Runtime/system constraints
- Mixed precision (bf16/fp16), gradient checkpointing, accumulation for memory fit.
- Sequence-length curriculum (short-to-long) to stabilize training under 32GB VRAM.
- Periodic LTM sleep windows:
  - episode write,
  - semantic consolidation,
  - forgetting pass,
  - rule-priority retuning.

## Readiness gates before scale-up
- Chain success rate >= 0.85
- Dead-end ratio <= 0.10
- Contradiction ratio <= 0.05
- Shape coverage rate >= 0.95

Training should only scale once `chain_completeness_eval_v1.json` reports all gates green.
