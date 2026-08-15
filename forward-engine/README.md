# System 2: Deliberative Cognitive Engine

This module implements the System 2 cognitive layer for the AGI prototype, acting as a bridge between the fast, reactive neural processing of System 1 and the deep, Bayesian deliberation of System 3.

## Overview

System 2 utilizes **Experta**, a Rete-based forward-chaining rules engine written in Python. Experta is highly hackable, which allows us to natively inject rudimentary fuzzy logic and probabilistic truth values (certainty factors) into rule matching via Python lambdas.

### Responsibilities:
- Receive streams of `Perception` facts from System 1.
- Apply boolean and rudimentary probabilistic logic to form `Belief` facts.
- Filter and prioritize information.
- Escalate complex, uncertain scenarios to System 3 (via ProbLog) for heavy Bayesian inference.

## Quick Start

Run the proof-of-concept to see the basic mechanics of perception handling, confidence calculation, and System 3 escalation:

```bash
python system2_poc.py
```

## Architecture

- `system2_poc.py`: Contains the core `System2Engine` utilizing `experta`. 
- **System 1 Integration**: System 1 declares `Perception` facts directly into the Experta engine.
- **System 3 Integration**: Rules that require high-level deliberation trigger outputs that are pushed to System 3 (e.g., via the `inbox_entries` queue or ProbLog direct integration).

*Note: The Experta library relies on older `collections.Mapping` references. `system2_poc.py` includes a compatibility patch for modern Python versions.*
