# Tri-System Neurosymbolic Architecture

This document describes the broader cognitive architecture into which modules like the Logic Transformer (LT) can be integrated.

The architecture aims to combine neural learning with symbolic reasoning by separating cognition into three distinct systems, all unified by a **Shared Working Memory (Global Workspace)**.

## Architecture Overview

### The Global Workspace (Working Memory)
Based on Global Workspace Theory (GWT), the architecture utilizes a central, shared Working Memory (WM). All three systems act as independent, specialized processors that observe the workspace and broadcast their findings back to it. This acts as the "conscious" state of the AI, holding active perceptions, deduced beliefs, and current goals.

### Synchronous Cognitive Cycle
To maximize efficiency and avoid complex distributed race conditions, the engine operates on a discrete, synchronous **Cognitive Cycle** (similar to the LIDA cognitive model):
1. **Broadcast:** At the start of a "tick", the current state of the Working Memory is broadcast to all three systems simultaneously.
2. **Compute & Wait:** System 1, 2, and 3 run in parallel (on separate CPU cores). The Orchestrator waits for all three systems to return their single-step deductions.
3. **Update:** The Orchestrator merges the outputs from all three systems back into the Working Memory, ready for the next cycle.

This pure-Python orchestrator approach avoids network/IO latency and simplifies the integration of drastically different reasoning paradigms.

### 1. System 1 (Logic Transformer / Neural Logic):
   - **Role:** Fast, differentiable, fuzzy logic rule application.
   - **Mechanism:** Uses gradient-based representation learning to process continuous semantic data. Acts as the intuitive, pattern-matching layer that maps raw perception into latent logical forms, broadcasting initial facts to the Working Memory.

### 2. System 2 (Rete Engine / Forward Chaining):
   - **Role:** Deterministic, discrete, and efficient pattern matching over established facts.
   - **Mechanism:** Observes the Working Memory for crystallized facts (e.g., from System 1). It uses a Rete network to rapidly deduce immediate logical consequences (forward chaining) and broadcasts new beliefs back to the workspace.

### 3. System 3 (ProbLog / Backward Chaining):
   - **Role:** Deep, complex reasoning, handling uncertainty, and goal-directed planning.
   - **Mechanism:** A probabilistic backward-chaining engine that watches the Working Memory for complex anomalies, uncertainties, or high-level goals. It performs deep hypothesis search and updates the Working Memory with resolved plans or probabilistic conclusions.

## Ontology and Memory Management

To prevent the combinatorial explosion of facts typically associated with classical symbolic AI (e.g., asserting a "dog" is an "animal", "organism", "object", etc.), the architecture employs a multi-tiered strategy for ontology resolution and memory fading:

### 1. Zero-Fact Ontology Unification (System 2)
Instead of cluttering Working Memory with sprawling taxonomic graphs, System 2 leverages native Python class inheritance within the Rete engine (Experta). A fact asserted as a `Dog` natively matches rules looking for an `Animal` via `isinstance()` checks. This allows for instantaneous ontological resolution without generating redundant facts.

### 2. Hyperbolic Embeddings (System 1)
System 1 utilizes hyperbolic space (such as Poincaré embeddings) to model hierarchical taxonomies natively. Because hyperbolic space accurately represents tree-like structures, System 1 can perform soft unification and output the most contextually salient abstraction directly to Working Memory, avoiding massive ontological dumps.

### 3. Memory Decay (Time-To-Live)
Working Memory operates with an integer-based Time-To-Live (TTL) decay mechanism. Facts are assigned a TTL in terms of cognitive cycles. If a fact is not actively reinforced (re-perceived by System 1 or re-deduced by Systems 2/3), its TTL decrements each cycle until it drops to zero and is forgotten. This "activation energy" limits memory bloat and keeps the Global Workspace focused on salient information.

By separating these concerns, the architecture leverages the strengths of differentiable neural networks for noisy input handling (System 1) while retaining the correctness, interpretability, and powerful search capabilities of classical symbolic engines (Systems 2 & 3).