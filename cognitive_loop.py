import concurrent.futures
import time
from dataclasses import dataclass
from typing import Any

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'forward-engine'))

from system2_poc import System2Engine, Dog, Person, Perception, Belief

# ==========================================
# GWT Unified Fact Representation
# ==========================================
@dataclass(frozen=True)
class WMNode:
    """A purely symbolic, library-agnostic fact in the Global Workspace."""
    concept: str
    attrs: frozenset

    @classmethod
    def create(cls, concept: str, **kwargs):
        # Sort items to ensure deterministic hashing
        return cls(concept, frozenset(sorted(kwargs.items())))
    
    def to_dict(self):
        return dict(self.attrs)

# ==========================================
# Working Memory (Global Workspace)
# ==========================================
class WorkingMemory:
    def __init__(self, default_ttl=3):
        # Dictionary mapping: WMNode -> remaining cycles (TTL)
        self.facts = {}
        self.default_ttl = default_ttl

    def update(self, new_facts):
        """Merges new deductions and resets their TTL to max."""
        if new_facts:
            for fact in new_facts:
                self.facts[fact] = self.default_ttl

    def decay(self):
        """Decrements TTL for all facts and forgets expired ones."""
        expired = []
        for fact in self.facts:
            self.facts[fact] -= 1
            if self.facts[fact] <= 0:
                expired.append(fact)
        
        for fact in expired:
            del self.facts[fact]

# ==========================================
# Sub-Systems
# ==========================================
def system1_step(current_wm, cycle):
    """Neural logic / Fast Perception."""
    new_facts = set()
    # Simulate System 1 seeing a dog and a person ONLY during cycles 1 and 2
    if cycle <= 2:
        new_facts.add(WMNode.create("Dog", state="barking", prob=0.90))
        new_facts.add(WMNode.create("Person", state="walking", prob=0.99))
        new_facts.add(WMNode.create("Perception", label="motion_detected", prob=0.60))
    return new_facts

def system2_step(current_wm, cycle):
    """Experta / Rete Engine - TRUE INTEGRATION."""
    # 1. Initialize a stateless engine for this tick
    engine = System2Engine()
    engine.reset()
    
    # 2. Map agnostic GWT nodes into Experta Objects
    for node in current_wm:
        # Filter out any internal keys just to be safe
        kwargs = {k: v for k, v in node.to_dict().items() if not str(k).startswith("__")}
        if node.concept == "Dog":
            engine.declare(Dog(**kwargs))
        elif node.concept == "Person":
            engine.declare(Person(**kwargs))
        elif node.concept == "Perception":
            engine.declare(Perception(**kwargs))
        elif node.concept == "Belief":
            engine.declare(Belief(**kwargs))
            
    # 3. Fire the Rete algorithm
    engine.run()
    
    # 4. Map deduced Experta Beliefs back into agnostic GWT Nodes
    new_facts = set()
    for f in engine.facts.values():
        # Only extract derived Beliefs to avoid infinite feedback loops
        if type(f).__name__ == "Belief":
            # Strip out Experta internal fields like __factid__
            clean_attrs = {k: v for k, v in f.items() if not str(k).startswith("__")}
            new_facts.add(WMNode.create("Belief", **clean_attrs))
            
    return new_facts

def system3_step(current_wm, cycle):
    """ProbLog / Deep inference."""
    new_facts = set()
    
    # Simulate System 3 forming a plan if System 2 deduced human activity
    for node in current_wm:
        if node.concept == "Belief" and node.to_dict().get("label") == "human_activity":
            new_facts.add(WMNode.create("Action", type="observe"))
            
    return new_facts

# ==========================================
# Main Cognitive Orchestrator
# ==========================================
def run_cognitive_loop(max_cycles=5):
    wm = WorkingMemory(default_ttl=2) 
    print("--- Booting AGI Synchronous Loop (with TRUE System 2 Integration) ---")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        for cycle in range(1, max_cycles + 1):
            # Print current WM state
            print(f"\n[Cycle {cycle}] Current WM:")
            for fact, ttl in wm.facts.items():
                print(f"  - {fact.concept} {dict(fact.attrs)} (TTL: {ttl})")
                
            if cycle == 3:
                print("   -> (System 1 stops perceiving the entities!)")
            
            # 1. BROADCAST
            # Convert dict_keys to a set so it can be pickled and sent across processes
            current_wm_snapshot = set(wm.facts.keys())
            f1 = executor.submit(system1_step, current_wm_snapshot, cycle)
            f2 = executor.submit(system2_step, current_wm_snapshot, cycle)
            f3 = executor.submit(system3_step, current_wm_snapshot, cycle)
            
            out1, out2, out3 = f1.result(), f2.result(), f3.result()
            
            # 2. UPDATE
            wm.update(out1)
            wm.update(out2)
            wm.update(out3)
            
            # 3. DECAY
            wm.decay()
            
            time.sleep(0.5)

    print(f"\n--- Halt ---")

if __name__ == "__main__":
    run_cognitive_loop()
