import sys
from pathlib import Path
import time

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from working_memory.working_memory import WMNode, WorkingMemory
from cognitive_loop import run_cognitive_loop
from ilp_learner.abductive_ilp import DuckEgg, ChickenEgg, Egg

def run_neurosymbolic_demo():
    print("=" * 70)
    print("Neurosymbolic Online Learning & Competitive Workspace Demo")
    print("=" * 70)
    
    # 1. Initialize Working Memory with Background Facts & Target Query
    # (No hardcoding in orchestrator - we inject them directly into Workspace!)
    wm = WorkingMemory(default_ttl=5)
    
    # Setup Janet's Egg Scenario in GWT
    wm.facts[WMNode.create("Produced", arg0="janet", arg1=DuckEgg, arg2=16)] = 5
    wm.facts[WMNode.create("Consumed", arg0="janet", arg1=DuckEgg, arg2=7)] = 5
    wm.facts[WMNode.create("Sells", arg0="janet", arg1=DuckEgg, arg2=2)] = 5
    
    # Target Query injected into Workspace: We want to solve Janet's daily profit (expected: 18)
    # This acts as our goal / objective
    wm.facts[WMNode.create("Query", goal="daily_profit", args=("janet", DuckEgg, "?ans"), expected_value=18.0)] = 5
    
    # 2. Run the Cognitive Orchestrator
    # We will pass a dummy input sentence. System 1 (DLN) will output a fuzzy guess,
    # and System 3 (ILP) will step in to learn the exact rule and solve it.
    print("\nStarting Cognitive Orchestrator Loop...")
    
    # We run the loop manually to illustrate cycle-by-cycle competitive dynamics
    from cognitive_loop import system1_step, system2_step, system3_step
    
    for cycle in range(1, 4):
        print(f"\n--- [Cycle {cycle}] ---")
        print("Current Working Memory Workspace:")
        for fact, ttl in list(wm.facts.items()):
            if fact.concept in ["Query", "Belief", "Produced", "Consumed", "Sells"]:
                # Render node and its continuous salience/activation score
                print(f"  - {fact.concept}({', '.join(f'{k}={v}' for k, v in fact.to_dict().items())}) [Salience: {fact.activation:.2f}]")
        
        # Pulse WMEs back and forth
        snapshot = set(wm.facts.keys())
        
        # System 1 (Neural) tries to guess
        # Let's say it outputs a fuzzy, lower-confidence guess
        out1 = set()
        if cycle == 1:
            print("\n[System 1] DLN outputs fuzzy, intuitive guess with lower confidence (0.85)...")
            out1.add(WMNode.create("Belief", label="daily_profit", value=19.0, activation=0.85))
            
        # System 2 (Rete) and System 3 (ILP / Reasoning) execute
        out2 = system2_step(snapshot, cycle, None, quiet=True)
        out3 = system3_step(snapshot, cycle, None, quiet=False)
        
        # Merge all pulsed WMEs back into the Workspace
        wm.update(out1)
        wm.update(out2)
        wm.update(out3)
        wm.decay()
        
        time.sleep(0.5)

    print("\n" + "=" * 70)
    print("Demo Completed Successfully!")
    print("=" * 70)

if __name__ == "__main__":
    run_neurosymbolic_demo()
