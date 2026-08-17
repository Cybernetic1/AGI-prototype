import concurrent.futures
import time
from typing import Any
import sys
import os
import torch
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / 'forward_engine'))
sys.path.append(str(BASE_DIR / 'logic_transformer'))
sys.path.append(str(BASE_DIR / 'ilp_learner'))

from working_memory.working_memory import WMNode, WorkingMemory
from system2_poc import System2Engine
from experta import Fact
from spacy_logical_form import SpacyLogicalFormParser
from train_pot_dln_pointer import DLNPointerDecoder
import train_pot_clause as tpc
from abductive_ilp import KnowledgeBase, OntologicalReasoner, OntologicalILPLearner

# ==========================================
# Sub-Systems
# ==========================================
_dln_model = None
_dln_vocab = None
_rev_vocab = None

def load_dln(quiet=False):
    global _dln_model, _dln_vocab, _rev_vocab
    if _dln_model is not None:
        return
        
    if not quiet:
        print("[System 1] Loading Trained Logic Transformer (DLNPointerDecoder)...")
    vocab_path = BASE_DIR / "logic_transformer" / "data" / "gsm8k_vocab.json"
    ckpt_path = BASE_DIR / "logic_transformer" / "models" / "dln_gsm8k_best.pt"
    
    with open(vocab_path, "r") as f:
        _dln_vocab = json.load(f)
        _rev_vocab = {v: k for k, v in _dln_vocab.items()}
        
    _dln_model = DLNPointerDecoder(
        input_vocab=len(_dln_vocab), 
        max_positions=151, 
        hidden_dim=128, 
        num_rules=16
    )
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    _dln_model.load_state_dict(checkpoint["model_state"])
    _dln_model.eval()

def system1_step(current_wm, cycle, input_text=None, quiet=False):
    new_facts = set()
    
    if cycle == 1 and input_text:
        # 1. Parse raw structural features via SpacyLogicalFormParser directly (acting as our semantic grounder)
        parser = SpacyLogicalFormParser()
        lf = parser.parse(input_text)
        
        def filter_core_clauses(logical_form_text):
            from spacy_logical_form import canonicalize_form, parse_clause_line
            clauses = []
            for line in canonicalize_form(logical_form_text).splitlines():
                parsed = parse_clause_line(line)
                if parsed is None: continue
                pred, args = parsed
                if pred in {"entity", "type", "tense", "question", "quantifier", "query_kind", "text"}:
                    continue
                clauses.append({"pred": pred, "args": list(args)})
            return clauses
            
        core_clauses = filter_core_clauses(lf.render())
        
        if not quiet:
            print(f"[System 1] Semantic Grounding Complete. Extracted {len(core_clauses)} semantic logic predicates.")
        
        # Map the output strings back to WMNodes with soft activation (0.85) representing System 1's neural confidence
        for clause in core_clauses:
            pred = clause["pred"]
            args = clause["args"]
            concept = pred.capitalize()
            if concept in ["Event", "Predicate", "Agent", "Patient", "Recipient", "Name", "Quantity", "Location", "Time", "Modifier"]:
                kwargs = {f"arg{j}": arg for j, arg in enumerate(args)}
                new_facts.add(WMNode.create(concept, activation=0.85, **kwargs))
            
    return new_facts

def system2_step(current_wm, cycle, _=None, quiet=False):
    engine = System2Engine()
    engine.reset()
    
    from system2_poc import Predicate, Agent, Recipient, Patient, Name, Quantity, Inventory, Belief, EventProcessed, Location, Time, Modifier
    CLASS_MAP = {
        "Predicate": Predicate, "Agent": Agent, "Recipient": Recipient, 
        "Patient": Patient, "Name": Name, "Quantity": Quantity, 
        "Inventory": Inventory, "Belief": Belief, "EventProcessed": EventProcessed,
        "Location": Location, "Time": Time, "Modifier": Modifier
    }
    
    for node in current_wm:
        kwargs = {k: v for k, v in node.to_dict().items() if not str(k).startswith("__")}
        fact_class = CLASS_MAP.get(node.concept)
        if not fact_class:
            fact_class = type(node.concept, (Fact,), {})
        engine.declare(fact_class(**kwargs))
            
    engine.run()
    
    new_facts = set()
    for f in engine.facts.values():
        if type(f).__name__ in ["Belief", "Inventory", "EventProcessed"]:
            clean_attrs = {k: v for k, v in f.items() if not str(k).startswith("__")}
            
            # Preserving the original node's activation/salience score from Working Memory
            concept = type(f).__name__
            temp_node = WMNode.create(concept, **clean_attrs)
            original = next((n for n in current_wm if n == temp_node), None)
            activation = original.activation if original is not None else 1.0
            
            new_facts.add(WMNode.create(concept, activation=activation, **clean_attrs))
            
    return new_facts

def system3_step(current_wm, cycle, _=None, quiet=False):
    new_facts = set()
    
    # 1. Search for any Query node currently in Working Memory
    query_node = next((n for n in current_wm if n.concept == "Query"), None)
    if not query_node:
        return new_facts
        
    q_attrs = query_node.to_dict()
    goal_pred = q_attrs.get("goal")
    goal_args = list(q_attrs.get("args", []))
    expected_value = q_attrs.get("expected_value")
    
    if not goal_pred or not goal_args:
        return new_facts
    
    # 2. Generate background KB dynamically from current Working Memory!
    kb = KnowledgeBase()
    for node in current_wm:
        # Declare all factual relationships from GWT as logical KB facts
        attrs = node.to_dict()
        pred = node.concept.lower()
        
        # Pull args safely
        args = [attrs.get(f"arg{i}") for i in range(len(attrs))]
        args = [a for a in args if a is not None]
        if args:
            kb.declare_fact(pred, *args)
            
    # Always possess baseline reasoning rules
    kb.declare_rule(
        head_pred="daily_profit",
        head_args=["?p", "?item", "?ans"],
        body=[
            ("sells", ["?p", "?item", "?price"]),
            ("sold_qty", ["?p", "?item", "?qty"])
        ],
        calc_expr="qty * price",
        calc_target="?ans"
    )
    
    # 3. Load dynamically induced rules currently stored in Working Memory!
    for node in current_wm:
        if node.concept == "Inducedrule":
            r = node.to_dict()
            kb.declare_rule(
                head_pred=r["head_pred"],
                head_args=list(r["head_args"]),
                body=[(b[0], list(b[1])) for b in r["body"]],
                calc_expr=r["calc"],
                calc_target=r["target"]
            )
        
    # Solve general query
    reasoner = OntologicalReasoner(kb)
    solutions = reasoner.solve(goal_pred, goal_args)
    
    success = False
    if solutions:
        for sol in solutions:
            ans_val = sol.get("ans")
            if ans_val is not None:
                success = True
                # Add solved belief with maximum activation/salience (1.0)
                new_facts.add(WMNode.create("Belief", label=goal_pred, value=ans_val, activation=1.0))
                if not quiet:
                    print(f"[System 3] Solved query via reasoning: {goal_pred} = {ans_val}")
                    
    if not success and expected_value is not None:
        # If proof failed, execute Abductive ILP to learn rule!
        learner = OntologicalILPLearner(kb)
        induced = learner.abduce_and_induce([(goal_pred, goal_args, expected_value)])
        if induced:
            # Broadcast the induced rule to Working Memory as a Rule Node!
            new_facts.add(WMNode.create(
                "Inducedrule",
                activation=1.0,
                head_pred=induced["head"][0],
                head_args=tuple(induced["head"][1]),
                body=tuple((b[0], tuple(b[1])) for b in induced["body"]),
                calc=induced["calc"],
                target=induced["target"]
            ))
            if not quiet:
                print(f"[System 3] Successfully induced general rule online! Saved to Working Memory.")
                
    return new_facts

# ==========================================
# Main Cognitive Orchestrator
# ==========================================
def run_cognitive_loop(input_text, max_cycles=4, quiet=False):
    wm = WorkingMemory(default_ttl=5)
    
    # Initialize World State
    wm.facts[WMNode.create("Inventory", owner="john", item="apple", qty=10)] = 5
    wm.facts[WMNode.create("Inventory", owner="mary", item="apple", qty=0)] = 5
    
    if not quiet:
        print(f"--- Booting AGI Synchronous Loop (DLN GSM8K Inference) ---")
        print(f"Initial State: John has 10 apples, Mary has 0 apples.")
        print(f"Input: '{input_text}'\n")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        for cycle in range(1, max_cycles + 1):
            if not quiet:
                print(f"\n[Cycle {cycle}] Current State in WM:")
                for fact, ttl in wm.facts.items():
                    if fact.concept in ["Inventory", "Belief"]:
                        print(f"  - {fact.concept}({', '.join(f'{k}={v}' for k, v in fact.to_dict().items())})")
            
            current_wm_snapshot = set(wm.facts.keys())
            f1 = executor.submit(system1_step, current_wm_snapshot, cycle, input_text, quiet)
            f2 = executor.submit(system2_step, current_wm_snapshot, cycle, None, quiet)
            f3 = executor.submit(system3_step, current_wm_snapshot, cycle, None, quiet)
            
            out1, out2, out3 = f1.result(), f2.result(), f3.result()
            
            wm.update(out1)
            wm.update(out2)
            wm.update(out3)
            wm.decay()
            
            if not quiet:
                time.sleep(0.5)

    if not quiet:
        print(f"\n--- Halt ---")
        
    return wm

if __name__ == "__main__":
    gsm8k_problem = "John gave Mary 5 apples."
    run_cognitive_loop(gsm8k_problem)
