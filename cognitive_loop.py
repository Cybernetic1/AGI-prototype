import concurrent.futures
import time
from dataclasses import dataclass
from typing import Any
import sys
import os
import torch
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / 'forward-engine'))
sys.path.append(str(BASE_DIR / 'gsm8k-tests'))
sys.path.append(str(BASE_DIR / 'gsm8k-tests' / 'lt_core'))

from system2_poc import System2Engine
from experta import Fact
from spacy_logical_form import SpacyLogicalFormParser
from train_pot_dln_pointer import DLNPointerDecoder
import train_pot_clause as tpc

# ==========================================
# GWT Unified Fact Representation
# ==========================================
@dataclass(frozen=True)
class WMNode:
    concept: str
    attrs: frozenset

    @classmethod
    def create(cls, concept: str, **kwargs):
        return cls(concept, frozenset(sorted(kwargs.items())))
    
    def to_dict(self):
        return dict(self.attrs)

# ==========================================
# Working Memory (Global Workspace)
# ==========================================
class WorkingMemory:
    def __init__(self, default_ttl=3):
        self.facts = {}
        self.default_ttl = default_ttl

    def update(self, new_facts):
        if new_facts:
            for fact in new_facts:
                if fact.concept == "Inventory":
                    d = fact.to_dict()
                    to_remove = [
                        f for f in self.facts 
                        if f.concept == "Inventory" 
                        and f.to_dict().get("owner") == d.get("owner") 
                        and f.to_dict().get("item") == d.get("item")
                    ]
                    for f in to_remove:
                        del self.facts[f]
                self.facts[fact] = self.default_ttl

    def decay(self):
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
_dln_model = None
_dln_vocab = None
_rev_vocab = None

def load_dln():
    global _dln_model, _dln_vocab, _rev_vocab
    if _dln_model is not None:
        return
        
    print("[System 1] Loading Trained Logic Transformer (DLNPointerDecoder)...")
    vocab_path = BASE_DIR / "gsm8k-tests" / "lt_core" / "data" / "gsm8k_vocab.json"
    ckpt_path = BASE_DIR / "gsm8k-tests" / "lt_core" / "models" / "dln_gsm8k_best.pt"
    
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

def system1_step(current_wm, cycle, input_text=None):
    global _dln_model, _dln_vocab, _rev_vocab
    new_facts = set()
    
    if cycle == 1 and input_text:
        load_dln()
        
        # 1. Parse raw structural features via SpacyLogicalFormParser
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
        
        # Format the row as the encoder expects it
        row = {"input_props": core_clauses}
        input_list = [tpc.clause_text(p) for p in core_clauses]
        
        in_ids = torch.tensor([tpc.encode_input(row, _dln_vocab)], dtype=torch.long)
        dec_ids = torch.tensor([[0]], dtype=torch.long)
        
        out_clauses = []
        with torch.no_grad():
            for step in range(len(core_clauses) + 1):
                out = _dln_model(in_ids, dec_ids)
                logits = out[0] if isinstance(out, tuple) else out
                next_pos = logits[0, -1].argmax().item()
                
                if next_pos == in_ids.size(1):  # EOS
                    break
                    
                # Decode the pointer!
                token_id = in_ids[0, next_pos].item()
                clause = _rev_vocab.get(token_id)
                if clause and clause not in out_clauses:
                    out_clauses.append(clause)
                
                dec_ids = torch.cat([dec_ids, torch.tensor([[next_pos]])], dim=1)
                
        print(f"[System 1] DLN Latent Processing Complete. Extracted {len(out_clauses)} semantic logic predicates.")
        
        # Map the output strings back to WMNodes
        from spacy_logical_form import parse_clause_line
        for clause in out_clauses:
            parsed = parse_clause_line(clause)
            if parsed:
                pred, args = parsed
                concept = pred.capitalize()
                if concept in ["Event", "Predicate", "Agent", "Patient", "Recipient", "Name", "Quantity", "Location", "Time", "Modifier"]:
                    kwargs = {f"arg{j}": arg for j, arg in enumerate(args)}
                    new_facts.add(WMNode.create(concept, **kwargs))
            
    return new_facts

def system2_step(current_wm, cycle, _=None):
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
            new_facts.add(WMNode.create(type(f).__name__, **clean_attrs))
            
    return new_facts

def system3_step(current_wm, cycle, _=None):
    return set()

# ==========================================
# Main Cognitive Orchestrator
# ==========================================
def run_cognitive_loop(input_text, max_cycles=4):
    wm = WorkingMemory(default_ttl=5)
    
    # Initialize World State
    wm.facts[WMNode.create("Inventory", owner="john", item="apple", qty=10)] = 5
    wm.facts[WMNode.create("Inventory", owner="mary", item="apple", qty=0)] = 5
    
    print(f"--- Booting AGI Synchronous Loop (DLN GSM8K Inference) ---")
    print(f"Initial State: John has 10 apples, Mary has 0 apples.")
    print(f"Input: '{input_text}'\n")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        for cycle in range(1, max_cycles + 1):
            print(f"\n[Cycle {cycle}] Current State in WM:")
            for fact, ttl in wm.facts.items():
                if fact.concept in ["Inventory", "Belief"]:
                    print(f"  - {fact.concept}({', '.join(f'{k}={v}' for k, v in fact.to_dict().items())})")
            
            current_wm_snapshot = set(wm.facts.keys())
            f1 = executor.submit(system1_step, current_wm_snapshot, cycle, input_text)
            f2 = executor.submit(system2_step, current_wm_snapshot, cycle, None)
            f3 = executor.submit(system3_step, current_wm_snapshot, cycle, None)
            
            out1, out2, out3 = f1.result(), f2.result(), f3.result()
            
            wm.update(out1)
            wm.update(out2)
            wm.update(out3)
            wm.decay()
            time.sleep(0.5)

    print(f"\n--- Halt ---")

if __name__ == "__main__":
    gsm8k_problem = "John gave Mary 5 apples."
    run_cognitive_loop(gsm8k_problem)
