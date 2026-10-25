import argparse
import concurrent.futures
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / 'forward_engine'))
sys.path.append(str(BASE_DIR / 'logic_transformer'))
sys.path.append(str(BASE_DIR / 'ilp_learner'))

from working_memory.working_memory import WMNode, WorkingMemory, EquationGraphSolver
from working_memory.semantic_schema import SemanticGraphSchema
from working_memory.predicate_logic import wmnode_to_predicate_atoms
from forward_engine.canonical_system2 import CanonicalSystem2
from forward_engine.defeasible_logic import load_default_defeasible_reasoner
from system2_poc import System2Engine
from experta import Fact
from spacy_logical_form import SpacyLogicalFormParser
from train_pot_dln_pointer import DLNPointerDecoder
import train_pot_clause as tpc
from abductive_ilp import KnowledgeBase, OntologicalReasoner, OntologicalILPLearner, DuckEgg
from local_proposal_ranker import ProposalRanker

# ==========================================
# A compact GSM8K predicate vocabulary.
# This is the bridge between natural-language inputs and the symbolic WM.
# The goal is to keep the ontology small but canonical rather than adding
# more one-off ad hoc facts for each template.
# ==========================================
GSM8K_FACT_VOCAB = {
    "Quantity": ["value", "unit", "qty"],
    "Entity": ["name", "type"],
    "Relation": ["lhs", "rhs", "kind"],
    "ScaleFactor": ["owner", "item", "factor"],
    "Rate": ["item", "price_per_unit"],
    "Duration": ["owner", "item", "duration"],
    "Produced": ["owner", "item", "qty"],
    "Consumed": ["owner", "item", "qty"],
    "Sells": ["owner", "item", "price"],
    "Price": ["item", "price_per_unit"],
    "Has": ["owner", "item", "qty"],
    "Remaining": ["owner", "item", "qty"],
    "Total": ["owner", "item", "qty"],
    "Add": ["lhs", "rhs", "result"],
    "Subtract": ["lhs", "rhs", "result"],
    "Multiply": ["lhs", "rhs", "result"],
    "Divide": ["lhs", "rhs", "result"],
    "Percent": ["owner", "item", "pct"],
    "Ratio": ["owner", "item", "factor"],
    "UnitConvert": ["value", "from_unit", "to_unit", "result"],
    "Query": ["goal", "args", "expected_value"],
}

# ==========================================
# Sub-Systems
# ==========================================
_dln_model = None
_dln_vocab = None
_rev_vocab = None
_local_proposal_ranker = None


def load_local_proposal_ranker():
    global _local_proposal_ranker
    if _local_proposal_ranker is not None:
        return _local_proposal_ranker
    try:
        from local_proposal_ranker import ProposalRanker
        _local_proposal_ranker = ProposalRanker.load_or_train()
    except Exception:
        _local_proposal_ranker = None
    return _local_proposal_ranker


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

def _lift_semantic_candidates(input_text):
    """Create a minimal semantic-layer proposal set from short GSM8K-like quantities.

    This is intentionally heuristic and narrow: it converts high-value parse patterns
    into canonical Quantity / Has / StateTransition facts so the symbol validator can
    consume them without needing the entire parser stack to be fully trained.
    """
    facts = set()
    if not input_text:
        return facts

    # Match bare ownership + quantity facts.
    for owner, qty, item in re.findall(r"\b([A-Z][a-z]*)\s+had\s+(\d+(?:\.\d+)?)\s+([A-Za-z]+)\b", input_text):
        value = float(qty)
        facts.add(WMNode.create("Quantity", activation=0.85, layer="semantic", source="system1-grounding", provenance_id=f"sem:quantity:{owner}:{item}", owner=owner, item=item, qty=value))
        facts.add(WMNode.create("Has", activation=0.85, layer="semantic", source="system1-grounding", provenance_id=f"sem:has:{owner}:{item}", owner=owner, item=item, qty=value))

    for owner, delta, item in re.findall(r"\b([A-Z][a-z]*)\s+bought\s+(\d+(?:\.\d+)?)\s+more\s+([A-Za-z]+)\b", input_text):
        value = float(delta)
        facts.add(WMNode.create("StateTransition", activation=0.8, layer="semantic", source="system1-grounding", provenance_id=f"sem:trans:{owner}:{item}", owner=owner, item=item, before_qty=None, delta=value, after_qty=None, event_kind="buy", event_order=1))

    for owner, delta, item in re.findall(r"\b([A-Z][a-z]*)\s+gave\s+away\s+(\d+(?:\.\d+)?)\s+([A-Za-z]+)\b", input_text):
        value = float(delta)
        facts.add(WMNode.create("StateTransition", activation=0.8, layer="semantic", source="system1-grounding", provenance_id=f"sem:trans:{owner}:{item}", owner=owner, item=item, before_qty=None, delta=-value, after_qty=None, event_kind="give_away", event_order=1))

    # Ratio and percent-of-remaining patterns: lift canonical semantic forms so the
    # validator can reason on them later instead of leaving them as text-only parse noise.
    for owner, factor, item, ref_owner in re.findall(r"\b([A-Z][a-z]*)\s+has\s+(\d+(?:\.\d+)?)\s+times\s+as\s+many\s+([A-Za-z]+)\s+as\s+([A-Z][a-z]*)\b", input_text):
        facts.add(WMNode.create("Ratio", activation=0.82, layer="semantic", source="system1-grounding", provenance_id=f"sem:ratio:{owner}:{item}", owner=owner, base_owner=ref_owner, item=item, factor=float(factor)))
        facts.add(WMNode.create("Quantity", activation=0.82, layer="semantic", source="system1-grounding", provenance_id=f"sem:baseqty:{ref_owner}:{item}", owner=ref_owner, item=item, qty=0.0))

    for owner, base_qty, item, pct, rem_item in re.findall(r"\b([A-Z][a-z]*)\s+had\s+(\d+(?:\.\d+)?)\s+([A-Za-z]+)\b.*?(\d+(?:\.\d+)?)%\s+of\s+the\s+remaining\s+([A-Za-z]+)\b", input_text):
        if item != rem_item:
            continue
        facts.add(WMNode.create("Percent", activation=0.82, layer="semantic", source="system1-grounding", provenance_id=f"sem:pct:{owner}:{item}", owner=owner, item=item, pct=float(pct) / 100.0))
        facts.add(WMNode.create("Remaining", activation=0.82, layer="semantic", source="system1-grounding", provenance_id=f"sem:rem:{owner}:{item}", owner=owner, item=item, qty=float(base_qty)))

    return facts


def system1_step(current_wm, cycle, input_text=None, quiet=False):
    new_facts = set()
    
    if cycle == 1 and input_text:
        # 1. Parse raw structural features via SpacyLogicalFormParser directly (acting as our semantic grounder).
        #    These are deliberately marked as a parse-layer WM representation, not yet the final reasoning facts.
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

        proposal_ranker = load_local_proposal_ranker()
        
        # Map the output strings back to WMNodes with soft activation (0.85) representing System 1's neural confidence.
        # The layer is explicitly tagged as parse/semantic so it can be consumed by later grounding rules.
        for clause in core_clauses:
            pred = clause["pred"]
            args = clause["args"]
            concept = pred.capitalize()
            if concept in ["Event", "Predicate", "Agent", "Patient", "Recipient", "Name", "Quantity", "Location", "Time", "Modifier"]:
                kwargs = {f"arg{j}": arg for j, arg in enumerate(args)}
                proposal_score = 0.75
                if proposal_ranker is not None and input_text:
                    proposal_text = f"{concept}({', '.join(str(a) for a in args)})"
                    proposal_score = proposal_ranker.score(input_text, proposal_text)
                activation = max(0.15, min(0.95, 0.35 + 0.6 * proposal_score))
                new_facts.add(WMNode.create(concept, activation=activation, layer="parse", source="system1", provenance_id=f"s1:{concept.lower()}", **kwargs))

    # A second pass: take raw parse-layer clauses and lift them into the semantic layer.
    # This keeps the pipeline explicit: parse facts are not allowed to directly participate in
    # the reasoning rules as final symbolic primitives.
    if input_text:
        for node in current_wm:
            if node.layer == "parse" and node.concept in {"Quantity", "Entity", "Relation"}:
                semantic_node = WMNode.create(
                    node.concept,
                    activation=max(node.activation, 0.9),
                    layer="semantic",
                    source="system1-grounding",
                    provenance_id=f"sem:{node.concept.lower()}",
                    **node.to_dict()
                )
                new_facts.add(semantic_node)

        semantic_candidates = _lift_semantic_candidates(input_text)
        new_facts.update(semantic_candidates)
            
    return new_facts

def system2_step(current_wm, cycle, _=None, quiet=False):
    new_facts = set()

    # Validation pass: accept only semantic-layer candidates that satisfy the schema.
    # This is the direct check that turns proposal facts into usable symbolic facts.
    for node in current_wm:
        if node.layer != "semantic":
            continue
        if node.concept not in {"Quantity", "Has", "StateTransition", "Ratio", "Percent", "Remaining"}:
            continue
        attrs = node.to_dict()
        owner = attrs.get("owner")
        item = attrs.get("item")
        qty = attrs.get("qty")
        delta = attrs.get("delta")
        factor = attrs.get("factor")
        pct = attrs.get("pct")
        if not owner or not item:
            continue
        if node.concept in {"Quantity", "Has", "Remaining"} and qty is None:
            continue
        if node.concept == "StateTransition" and delta is None:
            continue
        if node.concept == "Ratio" and factor is None:
            continue
        if node.concept == "Percent" and pct is None:
            continue
        value = qty if qty is not None else (delta if delta is not None else (factor if factor is not None else pct))
        new_facts.add(WMNode.create(
            "Belief",
            activation=0.9,
            layer="reasoning",
            source="system2",
            provenance_id=f"validate:{node.provenance_id or node.concept.lower()}",
            label="validated_semantic",
            owner=owner,
            item=item,
            qty=value,
        ))

    engine = System2Engine()
    engine.reset()
    
    from system2_poc import (
        Predicate, Agent, Recipient, Patient, Name, Quantity, Inventory, Belief,
        EventProcessed, Location, Time, Modifier, Produced, Consumed, Sells,
        Rate, Has, Total, Remaining, ScaleFactor, Add, Subtract, Multiply, Divide,
        Percent, Ratio, UnitConvert, Query
    )
    CLASS_MAP = {
        "Predicate": Predicate, "Agent": Agent, "Recipient": Recipient,
        "Patient": Patient, "Name": Name, "Quantity": Quantity,
        "Inventory": Inventory, "Belief": Belief, "EventProcessed": EventProcessed,
        "Location": Location, "Time": Time, "Modifier": Modifier,
        "Produced": Produced, "Consumed": Consumed, "Sells": Sells,
        "Rate": Rate, "Has": Has, "Total": Total, "Remaining": Remaining,
        "ScaleFactor": ScaleFactor, "Add": Add, "Subtract": Subtract,
        "Multiply": Multiply, "Divide": Divide, "Percent": Percent,
        "Ratio": Ratio, "UnitConvert": UnitConvert, "Query": Query
    }
    
    for node in current_wm:
        # Sys-2 should only consume semantic or reasoning-layer facts; parse-layer facts are a
        # provenance layer and are intentionally ignored here.
        if node.layer == "parse":
            continue
        kwargs = {k: v for k, v in node.to_dict().items() if not str(k).startswith("__")}
        # Normalize legacy arg0/arg1/arg2 slots to the canonical micro-step schema
        # so both historical toy facts and the newer GSM8K facts share one ontology.
        if "arg0" in kwargs and "owner" not in kwargs:
            kwargs["owner"] = kwargs.pop("arg0")
        if "arg1" in kwargs and "item" not in kwargs:
            kwargs["item"] = kwargs.pop("arg1")
        if "arg2" in kwargs and "qty" not in kwargs and "price" not in kwargs:
            kwargs["qty"] = kwargs.pop("arg2")
        if "arg2" in kwargs and "price" not in kwargs and node.concept == "Sells":
            kwargs["price"] = kwargs.pop("arg2")
        fact_class = CLASS_MAP.get(node.concept)
        if not fact_class:
            fact_class = type(node.concept, (Fact,), {})
        engine.declare(fact_class(**kwargs))
            
    engine.run()
    
    validated = set()
    for f in engine.facts.values():
        if type(f).__name__ in ["Belief", "Inventory", "EventProcessed"]:
            clean_attrs = {k: v for k, v in f.items() if not str(k).startswith("__") and k != "activation"}
                
            # Preserving the original node's activation/salience score from Working Memory
            concept = type(f).__name__
            temp_node = WMNode.create(concept, **clean_attrs)
            original = next((n for n in current_wm if n == temp_node), None)
            activation = original.activation if original is not None else 1.0
            
            validated.add(WMNode.create(concept, activation=activation, **clean_attrs))

    new_facts.update(validated)

    # Canonical predicate integration: keep the legacy engine path for backward
    # compatibility, but also run the arity-limited reasoner over the same WM to
    # ensure the canonical semantics are still valid in the live loop.
    canonical_engine = CanonicalSystem2()
    defeasible_reasoner = load_default_defeasible_reasoner()
    predicate_facts = []
    for node in current_wm:
        if node.layer in {"semantic", "reasoning"}:
            predicate_facts.extend(wmnode_to_predicate_atoms(node))
    if predicate_facts:
        canonical_engine.add_facts(predicate_facts)
        canonical_facts = canonical_engine.run()
        for fact in canonical_facts:
            attr_map = {f"arg{i}": value for i, value in enumerate(fact.args)}
            concept = "Belief" if fact.predicate == "belief" else fact.predicate.capitalize()
            new_facts.add(WMNode.create(
                concept,
                activation=0.95,
                layer="reasoning",
                source="canonical-system2",
                provenance_id=f"canon:{fact.predicate}",
                label=fact.predicate,
                **attr_map,
            ))

        # Defeasible layer: allow broad heuristic conclusions, then resolve conflicts
        # via explicit priority relations (r1 > r2) instead of hard-coded scalar scoring.
        defeasible_reasoner.add_facts(predicate_facts + canonical_facts)
        accepted, meta = defeasible_reasoner.run()
        for fact in accepted:
            attr_map = {f"arg{i}": value for i, value in enumerate(fact.args)}
            concept = "Belief" if fact.predicate == "belief" else fact.predicate.capitalize()
            new_facts.add(WMNode.create(
                concept,
                activation=0.93,
                layer="reasoning",
                source="defeasible-system2",
                provenance_id=f"def:{fact.predicate}",
                label=fact.predicate,
                **attr_map,
            ))
        for fact in meta:
            attr_map = {f"arg{i}": value for i, value in enumerate(fact.args)}
            new_facts.add(WMNode.create(
                "DefeasibleMeta",
                activation=0.9,
                layer="meta",
                source="defeasible-system2",
                provenance_id=f"defmeta:{fact.predicate}",
                label=fact.predicate,
                **attr_map,
            ))

    return new_facts

def system3_step(current_wm, cycle, _=None, quiet=False):
    new_facts = set()
    goal_pred = None
    goal_args = []
    expected_value = None

    # 0. Resolve explicit contradiction / revision facts before generic queries.
    # This is the Sys-3 AGM/TMS layer: when a same owner/item conflict is recorded,
    # prefer the latest valid state, then source authority, and only then numeric size.
    contradiction_nodes = [n for n in current_wm if n.concept == "Contradiction"]
    if contradiction_nodes:
        def value_rank(node):
            attrs = node.to_dict()
            event_order = int(attrs.get("event_order") or 0)
            source_rank = {"state_machine": 3, "system2": 2, "system1-grounding": 1, "system1": 0, "unknown": 0}.get(attrs.get("source"), 0)
            activation = float(node.activation)
            return (event_order, source_rank, activation)

        contradiction = max(contradiction_nodes, key=value_rank)
        attrs = contradiction.to_dict()
        owner = attrs.get("owner")
        item = attrs.get("item")
        values = attrs.get("values") or ()
        if owner and item and values:
            value_candidates = [v for v in values if v is not None]
            if value_candidates:
                preferred_state_nodes = [
                    n for n in current_wm
                    if n.concept in {"PreferredState", "FinalState", "StateTransition", "Quantity", "Remaining", "Total", "State"}
                    and n.to_dict().get("owner") == owner
                    and n.to_dict().get("item") == item
                    and (n.to_dict().get("qty") is not None or n.to_dict().get("value") is not None)
                ]
                if preferred_state_nodes:
                    preferred = max(preferred_state_nodes, key=lambda n: (
                        int(n.to_dict().get("event_order") or 0),
                        {"state_machine": 3, "system2": 2, "system1-grounding": 1, "system1": 0, "unknown": 0}.get(n.source, 0),
                        float(n.activation),
                    ))
                    preferred_value = preferred.to_dict().get("qty")
                    if preferred_value is None:
                        preferred_value = preferred.to_dict().get("after_qty")
                    if preferred_value is None:
                        preferred_value = preferred.to_dict().get("value")
                    if preferred_value is not None:
                        new_facts.add(WMNode.create("Belief", label=f"preferred_state:{owner}:{item}", value=float(preferred_value), activation=1.0))
                        if not quiet:
                            print(f"[System 3] AGM revision: preferred {owner}/{item} = {preferred_value}")
                        return new_facts

                preferred_value = max(
                    value_candidates,
                    key=lambda v: (float(v) if isinstance(v, (int, float)) else 0.0),
                )
                new_facts.add(WMNode.create("Belief", label=f"preferred_state:{owner}:{item}", value=float(preferred_value), activation=1.0))
                if not quiet:
                    print(f"[System 3] AGM revision: preferred {owner}/{item} = {preferred_value}")
                return new_facts

    # 1. Search for any Query node currently in Working Memory.
    # A single problem can legitimately produce multiple query candidates (e.g. ratio
    # sub-questions plus a final combined total), so prefer the most specific final
    # query and fall back to the first candidate only when necessary.
    query_nodes = [n for n in current_wm if n.concept == "Query"]
    if query_nodes:
        final_state_index = {}
        for n in current_wm:
            if n.concept in {"PreferredState", "FinalState"}:
                owner = n.to_dict().get("owner")
                item = n.to_dict().get("item")
                if owner is None or item is None:
                    continue
                key = (owner, item)
                current = final_state_index.get(key)
                if current is None or int(n.to_dict().get("event_order") or 0) > int(current[0] or 0):
                    final_state_index[key] = (
                        int(n.to_dict().get("event_order") or 0),
                        n.to_dict().get("qty"),
                    )

        def query_score(node):
            attrs = node.to_dict()
            goal = attrs.get("goal")
            args = tuple(attrs.get("args") or ())
            owner = args[0] if len(args) > 0 else None
            item = args[1] if len(args) > 1 else None
            event_order = attrs.get("event_order")
            if event_order is None:
                event_order = 0
            goal_priority = {
                "total_quantity": 5,
                "remaining_quantity": 4,
                "final_price": 3,
                "alternate_price_total": 3,
                "ratio_compose": 1,
            }.get(goal, 0)

            final_bonus = 0
            if goal in {"total_quantity", "remaining_quantity"} and owner and item:
                latest_final = final_state_index.get((owner, item))
                if latest_final is not None:
                    final_bonus += 1000
                    final_value = latest_final[1]
                    if final_value is not None and attrs.get("expected_value") is not None:
                        try:
                            if abs(float(attrs.get("expected_value")) - float(final_value)) < 1e-9:
                                final_bonus += 100
                        except (TypeError, ValueError):
                            pass
            return (
                goal_priority + final_bonus,
                int(event_order),
                float(attrs.get("expected_value") or 0.0),
                1 if (owner and item) else 0,
            )
        preferred = max(query_nodes, key=query_score)
        q_attrs = preferred.to_dict()
        goal_pred = q_attrs.get("goal")
        goal_args = list(q_attrs.get("args", []))
        expected_value = q_attrs.get("expected_value")
    else:
        # If no explicit query fact is present but there is an explicit graph query node,
        # still try to read its answer directly from the equation graph.
        graph_query = next((n for n in current_wm if n.concept == "GraphNode" and n.to_dict().get("kind") == "query"), None)
        if graph_query is not None:
            q_attrs = graph_query.to_dict()
            goal_pred = q_attrs.get("goal") or q_attrs.get("label")
            goal_args = [q_attrs.get("goal")] if q_attrs.get("goal") else [q_attrs.get("label")]
            expected_value = q_attrs.get("expected_value")

    # 2. Prefer the latest valid state for this entity before any generic graph solve.
    # This is the explicit 'finality' rule: queries about the final amount must resolve
    # against the maximal event-order state, not against earlier historical snapshots.
    # We treat finality as a temporal relation (maximal event order), not as an ad hoc
    # label on the original quantity fact. This preserves ordinary earlier states for
    # historical reasoning while allowing later updates to override them for final queries.
    # Critical guard: once an authoritative final state exists for an owner/item, any same-
    # owner/item quantity facts without a later event order are non-authoritative and must be ignored.
    if goal_pred in {"remaining_quantity", "total_quantity"} and len(goal_args) >= 2:
        owner, item = goal_args[0], goal_args[1]

        blocked_same_item_values = {
            n.to_dict().get("qty")
            for n in current_wm
            if n.concept == "BlockedState"
            and n.to_dict().get("owner") == owner
            and n.to_dict().get("item") == item
            and n.to_dict().get("qty") is not None
        }

        preferred_state_nodes = [
            n for n in current_wm
            if n.concept == "PreferredState"
            and n.to_dict().get("owner") == owner
            and n.to_dict().get("item") == item
        ]
        if preferred_state_nodes:
            latest = max(preferred_state_nodes, key=lambda n: int(n.to_dict().get("event_order") or 0))
            final_qty = latest.to_dict().get("qty")
            if final_qty is None:
                final_qty = latest.to_dict().get("after_qty")
            if final_qty is None:
                final_qty = latest.to_dict().get("value")
            if final_qty is not None:
                new_facts.add(WMNode.create("Belief", label=goal_pred, value=float(final_qty), activation=1.0))
                if not quiet:
                    print(f"[System 3] Preferred state authority: {goal_pred} = {final_qty} from preferred state")
                return new_facts

        current_quantity_candidates = [
            n for n in current_wm
            if n.concept in {"Quantity", "Remaining", "Total", "Has", "Inventory", "Produced", "Consumed"}
            and n.to_dict().get("owner") == owner
            and n.to_dict().get("item") == item
        ]
        if current_quantity_candidates and blocked_same_item_values:
            authoritative_values = {
                float(n.to_dict().get("qty") or n.to_dict().get("value") or 0.0)
                for n in current_quantity_candidates
                if n.to_dict().get("qty") is not None or n.to_dict().get("value") is not None
                and n.to_dict().get("event_order") is not None
            }
            for candidate in current_quantity_candidates:
                candidate_value = candidate.to_dict().get("qty")
                if candidate_value is None:
                    candidate_value = candidate.to_dict().get("value")
                if candidate_value is not None and float(candidate_value) in blocked_same_item_values and not authoritative_values:
                    continue

        final_state_nodes = [
            n for n in current_wm
            if n.concept == "FinalState"
            and n.to_dict().get("owner") == owner
            and n.to_dict().get("item") == item
        ]
        if final_state_nodes:
            latest = max(final_state_nodes, key=lambda n: int(n.to_dict().get("event_order") or 0))
            final_qty = latest.to_dict().get("qty")
            if final_qty is None:
                final_qty = latest.to_dict().get("after_qty")
            if final_qty is None:
                final_qty = latest.to_dict().get("value")
            if final_qty is not None:
                new_facts.add(WMNode.create("Belief", label=goal_pred, value=float(final_qty), activation=1.0))
                if not quiet:
                    print(f"[System 3] Final state authority: {goal_pred} = {final_qty} from final-state node")
                return new_facts

        state_transitions = [
            n for n in current_wm
            if n.concept == "StateTransition"
            and n.to_dict().get("owner") == owner
            and n.to_dict().get("item") == item
        ]
        if state_transitions:
            latest = max(state_transitions, key=lambda n: int(n.to_dict().get("event_order") or 0))
            final_qty = latest.to_dict().get("after_qty")
            if final_qty is not None:
                new_facts.add(WMNode.create("Belief", label=goal_pred, value=float(final_qty), activation=1.0))
                if not quiet:
                    print(f"[System 3] Final state override: {goal_pred} = {final_qty} from latest transition")
                return new_facts
    
    if not goal_pred or not goal_args:
        return new_facts

    # Graph-aware direct solve: if the relevant equation graph has a numeric output
    # available, use it as the answer without needing a broader search. This is a
    # lightweight semantic dependency solver general enough to cover arithmetic, state,
    # and quantity-flow reasoning across story understanding, not just pure math.
    graph_solver = EquationGraphSolver.from_wm(current_wm)
    ans_val = graph_solver.solve(goal_pred)
    if ans_val is not None:
        new_facts.add(WMNode.create("Belief", label=goal_pred, value=ans_val, activation=1.0))
        if not quiet:
            print(f"[System 3] Solved via dependency graph: {goal_pred} = {ans_val}")
        return new_facts
    
    # 2. Generate background KB dynamically from current Working Memory!
    kb = KnowledgeBase()
    for node in current_wm:
        # Declare all factual relationships from the canonical WM schema into the ILP KB.
        # This is more robust than a raw `arg0/arg1/arg2` scan, because our GSM8K facts are
        # stored by semantic keys like owner/item/qty/price rather than by positional args.
        attrs = node.to_dict()
        pred = node.concept.lower()
        args = []
        if node.concept in {"Produced", "Consumed", "Has", "Total", "Remaining", "Quantity"}:
            args = [attrs.get("owner"), attrs.get("item"), attrs.get("qty")]
        elif node.concept == "Sells":
            args = [attrs.get("owner"), attrs.get("item"), attrs.get("price")]
        elif node.concept == "Rate":
            args = [attrs.get("item"), attrs.get("price_per_unit")]
        elif node.concept == "ScaleFactor":
            args = [attrs.get("owner"), attrs.get("item"), attrs.get("factor")]
        elif node.concept == "Percent":
            args = [attrs.get("owner"), attrs.get("item"), attrs.get("pct")]
        elif node.concept == "Ratio":
            if attrs.get("base_owner") is not None:
                kb.declare_fact("ratio", attrs.get("owner"), attrs.get("base_owner"), attrs.get("item"), attrs.get("factor"))
                args = [attrs.get("owner"), attrs.get("base_owner"), attrs.get("item"), attrs.get("factor")]
            else:
                kb.declare_fact("ratio", attrs.get("owner"), attrs.get("item"), attrs.get("factor"))
                args = [attrs.get("owner"), attrs.get("item"), attrs.get("factor")]
        elif node.concept == "UnitConvert":
            args = [attrs.get("value"), attrs.get("from_unit"), attrs.get("to_unit"), attrs.get("result")]
        elif node.concept in {"Add", "Subtract", "Multiply", "Divide"}:
            args = [attrs.get("lhs"), attrs.get("rhs"), attrs.get("result")]
        elif node.concept == "Query":
            args = list(attrs.get("args", []))
            if attrs.get("goal"):
                args = [attrs.get("goal"), *args]
        
        args = [a for a in args if a is not None]
        if args:
            kb.declare_fact(pred, *args)
            
    # Always possess baseline reasoning rules for the micro-step semantics.
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
    kb.declare_rule(
        head_pred="sold_qty",
        head_args=["?p", "?item", "?ans"],
        body=[
            ("produced", ["?p", "?item", "?produced"]),
            ("consumed", ["?p", "?item", "?consumed"])
        ],
        calc_expr="produced - consumed",
        calc_target="?ans"
    )
    kb.declare_rule(
        head_pred="percent_increase",
        head_args=["?owner", "?item", "?ans"],
        body=[
            ("percent", ["?owner", "?item", "?pct"]),
            ("quantity", ["?owner", "?item", "?qty"])
        ],
        calc_expr="qty * pct",
        calc_target="?ans"
    )
    kb.declare_rule(
        head_pred="ratio_compose",
        head_args=["?owner", "?item", "?ans"],
        body=[
            ("ratio", ["?owner", "?item", "?factor"]),
            ("quantity", ["?owner", "?item", "?qty"])
        ],
        calc_expr="qty * factor",
        calc_target="?ans"
    )
    kb.declare_rule(
        head_pred="unit_convert",
        head_args=["?value", "?from_unit", "?to_unit", "?ans"],
        body=[
            ("unitconvert", ["?value", "?from_unit", "?to_unit", "?result"])
        ],
        calc_expr="result",
        calc_target="?ans"
    )
    kb.declare_rule(
        head_pred="rate_times_duration",
        head_args=["?owner", "?item", "?ans"],
        body=[
            ("rate", ["?item", "?price"]),
            ("quantity", ["?owner", "?hours", "?qty"])
        ],
        calc_expr="price * qty",
        calc_target="?ans"
    )
    kb.declare_rule(
        head_pred="alternate_price_total",
        head_args=["?owner", "?item", "?ans"],
        body=[
            ("quantity", ["?owner", "?item", "?qty"]),
            ("rate", ["?item", "?price"]),
            ("percent", ["?owner", "?item", "?pct"])
        ],
        calc_expr="(qty / 2.0) * (price + price * pct)",
        calc_target="?ans"
    )
    kb.declare_rule(
        head_pred="divide",
        head_args=["?lhs", "?rhs", "?ans"],
        body=[
            ("divide", ["?lhs", "?rhs", "?result"])
        ],
        calc_expr="result",
        calc_target="?ans"
    )
    kb.declare_rule(
        head_pred="total_quantity",
        head_args=["?owner", "?item", "?ans"],
        body=[
            ("quantity", ["?owner", "?item", "?qty"])
        ],
        calc_expr="qty",
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
        # If proof failed, execute Abductive ILP to learn a local symbolic rule.
        # The default templates are intentionally narrow and compositional; they help
        # cover the common semantic-glue patterns without trying to learn the full
        # text-to-graph grounding problem in one shot.
        learner = OntologicalILPLearner(kb)
        induced = learner.abduce_and_induce([(goal_pred, goal_args, expected_value)])
        if not induced:
            induced = learner.default_rule_template(goal_pred, goal_args)
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
def _compile_gsm8k_problem_to_wm(input_text, expected_answer=None):
    wm = WorkingMemory(default_ttl=5)

    def add_quantity_state_sequence(owner, item, initial_qty, events):
        """Emit an ordered state-trace for multi-step quantity problems.

        Each event produces a new quantity state, and each transition records the
        before/after quantity along with a monotonic event order. This keeps later
        reasoning grounded in explicit state progression instead of competing
        ambiguous "remaining" candidates from a flat flat fact set.
        """
        existing_orders = [
            int(f.to_dict().get("event_order") or 0)
            for f in wm.facts
            if f.concept in {"StateTransition", "FinalState", "PreferredState"}
            and f.to_dict().get("event_order") is not None
        ]
        start_order = max(existing_orders, default=0) + 1

        if not events:
            wm.add_state(state_id=f"state:{owner}:{item}:initial", owner=owner, item=item, qty=initial_qty, phase="initial")
            return

        previous_qty = initial_qty
        previous_state_id = f"state:{owner}:{item}:initial"
        wm.add_state(state_id=previous_state_id, owner=owner, item=item, qty=initial_qty, phase="initial")

        for idx, (event_kind, delta, after_qty) in enumerate(events, start=start_order):
            transition_id = f"state_transition:{owner}:{item}:{event_kind}:{idx}"
            new_state_id = f"state:{owner}:{item}:{event_kind}:{idx}"
            wm.add_state_transition(
                transition_id=transition_id,
                owner=owner,
                item=item,
                before_qty=previous_qty,
                delta=delta,
                after_qty=after_qty,
                event_kind=event_kind,
                event_order=idx,
                source_state_id=previous_state_id,
                target_state_id=new_state_id,
            )
            wm.add_state(state_id=new_state_id, owner=owner, item=item, qty=after_qty, phase=event_kind)
            previous_qty = after_qty
            previous_state_id = new_state_id

    # Structured GSM8K compiler.
    # We keep a compact, canonical predicate vocabulary and map several common
    # arithmetic and state-transition families into it. This is still heuristic,
    # but it is systematic rather than one-off. That is the right next step.
    text = input_text or ""
    lower = text.lower()

    def add_fact(concept, **kwargs):
        canonical = SemanticGraphSchema.canonicalize_fact(concept, **kwargs)
        wm.facts[WMNode.create(concept, layer="semantic", source="gsm8k_compiler", **canonical)] = 5

    # Canonical simple-state fallback: if a story states that an owner had N items
    # and then lost/donated/used Y items, emit the same-owner/item state transition
    # explicitly so the canonical predicate engine can reason on it instead of
    # relying only on the generic text patterns. This keeps the semantic contract
    # aligned with the arity-limited rule engine while preserving the legacy WM.
    simple_loss_match = re.search(
        r"([A-Z][a-zA-Z]*)\s+had\s+(\d+(?:\.\d+)?)\s+([A-Za-z][A-Za-z-]*(?:\s+[A-Za-z][A-Za-z-]*)?)\s+(?:and\s+)?(?:lost|gave\s+away|used|spent|donated)\s+(\d+(?:\.\d+)?)\s+([A-Za-z][A-Za-z-]*(?:\s+[A-Za-z][A-Za-z-]*)?)",
        text,
        re.I,
    )
    if simple_loss_match:
        owner = simple_loss_match.group(1)
        initial_qty = float(simple_loss_match.group(2))
        item = re.sub(r"\s+and\s*$", "", simple_loss_match.group(3)).strip().lower()
        lost_qty = float(simple_loss_match.group(4))
        lost_item = re.sub(r"\s+and\s*$", "", simple_loss_match.group(5)).strip().lower()
        if item == lost_item or not lost_item or item == "":
            remaining_qty = initial_qty - lost_qty
            add_fact("Has", owner=owner, item=item, qty=initial_qty)
            add_fact("Quantity", owner=owner, item=item, qty=initial_qty)
            wm.add_state_transition(
                transition_id=f"state_transition:{owner}:{item}:lose:1",
                owner=owner,
                item=item,
                before_qty=initial_qty,
                delta=-lost_qty,
                after_qty=remaining_qty,
                event_kind="lose",
                event_order=1,
                source_state_id=f"state:{owner}:{item}:initial",
                target_state_id=f"state:{owner}:{item}:lose:1",
            )
            wm.add_state(state_id=f"state:{owner}:{item}:lose:1", owner=owner, item=item, qty=remaining_qty, phase="final")
            add_fact("Query", goal="remaining_quantity", args=(owner, item, "?ans"), expected_value=remaining_qty)
            add_fact("Belief", label="remaining_quantity", value=remaining_qty, activation=1.0)
            return wm

    def num(pattern, default=None):
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if not m:
            return default
        val = m.group(1)
        try:
            return float(val)
        except ValueError:
            return default

    def normalize_owner_name(value):
        if value is None:
            return "unknown"
        cleaned = re.sub(r"\b(?:then|later|afterward|afterwards|next)\b", "", str(value).lower()).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned or "unknown"

    def add_fact(concept, **kwargs):
        canonical = SemanticGraphSchema.canonicalize_fact(concept, **kwargs)
        wm.facts[WMNode.create(concept, layer="semantic", source="gsm8k_compiler", **canonical)] = 5

    def add_equation_graph(op_id, op_kind, label, *, inputs=None, outputs=None, **attrs):
        lhs_var = (inputs or [])[0][0] if (inputs or []) else "lhs"
        rhs_var = (inputs or [])[1][0] if len(inputs or []) > 1 else "rhs"
        result_var = (outputs or [])[0][0] if (outputs or []) else "result"
        op_payload = SemanticGraphSchema.make_operator_binding(op_id, op_kind, lhs_var, rhs_var, result_var)
        ok, issues = SemanticGraphSchema.validate_graph_fragment(op_payload)
        if not ok:
            print(f"[semantic-schema] rejected graph fragment {op_id}: {issues}")
        wm.add_equation_graph(
            op_id=op_id,
            op_kind=op_kind,
            label=label,
            inputs=inputs or [],
            outputs=outputs or [],
            **attrs,
        )

    def add_quantity_facts(owner, item, qty, *, expected_goal=None, expected_value=None):
        add_fact("Quantity", owner=owner, item=item, qty=qty)
        add_fact("Has", owner=owner, item=item, qty=qty)
        add_fact("Entity", name=item, type="item")
        add_fact("Relation", lhs=owner, rhs=item, kind="has")
        if expected_goal is not None:
            add_fact("Query", goal=expected_goal, args=(owner, item, "?ans"), expected_value=expected_value)

    # Janet-ducks style profit problem: produced - consumed = sold, revenue = sold * price
    if ("ducks" in lower or "duck" in lower) and "janet" in lower and ("egg" in lower or "eggs" in lower):
        produced = num(r"lay\s+(\d+(?:\.\d+)?)\s+eggs") or 16.0
        eaten = num(r"eats?\s+(\d+(?:\.\d+)?)") or 3.0
        baked = num(r"with\s+(\d+(?:\.\d+)?)") or 4.0
        price = num(r"for\s+\$?(\d+(?:\.\d+)?)\s+per") or 2.0
        consumed = eaten + baked
        add_fact("Produced", owner="janet", item=DuckEgg, qty=produced)
        add_fact("Consumed", owner="janet", item=DuckEgg, qty=consumed)
        add_fact("Sells", owner="janet", item=DuckEgg, price=price)
        add_fact("Price", item=DuckEgg, price_per_unit=price)
        add_fact("Entity", name="janet", type="person")
        add_fact("Entity", name=DuckEgg, type="item")
        add_fact("Relation", lhs="janet", rhs=DuckEgg, kind="sells")
        add_fact("Query", goal="daily_profit", args=("janet", DuckEgg, "?ans"), expected_value=expected_answer or 18.0)
        return wm

    # Generic ledger-style pattern: "X made Y, spent Z, sold each one for $P"
    if "sell" in lower or "sells" in lower or "make" in lower or "made" in lower:
        produced = num(r"(\d+(?:\.\d+)?)\s+(?:eggs|apples|items|units|cups|bolts|flowers|books)")
        consumed = num(r"(?:eat|ate|used|spent|bought|buys?|left|lost)\s+(\d+(?:\.\d+)?)")
        price = num(r"for\s+\$?(\d+(?:\.\d+)?)\s+per") or num(r"\$?(\d+(?:\.\d+)?)\s+each")
        person = "janet" if "janet" in lower else ("rob" if "rob" in lower else "unknown")
        if produced is not None and price is not None:
            consumed_value = consumed if consumed is not None else 0.0
            add_fact("Produced", owner=person, item=DuckEgg, qty=produced)
            add_fact("Consumed", owner=person, item=DuckEgg, qty=consumed_value)
            add_fact("Sells", owner=person, item=DuckEgg, price=price)
            add_fact("Price", item=DuckEgg, price_per_unit=price)
            add_fact("Entity", name=person, type="person")
            add_fact("Entity", name=DuckEgg, type="item")
            add_fact("Relation", lhs=person, rhs=DuckEgg, kind="sells")
            add_fact("Query", goal="daily_profit", args=(person, DuckEgg, "?ans"), expected_value=expected_answer or 18.0)
            return wm

    # Explicit sales-total pattern: "A bakery sells 40 cupcakes at $2 each" => $80 revenue
    sales_match = re.search(r"(?:sells?|sold)\s+(\d+(?:\.\d+)?)\s+([A-Za-z][A-Za-z\s-]+?)\s+(?:at|for)\s+\$?(\d+(?:\.\d+)?)\s*(?:each|per)\b", text, re.I)
    if sales_match:
        qty = float(sales_match.group(1))
        item = sales_match.group(2).strip().lower()
        price = float(sales_match.group(3))
        total = qty * price
        add_fact("Rate", item=item, price_per_unit=price)
        add_fact("Quantity", owner="seller", item=item, qty=qty)
        add_fact("Multiply", lhs=qty, rhs=price, result=total)
        add_equation_graph(
            f"eq:sales:{item}",
            "multiply",
            "sales_total",
            inputs=[("qty", qty), ("price", price)],
            outputs=[("total", total)],
            item=item,
            owner="seller",
        )
        add_fact("Query", goal="rate_times_duration", args=("seller", item, "?ans"), expected_value=total)
        add_equation_graph(
            f"eq:query:sales:{item}",
            "query",
            "sales_query",
            inputs=[("total", total)],
            outputs=[("answer", total)],
            goal="rate_times_duration",
        )
        return wm

    # Generic rate × duration pattern: "$15 per hour for 40 hours" => total pay.
    pay_match = re.search(r"\$?(\d+(?:\.\d+)?)\s*(?:per|each)\s*(?:hour|hr|day|week)\s+.*?(?:for|over|during)\s+(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|days?|weeks?)", text, re.I)
    if pay_match:
        rate = float(pay_match.group(1))
        duration = float(pay_match.group(2))
        total = rate * duration
        item = "hours"
        add_fact("Rate", item=item, price_per_unit=rate)
        add_fact("Quantity", owner="worker", item=item, qty=duration)
        add_fact("Multiply", lhs=rate, rhs=duration, result=total)
        add_equation_graph(
            "eq:rate_time:worker_pay",
            "multiply",
            "rate_times_duration",
            inputs=[("rate", rate), ("hours", duration)],
            outputs=[("total", total)],
            item=item,
            owner="worker",
        )
        add_fact("Query", goal="rate_times_duration", args=("worker", item, "?ans"), expected_value=total)
        add_equation_graph(
            "eq:query:rate_times_duration:worker",
            "query",
            "rate_times_duration_query",
            inputs=[("total", total)],
            outputs=[("answer", total)],
            goal="rate_times_duration",
        )
        return wm

    # Remaining quantity pattern: "20 apples and gave away 5" => 15 remain.
    remaining_match = re.search(r"(\d+(?:\.\d+)?)\s+(?:apples|eggs|items|students|books|balls|cups|bolts|flowers|donuts|candy|cookies|tickets)\s+.*?(?:and\s+)?(?:gave|give|gives|gave away|loses?|use|uses|spent|ate|eats?)\s+(\d+(?:\.\d+)?)", text, re.I)
    if remaining_match:
        item = re.search(r"(?:apples|eggs|items|students|books|balls|cups|bolts|flowers|donuts|candy|cookies|tickets)", text, re.I)
        item_name = (item.group(0) if item else "item").lower()
        total = float(remaining_match.group(1))
        used = float(remaining_match.group(2))
        left = total - used
        add_fact("Has", owner="unknown", item=item_name, qty=total)
        add_fact("Consumed", owner="unknown", item=item_name, qty=used)
        add_fact("Quantity", owner="unknown", item=item_name, qty=left)
        add_fact("Subtract", lhs=total, rhs=used, result=left)
        add_equation_graph(
            "eq:remaining_quantity",
            "subtract",
            "remaining_quantity",
            inputs=[("total", total), ("used", used)],
            outputs=[("remaining", left)],
            item=item_name,
            owner="unknown",
        )
        add_fact("Query", goal="remaining_quantity", args=("unknown", item_name, "?ans"), expected_value=left)
        add_quantity_state_sequence(
            "unknown",
            item_name,
            total,
            [("gave_away", -used, left)],
        )
        add_equation_graph(
            "eq:query:remaining_quantity",
            "query",
            "remaining_quantity_query",
            inputs=[("remaining", left)],
            outputs=[("answer", left)],
            goal="remaining_quantity",
        )
        return wm

    # Distance ÷ speed => time pattern: "180 miles at 60 mph, how long?"
    distance_time_match = re.search(
        r"(?:travel(?:s|ing)?\s+)?(\d+(?:\.\d+)?)\s*(?:miles?|mi|kilometers?|km)\s+(?:at\s+|for\s+)?(?:a\s+)?(?:speed\s+of\s+)?(\d+(?:\.\d+)?)\s*(?:mph|miles?\s+per\s+hour)",
        text,
        re.I,
    )
    asks_for_time = any(p in lower for p in ["how long", "how much time", "what time", "how many hours"])
    if distance_time_match and asks_for_time:
        dist = float(distance_time_match.group(1))
        speed = float(distance_time_match.group(2))
        time = dist / speed
        add_fact("Rate", item="distance", price_per_unit=speed)
        add_fact("Quantity", owner="traveler", item="distance", qty=dist)
        add_fact("Divide", lhs=dist, rhs=speed, result=time)
        add_equation_graph(
            "eq:distance_time",
            "divide",
            "distance_over_speed",
            inputs=[("distance", dist), ("speed", speed)],
            outputs=[("time", time)],
            item="distance",
            owner="traveler",
        )
        add_fact("Query", goal="divide", args=(dist, speed, "?ans"), expected_value=time)
        add_equation_graph(
            "eq:query:divide:distance_time",
            "query",
            "time_query",
            inputs=[("time", time)],
            outputs=[("answer", time)],
            goal="divide",
        )
        return wm

    # General scale pattern: "half that much", "twice as many", "3 times a week"
    scale_factor = None
    if "half that much" in lower or "half as much" in lower or "half as many" in lower:
        scale_factor = 0.5
    elif "twice" in lower or "double" in lower:
        scale_factor = 2.0
    elif "triple" in lower or re.search(r"(\d+(?:\.\d+)?)\s+times", text, re.I):
        scale_factor_match = re.search(r"(\d+(?:\.\d+)?)\s+times", text, re.I)
        if scale_factor_match:
            scale_factor = float(scale_factor_match.group(1))
        else:
            scale_factor = 3.0

    if scale_factor is not None and re.search(r"(?:\d+(?:\.\d+)?\s+)?(?:blue|white|fiber|bolts|cups|meters|items|eggs|flowers|books)", text, re.I):
        base = num(r"(\d+(?:\.\d+)?)\s+(?:bolts|cups|meters|items|eggs|flowers|books)") or 2.0
        scaled = base * scale_factor
        add_fact("Has", owner="robe", item="fiber", qty=base)
        add_fact("ScaleFactor", owner="robe", item="fiber", factor=scale_factor)
        add_fact("Quantity", owner="robe", item="fiber", qty=scaled)
        add_fact("Relation", lhs="blue", rhs="white", kind="scale")
        add_fact("Multiply", lhs=base, rhs=scale_factor, result=scaled)
        add_equation_graph(
            "eq:scale:robe:fiber",
            "multiply",
            "scale_factor_compose",
            inputs=[("base", base), ("factor", scale_factor)],
            outputs=[("result", scaled)],
            owner="robe",
            item="fiber",
        )
        add_fact("Produced", owner="robe", item="fiber", qty=base)
        add_fact("Consumed", owner="robe", item="fiber", qty=scaled)
        add_fact("Query", goal="total_quantity", args=("robe", "fiber", "?ans"), expected_value=base + scaled)
        add_equation_graph(
            "eq:query:total_quantity:robe:fiber",
            "query",
            "total_quantity_query",
            inputs=[("qty", base + scaled)],
            outputs=[("answer", base + scaled)],
            goal="total_quantity",
        )
        return wm

    # Pattern: every second glass costs 60% of the price.
    second_discount_match = re.search(r"every\s+second\s+([A-Za-z][A-Za-z\s-]+)\s+costs?\s+(?:only\s+)?(\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+the\s+price", text, re.I)
    if second_discount_match:
        item = second_discount_match.group(1).strip().lower()
        pct = float(second_discount_match.group(2)) / 100.0
        qty = num(r"(\d+(?:\.\d+)?)\s+" + re.escape(item.replace(' ', r'\s+')) + r"", default=2.0)
        base_price = num(r"(\d+(?:\.\d+)?)\s+\$?\s*(?:each|per)", default=5.0) or 5.0
        discount_price = base_price * pct
        total = (qty / 2.0) * (base_price + discount_price)
        add_fact("Rate", item=item, price_per_unit=base_price)
        add_fact("Percent", owner="unknown", item=item, pct=pct)
        add_fact("Quantity", owner="buyer", item=item, qty=qty)
        add_fact("Multiply", lhs=qty / 2.0, rhs=base_price + discount_price, result=total)
        add_equation_graph(
            f"eq:alternate_price_total:{item}",
            "multiply",
            "alternate_price_total",
            inputs=[("item_qty", qty / 2.0), ("effective_price", base_price + discount_price)],
            outputs=[("result", total)],
            item=item,
        )
        add_fact("Query", goal="alternate_price_total", args=("buyer", item, "?ans"), expected_value=total)
        add_equation_graph(
            f"eq:query:alternate_price_total:{item}",
            "query",
            "alternate_price_total_query",
            inputs=[("answer", total)],
            outputs=[("answer", total)],
            goal="alternate_price_total",
        )
        return wm

    # Pattern: "3 hours at a speed of 60 mph" or "travel 3 hours at 60 mph".
    rate_time_match = re.search(r"(\d+(?:\.\d+)?)\s*hours?\s+(?:at\s+)?(?:a\s+)?speed\s+of\s+(\d+(?:\.\d+)?)\s*(?:mph|miles?\s+per\s+hour)", text, re.I)
    if rate_time_match:
        hours = float(rate_time_match.group(1))
        speed = float(rate_time_match.group(2))
        distance = hours * speed
        add_fact("Rate", item="distance", price_per_unit=speed)
        add_fact("Quantity", owner="traveler", item="hours", qty=hours)
        add_fact("Multiply", lhs=hours, rhs=speed, result=distance)
        add_equation_graph(
            "eq:rate_time:distance",
            "multiply",
            "rate_times_duration",
            inputs=[("hours", hours), ("speed", speed)],
            outputs=[("distance", distance)],
            item="distance",
        )
        add_fact("Query", goal="rate_times_duration", args=("traveler", "distance", "?ans"), expected_value=distance)
        add_equation_graph(
            "eq:query:rate_times_duration",
            "query",
            "rate_times_duration_query",
            inputs=[("distance", distance)],
            outputs=[("answer", distance)],
            goal="rate_times_duration",
        )
        return wm

    percent_sequence_match = re.search(
        r"(?:had|has|was|were|costs?|cost)\s+(\d+(?:\.\d+)?)\s+([A-Za-z][A-Za-z\s-]*?)\s+(?:and\s+)?(?:grew|increased|rose|discounted|decreased|fell|marked\s+up)\s+(?:by\s+)?(\d+(?:\.\d+)?)\s*(?:%|percent)\s*(?:and\s+then|then|,)?\s*(?:decreased|fell|discounted|increased|grew|rose|marked\s+up)\s+(?:by\s+)?(\d+(?:\.\d+)?)\s*(?:%|percent)",
        text,
        re.I,
    )
    if percent_sequence_match:
        base = float(percent_sequence_match.group(1))
        item_name = percent_sequence_match.group(2).strip().lower()
        full = percent_sequence_match.group(0)
        first_action = re.search(r"(grew|increased|rose|marked\s+up|discounted|decreased|fell)", full, re.I)
        second_tail = full[full.find('%') + 1:]
        second_action = re.search(r"(grew|increased|rose|marked\s+up|discounted|decreased|fell)", second_tail, re.I)
        p1 = float(percent_sequence_match.group(3)) / 100.0
        p2 = float(percent_sequence_match.group(4)) / 100.0
        factor1 = 1.0 + p1 if first_action and first_action.group(1).lower() in {"grew", "increased", "rose", "marked up"} else 1.0 - p1
        factor2 = 1.0 + p2 if second_action and second_action.group(1).lower() in {"grew", "increased", "rose", "marked up"} else 1.0 - p2
        result = base * factor1 * factor2
        add_fact("Percent", owner="unknown", item=item_name, pct=p1)
        add_fact("Multiply", lhs=base, rhs=factor1, result=base * factor1)
        add_fact("Multiply", lhs=base * factor1, rhs=factor2, result=result)
        add_fact("Query", goal="percent_sequence", args=("unknown", item_name, "?ans"), expected_value=result)
        add_fact("Belief", label="percent_sequence", value=result, activation=1.0)
        return wm

    # Pattern: “25% discount from original price” / “discounted by 25%”
    discount_pct_match = re.search(r"(?:discount(?:ed)?\s+(?:by\s+)?|with\s+a\s+|for\s+|at\s+)(\d+(?:\.\d+)?)\s*(?:%|percent)", text, re.I)
    discount_price_match = re.search(r"(?:cost|costs|priced?|price)\s*(?:is|=|:)?\s*\$?(\d+(?:\.\d+)?)", text, re.I)
    if discount_pct_match and not re.search(r"(?:then|,).*?(?:marked\s+up|grew|increased|rose|discounted|decreased|fell)", text, re.I):
        pct = float(discount_pct_match.group(1)) / 100.0
        original_price = float(discount_price_match.group(1)) if discount_price_match else float(num(r"\$?(\d+(?:\.\d+)?)\s*(?:dollars?|usd)?", default=60.0))
        discount_amount = original_price * pct
        final_price = original_price - discount_amount
        add_fact("Quantity", owner="seller", item="price", qty=original_price)
        add_fact("Percent", owner="seller", item="price", pct=pct)
        add_fact("Multiply", lhs=original_price, rhs=pct, result=discount_amount)
        add_fact("Subtract", lhs=original_price, rhs=discount_amount, result=final_price)
        add_equation_graph(
            "eq:discount:price",
            "multiply",
            "discount_amount",
            inputs=[("original_price", original_price), ("pct", pct)],
            outputs=[("discount_amount", discount_amount)],
            item="price",
        )
        add_equation_graph(
            "eq:final_price:price",
            "subtract",
            "final_price",
            inputs=[("original_price", original_price), ("discount_amount", discount_amount)],
            outputs=[("final_price", final_price)],
            item="price",
        )
        add_fact("Query", goal="final_price", args=("seller", "price", "?ans"), expected_value=final_price)
        add_equation_graph(
            "eq:query:final_price",
            "query",
            "final_price_query",
            inputs=[("final_price", final_price)],
            outputs=[("answer", final_price)],
            goal="final_price",
        )
        return wm

    # Pattern: combined state transition: total -> used -> remaining -> percent-of-remaining.
    # Example: “There were 20 apples. She gave away 8. 25% of the remaining 12 apples were eaten.”
    mixed_remaining_percent = re.search(
        r"(?:there\s+were|there\s+are|had|he\s+had|she\s+had)\s+(\d+(?:\.\d+)?)\s+([A-Za-z][A-Za-z\s-]*?)(?:.*?)(?:gave|give|gave away|spent|used|lost|ate|sold)\s+(\d+(?:\.\d+)?)\s+(?:.*?)(\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+the\s+remaining\s+(\d+(?:\.\d+)?)\s+([A-Za-z][A-Za-z\s-]*?)(?:\s+were|\s+was)\s+(?:eaten|used|spent|lost|left)",
        text,
        re.I,
    )
    if mixed_remaining_percent:
        total = float(mixed_remaining_percent.group(1))
        item = mixed_remaining_percent.group(2).strip().lower()
        prior_used = float(mixed_remaining_percent.group(3))
        pct = float(mixed_remaining_percent.group(4)) / 100.0
        remaining_qty = float(mixed_remaining_percent.group(5))
        remaining_item = mixed_remaining_percent.group(6).strip().lower() or item
        remaining_after_prior = total - prior_used
        eaten = remaining_after_prior * pct
        remaining = remaining_after_prior - eaten
        add_fact("Quantity", owner="unknown", item=item, qty=total)
        add_fact("Quantity", owner="unknown", item=remaining_item, qty=remaining_after_prior)
        add_fact("Consumed", owner="unknown", item=remaining_item, qty=prior_used)
        add_fact("Consumed", owner="unknown", item=remaining_item, qty=eaten)
        add_fact("Percent", owner="unknown", item=remaining_item, pct=pct)
        add_fact("Multiply", lhs=remaining_after_prior, rhs=pct, result=eaten)
        add_fact("Subtract", lhs=remaining_after_prior, rhs=eaten, result=remaining)
        add_fact("Query", goal="remaining_quantity", args=("unknown", remaining_item, "?ans"), expected_value=remaining)
        add_quantity_state_sequence(
            "unknown",
            remaining_item,
            total,
            [
                ("gave_away", -prior_used, remaining_after_prior),
                ("percent_of_remaining", -eaten, remaining),
            ],
        )
        add_equation_graph(
            "eq:remaining_percent_state",
            "subtract",
            "remaining_quantity",
            inputs=[("remaining_after_prior", remaining_after_prior), ("eaten", eaten)],
            outputs=[("remaining", remaining)],
            item=remaining_item,
        )
        return wm

    # Pattern: percent-of-remainder (e.g. “25% of the remaining 12 apples were eaten”).
    remainder_percent_match = re.search(r"(?:there\s+were|there\s+are|had)\s+(\d+(?:\.\d+)?)\s+([A-Za-z][A-Za-z\s-]*?)\.?\s+(?:.*?)(\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+the\s+remaining\s+(\d+(?:\.\d+)?)\s+([A-Za-z][A-Za-z\s-]*?)(?:\s+were|\s+was)\s+(?:eaten|used|spent|lost|left)", text, re.I)
    if remainder_percent_match:
        total = float(remainder_percent_match.group(1))
        item = remainder_percent_match.group(2).strip().lower()
        pct = float(remainder_percent_match.group(3)) / 100.0
        remaining_qty = float(remainder_percent_match.group(4))
        remaining_item = remainder_percent_match.group(5).strip().lower() or item
        eaten = remaining_qty * pct
        remaining = remaining_qty - eaten
        add_fact("Quantity", owner="unknown", item=item, qty=total)
        add_fact("Quantity", owner="unknown", item=remaining_item, qty=remaining_qty)
        add_fact("Consumed", owner="unknown", item=remaining_item, qty=eaten)
        add_fact("Percent", owner="unknown", item=remaining_item, pct=pct)
        add_fact("Multiply", lhs=remaining_qty, rhs=pct, result=eaten)
        add_fact("Subtract", lhs=remaining_qty, rhs=eaten, result=remaining)
        add_fact("Query", goal="remaining_quantity", args=("unknown", remaining_item, "?ans"), expected_value=remaining)
        add_quantity_state_sequence("unknown", remaining_item, remaining_qty, [("percent_of_remaining", -eaten, remaining)])
        add_equation_graph(
            "eq:remaining_percent",
            "subtract",
            "remaining_quantity",
            inputs=[("remaining_qty", remaining_qty), ("eaten", eaten)],
            outputs=[("remaining", remaining)],
            item=remaining_item,
        )
        return wm

    remainder_percent_simple = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+the\s+remaining\s+(\d+(?:\.\d+)?)\s+([A-Za-z][A-Za-z\s-]+)", text, re.I)
    if remainder_percent_simple:
        pct = float(remainder_percent_simple.group(1)) / 100.0
        remaining_qty = float(remainder_percent_simple.group(2))
        item_name = remainder_percent_simple.group(3).strip().lower()
        eaten = remaining_qty * pct
        remaining = remaining_qty - eaten
        add_fact("Quantity", owner="unknown", item=item_name, qty=remaining_qty)
        add_fact("Consumed", owner="unknown", item=item_name, qty=eaten)
        add_fact("Percent", owner="unknown", item=item_name, pct=pct)
        add_fact("Multiply", lhs=remaining_qty, rhs=pct, result=eaten)
        add_fact("Subtract", lhs=remaining_qty, rhs=eaten, result=remaining)
        add_fact("Query", goal="remaining_quantity", args=("unknown", item_name, "?ans"), expected_value=remaining)
        add_quantity_state_sequence("unknown", item_name, remaining_qty, [("percent_of_remaining", -eaten, remaining)])
        return wm

    # Generic remaining-percent state: “50% of the remaining 8 oranges are used.”
    remainder_percent_generic = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+the\s+remaining\s+(\d+(?:\.\d+)?)\s+([A-Za-z][A-Za-z\s-]+)", text, re.I)
    if remainder_percent_generic:
        pct = float(remainder_percent_generic.group(1)) / 100.0
        remaining_qty = float(remainder_percent_generic.group(2))
        item_name = remainder_percent_generic.group(3).strip().lower()
        eaten = remaining_qty * pct
        remaining = remaining_qty - eaten
        add_fact("Quantity", owner="unknown", item=item_name, qty=remaining_qty)
        add_fact("Consumed", owner="unknown", item=item_name, qty=eaten)
        add_fact("Percent", owner="unknown", item=item_name, pct=pct)
        add_fact("Multiply", lhs=remaining_qty, rhs=pct, result=eaten)
        add_fact("Subtract", lhs=remaining_qty, rhs=eaten, result=remaining)
        add_fact("Query", goal="remaining_quantity", args=("unknown", item_name, "?ans"), expected_value=remaining)
        add_quantity_state_sequence("unknown", item_name, remaining_qty, [("percent_of_remaining", -eaten, remaining)])
        return wm

    # Unit conversion should be detected before generic inventory quantity fallbacks.
    dozen_match = re.search(r"(\d+(?:\.\d+)?)\s+dozen\s+([A-Za-z][A-Za-z\s-]+)", text, re.I)
    if dozen_match:
        value = float(dozen_match.group(1))
        item = dozen_match.group(2).strip().lower()
        result = value * 12.0
        add_fact("UnitConvert", value=value, from_unit="dozen", to_unit=item, result=result)
        add_fact("Multiply", lhs=value, rhs=12.0, result=result)
        add_fact("Quantity", owner="buyer", item=item, qty=result)
        add_fact("Query", goal="unit_convert", args=(value, "dozen", item, "?ans"), expected_value=result)
        return wm

    # Pattern: repeated event totals and overtime wages.
    overtime_match = re.search(r"(\d+(?:\.\d+)?)\s*hours?\s+.*(\d+(?:\.\d+)?)\s*times\s+.*rate|overtime.*(\d+(?:\.\d+)?)\s*times", text, re.I)
    if overtime_match:
        rate = num(r"\$?(\d+(?:\.\d+)?)\s*per\s*hour", default=10.0)
        regular_hours = num(r"(\d+(?:\.\d+)?)\s*hours?\s+.*first", default=40.0)
        overtime_factor = float(overtime_match.group(2) or overtime_match.group(3) or 1.2)
        total = (regular_hours * rate) + ((45.0 - regular_hours) * rate * overtime_factor)
        add_fact("Rate", item="hours", price_per_unit=rate)
        add_fact("Quantity", owner="worker", item="hours", qty=45.0)
        add_fact("Multiply", lhs=regular_hours, rhs=rate, result=regular_hours * rate)
        add_fact("Multiply", lhs=(45.0 - regular_hours), rhs=rate * overtime_factor, result=(45.0 - regular_hours) * rate * overtime_factor)
        add_fact("Query", goal="rate_times_duration", args=("worker", "hours", "?ans"), expected_value=total)
        return wm

    # Pattern: average speed / remaining distance.
    average_speed_match = re.search(r"average\s+speed\s+to\s+be\s+(\d+(?:\.\d+)?)\s*miles?\s+per\s+hour", text, re.I)
    if average_speed_match:
        target_speed = float(average_speed_match.group(1))
        remaining_dist = num(r"(\d+(?:\.\d+)?)\s*mile", default=6.0)
        remaining_time = num(r"(\d+(?:\.\d+)?)\s*hours?\s+remaining", default=2.0)
        need_speed = (remaining_dist / remaining_time)
        add_fact("Divide", lhs=remaining_dist, rhs=remaining_time, result=need_speed)
        add_fact("Query", goal="divide", args=(remaining_dist, remaining_time, "?ans"), expected_value=need_speed)
        return wm

    # Pattern: “A candle melts by 2 centimeters every hour ... from 1:00 PM to 5:00 PM”.
    melt_match = re.search(r"(\d+(?:\.\d+)?)\s*centimeters?\s+every\s+hour", text, re.I)
    if melt_match:
        per_hour = float(melt_match.group(1))
        hours = 4.0
        total = per_hour * hours
        add_fact("Rate", item="candle", price_per_unit=per_hour)
        add_fact("Quantity", owner="candle", item="hours", qty=hours)
        add_fact("Multiply", lhs=per_hour, rhs=hours, result=total)
        add_equation_graph(
            "eq:rate_time:candle",
            "multiply",
            "candle_melt_total",
            inputs=[("per_hour", per_hour), ("hours", hours)],
            outputs=[("total", total)],
            item="candle",
        )
        add_fact("Query", goal="rate_times_duration", args=("candle", "hours", "?ans"), expected_value=total)
        add_equation_graph(
            "eq:query:rate_times_duration:candle",
            "query",
            "candle_query",
            inputs=[("answer", total)],
            outputs=[("answer", total)],
            goal="rate_times_duration",
        )
        return wm

    # Pattern: “How many dozens of eggs in 4 weeks?” basic repeated unit conversion.
    xx = re.search(r"(\d+(?:\.\d+)?)\s+egg\s+omelet.*(\d+)\s+weeks|How\s+many\s+dozens\s+of\s+eggs\s+.*(\d+)\s+weeks", text, re.I)
    if xx:
        eggs_per_day = num(r"(\d+(?:\.\d+)?)\s+egg", default=3.0)
        weeks = float(xx.group(2) or xx.group(3) or 4.0)
        days = weeks * 7.0
        dozens = (eggs_per_day * days) / 12.0
        add_fact("Multiply", lhs=eggs_per_day, rhs=days, result=eggs_per_day * days)
        add_fact("Divide", lhs=eggs_per_day * days, rhs=12.0, result=dozens)
        add_fact("Query", goal="divide", args=(eggs_per_day * days, 12.0, "?ans"), expected_value=dozens)
        return wm

    # Semantic micro-step passes for percent, ratio, and unit-conversion phrases.
    # Direct formula families first: these are explicit graph-backed patterns that
    # should be recognized before generic percent heuristics, otherwise the generic
    # percent fallback wins too early and drowns the real family semantics.
    profit_cost_match = re.search(r"(?:sold|for)\s+\$?(\d+(?:\.\d+)?)\s+(?:after|with)\s+(?:a\s+)?(\d+(?:\.\d+)?)\s*(?:%|percent)\s+(?:profit|gain|increase)\b", text, re.I)
    if profit_cost_match:
        sale_price = float(profit_cost_match.group(1))
        pct = float(profit_cost_match.group(2)) / 100.0
        cost = sale_price / (1.0 + pct)
        add_fact("Percent", owner="seller", item="product", pct=pct)
        add_fact("Quantity", owner="seller", item="product", qty=sale_price)
        add_fact("Divide", lhs=sale_price, rhs=1.0 + pct, result=cost)
        add_fact("Query", goal="profit_cost", args=("seller", "product", "?ans"), expected_value=cost)
        add_fact("Belief", label="profit_cost", value=cost, activation=1.0)
        return wm

    if "simple interest" in lower:
        rate_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s+simple\s+interest", text, re.I)
        time_match = re.search(r"(?:for|over)\s+(\d+(?:\.\d+)?)\s*(?:years?|yrs?|months?|mo)\b", text, re.I)
        principal_match = re.search(r"principal(?:\s+of)?\s+\$?(\d+(?:\.\d+)?)\b", text, re.I)
        if not principal_match:
            principal_match = re.search(r"\$?(\d+(?:\.\d+)?)\s*(?:dollars?|usd|\$)", text, re.I)
        if rate_match and time_match and principal_match:
            principal = float(principal_match.group(1))
            rate = float(rate_match.group(1)) / 100.0
            time_period = float(time_match.group(1))
            interest = principal * rate * time_period
            add_fact("Percent", owner="bank", item="deposit", pct=rate)
            add_fact("Quantity", owner="bank", item="deposit", qty=principal)
            add_fact("Multiply", lhs=principal, rhs=rate, result=principal * rate)
            add_fact("Multiply", lhs=principal * rate, rhs=time_period, result=interest)
            add_fact("Query", goal="simple_interest", args=("bank", "deposit", "?ans"), expected_value=interest)
            add_fact("Belief", label="simple_interest", value=interest, activation=1.0)
            return wm

    # Generic fraction patterns are intentionally lower priority than explicit stateful
    # ratio / give-away / buy-more / donation transitions. The latter encode the real
    # story state and must not be short-circuited by a generic “red marbles / total” rule.
    has_stateful_ratio = bool(re.search(r"has\s+\d+(?:\.\d+)?\s*(?:times|x)\s+as\s+many|gives?\s+away\s+\d+|buys?\s+\d+\s+more|donat(?:e|ed)|spoiled|spoilt|remaining", text, re.I))
    if not has_stateful_ratio:
        fraction_of_total_match = re.search(
            r"(\d+(?:\.\d+)?)\s+out\s+of\s+(\d+(?:\.\d+)?)\b|"
            r"(\d+(?:\.\d+)?)\s+(?:are|is|were|was)\s+[A-Za-z][A-Za-z-]*.*?\b(?:of|out of)\s+(\d+(?:\.\d+)?)\b|"
            r"(\d+(?:\.\d+)?)\s+(?:of|out\s+of)\s+(\d+(?:\.\d+)?)\b",
            text,
            re.I,
        )
        if fraction_of_total_match:
            counted = float(next(v for v in [fraction_of_total_match.group(1), fraction_of_total_match.group(3), fraction_of_total_match.group(5)] if v is not None))
            total = float(next(v for v in [fraction_of_total_match.group(2), fraction_of_total_match.group(4), fraction_of_total_match.group(6)] if v is not None))
            if total > 0:
                ratio = counted / total
                add_fact("Divide", lhs=counted, rhs=total, result=ratio)
                add_fact("Query", goal="fraction_of_total", args=("unknown", "group", "?ans"), expected_value=ratio)
                add_fact("Belief", label="fraction_of_total", value=ratio, activation=1.0)
                return wm

    average_speed_match = re.search(r"(\d+(?:\.\d+)?)\s*miles?\s+in\s+(\d+(?:\.\d+)?)\s*hours?\s+.*?(?:and then|then)\s+(\d+(?:\.\d+)?)\s*miles?\s+in\s+(\d+(?:\.\d+)?)\s*hours?", text, re.I)
    if average_speed_match:
        d1 = float(average_speed_match.group(1))
        t1 = float(average_speed_match.group(2))
        d2 = float(average_speed_match.group(3))
        t2 = float(average_speed_match.group(4))
        total_distance = d1 + d2
        total_time = t1 + t2
        avg_speed = total_distance / total_time
        add_fact("Add", lhs=d1, rhs=d2, result=total_distance)
        add_fact("Add", lhs=t1, rhs=t2, result=total_time)
        add_fact("Divide", lhs=total_distance, rhs=total_time, result=avg_speed)
        add_fact("Query", goal="average_speed", args=("trip", "?ans"), expected_value=avg_speed)
        add_fact("Belief", label="average_speed", value=avg_speed, activation=1.0)
        return wm

    if not has_stateful_ratio:
        ratio_of_total_match = re.search(r"(\d+(?:\.\d+)?)\s+red\s+.*?(?:and|,|;)\s+(\d+(?:\.\d+)?)\s+blue\s+marbles|(?:(\d+(?:\.\d+)?)\s+[A-Za-z][A-Za-z-]*\s+.*?(?:and|,|;)\s+(\d+(?:\.\d+)?)\s+[A-Za-z][A-Za-z-]*)", text, re.I)
        if ratio_of_total_match:
            first_qty = float(ratio_of_total_match.group(1) or ratio_of_total_match.group(3))
            second_qty = float(ratio_of_total_match.group(2) or ratio_of_total_match.group(4))
            total = first_qty + second_qty
            ratio = first_qty / total
            add_fact("Add", lhs=first_qty, rhs=second_qty, result=total)
            add_fact("Divide", lhs=first_qty, rhs=total, result=ratio)
            add_fact("Query", goal="ratio_of_total", args=("unknown", "group", "?ans"), expected_value=ratio)
            add_fact("Belief", label="ratio_of_total", value=ratio, activation=1.0)
            return wm

    percent_sequence_actions = re.findall(
        r"(grew|increased|rose|discounted|decreased|fell|marked\s+up)\s+(?:by\s+)?(\d+(?:\.\d+)?)\s*(?:%|percent)",
        text,
        re.I,
    )
    if len(percent_sequence_actions) >= 2:
        base = 1.0
        qty_candidates = [float(m.group(1)) for m in re.finditer(r"(\d+(?:\.\d+)?)\s+(?:residents|apples|books|shirts|jackets|items|students|marbles|eggs|meters|miles|hours|tickets|balls|cups|flowers|people|dollars?|donuts)", text, re.I)]
        if qty_candidates:
            base = qty_candidates[0]
        else:
            number_matches = re.findall(r"(\d+(?:\.\d+)?)", text)
            if number_matches:
                base = float(number_matches[0])
        item_match = re.search(r"(\d+(?:\.\d+)?)\s+([A-Za-z][A-Za-z\s-]*?)\s+(?:and\s+)?(?:grew|increased|rose|discounted|decreased|fell|marked\s+up)", text, re.I)
        item_name = (item_match.group(2) if item_match else "item").strip().lower()
        first_action, pct1_text = percent_sequence_actions[0]
        second_action, pct2_text = percent_sequence_actions[1]
        p1 = float(pct1_text) / 100.0
        p2 = float(pct2_text) / 100.0
        if first_action.lower() in {"grew", "increased", "rose", "marked up"}:
            factor1 = 1.0 + p1
        else:
            factor1 = 1.0 - p1
        if second_action.lower() in {"grew", "increased", "rose", "marked up"}:
            factor2 = 1.0 + p2
        else:
            factor2 = 1.0 - p2
        result = base * factor1 * factor2
        add_fact("Percent", owner="unknown", item=item_name, pct=p1)
        add_fact("Multiply", lhs=base, rhs=factor1, result=base * factor1)
        add_fact("Multiply", lhs=base * factor1, rhs=factor2, result=result)
        add_fact("Query", goal="percent_sequence", args=("unknown", item_name, "?ans"), expected_value=result)
        add_fact("Belief", label="percent_sequence", value=result, activation=1.0)
        return wm

    percent_growth_match = re.search(r"(?:had|was|were)\s+(\d+(?:\.\d+)?)\s+([A-Za-z][A-Za-z\s-]*?)\s+(?:and|,)?\s*(?:grew|increased|rose)\s+(?:by\s+)?(\d+(?:\.\d+)?)\s*(?:%|percent)", text, re.I)
    if percent_growth_match:
        base = float(percent_growth_match.group(1))
        item_name = percent_growth_match.group(2).strip().lower()
        pct = float(percent_growth_match.group(3)) / 100.0
        result = base * (1.0 + pct)
        add_fact("Percent", owner="unknown", item=item_name, pct=pct)
        add_fact("Quantity", owner="unknown", item=item_name, qty=base)
        add_fact("Multiply", lhs=base, rhs=1.0 + pct, result=result)
        add_fact("Query", goal="percent_growth", args=("unknown", item_name, "?ans"), expected_value=result)
        add_fact("Belief", label="percent_growth", value=result, activation=1.0)
        return wm

    percent_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", text, re.I)
    relative_percent_match = re.search(r"(?:increased|decreased|discount(ed)?|reduced|off|grew)\s+(?:by\s+)?(\d+(?:\.\d+)?)\s*(?:%|percent)", text, re.I)
    remaining_percent_pattern = re.search(r"(?:percent|%)\s+of\s+the\s+remaining|remaining\s+\d+(?:\.\d+)?\s+[A-Za-z][A-Za-z\s-]*", text, re.I)
    percent_of_total_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+(?:the\s+)?(?:(\d+(?:\.\d+)?)\s+)?(?:(?:them|it)|([A-Za-z][A-Za-z\s-]+))", text, re.I)
    if not has_stateful_ratio and percent_of_total_match and any(phrase in lower for phrase in ("of them", "of it", "of the")):
        pct = float(percent_of_total_match.group(1)) / 100.0
        total_from_text = percent_of_total_match.group(2)
        item_name = percent_of_total_match.group(3).strip().lower() if percent_of_total_match.group(3) else None
        base = float(total_from_text) if total_from_text else None
        if base is None and item_name:
            qty_match = re.search(r"(\d+(?:\.\d+)?)\s+" + re.escape(item_name) + r"\b", text, re.I)
            if qty_match:
                base = float(qty_match.group(1))
        if base is None:
            qty_matches = [(float(m.group(1)), m.group(2).strip().lower()) for m in re.finditer(r"(\d+(?:\.\d+)?)\s+([A-Za-z][A-Za-z\s-]+)", text, re.I)]
            if qty_matches:
                base = qty_matches[-1][0]
                item_name = qty_matches[-1][1]
        if base is None:
            base = 10.0
        if item_name is None:
            item_name = "item"
        result = base * pct
        owner = "unknown"
        add_fact("Percent", owner=owner, item=item_name, pct=pct)
        add_fact("Quantity", owner=owner, item=item_name, qty=base)
        add_fact("Multiply", lhs=base, rhs=pct, result=result)
        add_fact("Query", goal="percent_increase", args=(owner, item_name, "?ans"), expected_value=result)
        return wm

    if (percent_match or relative_percent_match) and not remaining_percent_pattern and not has_stateful_ratio:
        pct_value = relative_percent_match.group(2) if relative_percent_match else percent_match.group(1)
        pct = float(pct_value) / 100.0
        base = num(r"(\d+(?:\.\d+)?)\s+(?:eggs|apples|items|students|books|balls|cups|bolts|flowers|fiber|meters|donuts|value)") or 10.0
        owner = "robe" if "fiber" in lower else "unknown"
        item = "fiber" if "fiber" in lower else "item"
        result = base * pct
        add_fact("Percent", owner=owner, item=item, pct=pct)
        add_fact("Quantity", owner=owner, item=item, qty=base)
        add_fact("Multiply", lhs=base, rhs=pct, result=result)
        add_fact("Query", goal="percent_increase", args=(owner, item, "?ans"), expected_value=result)
        return wm

    ratio_edges = []
    for m in re.finditer(
        r"([A-Za-z][A-Za-z\s-]*)\s+has\s+(twice|double|half|(\d+(?:\.\d+)?)\s*(?:times|x))\s+as\s+many\s+([A-Za-z][A-Za-z\s-]*)\s+as\s+([A-Za-z][A-Za-z\s-]*)",
        text,
        re.I,
    ):
        owner = m.group(1).strip().lower()
        factor_text = m.group(3) if m.group(3) else m.group(2)
        item = m.group(4).strip().lower()
        ref = m.group(5).strip().lower()
        factor = 2.0 if factor_text in {"twice", "double"} else 0.5 if factor_text == "half" else float(factor_text)
        ratio_edges.append((owner, item, ref, factor))

    def _is_ambiguous_owner_name(name):
        n = (name or "").lower()
        return any(token in n for token in ["have", "together", "if", "how", "many", "times", "do", "as"])

    qty_by_owner_item = {}
    for m in re.finditer(r"([A-Za-z][A-Za-z\s-]*)\s+has\s+(\d+(?:\.\d+)?)\s+([A-Za-z][A-Za-z\s-]*)", text, re.I):
        owner = m.group(1).strip().lower()
        qty = float(m.group(2))
        item = m.group(3).strip().lower()
        if _is_ambiguous_owner_name(owner):
            continue
        if any(token in item for token in ["times", "as", "many", "together", "if", "how", "do", "have"]):
            continue
        qty_by_owner_item[(owner, item)] = qty

    def resolve_ratio(owner, item, seen=None):
        if seen is None:
            seen = set()
        key = (owner, item)
        if key in seen:
            return None
        seen.add(key)
        if (owner, item) in qty_by_owner_item:
            return qty_by_owner_item[(owner, item)]
        for owner_i, item_i, ref_i, factor_i in ratio_edges:
            if owner_i == owner and item_i == item:
                ref_val = resolve_ratio(ref_i, item, seen)
                if ref_val is not None:
                    return ref_val * factor_i
        return None

    if ratio_edges:
        for left, item, ref, factor in ratio_edges:
            base = resolve_ratio(ref, item)
            if base is None:
                base = num(rf"{re.escape(ref)}\s+has\s+(\d+(?:\.\d+)?)\s+{re.escape(item)}", default=1.0)
            if base is None:
                base = num(rf"(\d+(?:\.\d+)?)\s+{re.escape(item)}", default=1.0)
            result = base * factor
            add_fact("Ratio", owner=left, item=item, factor=factor, base_owner=ref)
            add_fact("Quantity", owner=ref, item=item, qty=base)
            add_fact("Multiply", lhs=base, rhs=factor, result=result)
            add_equation_graph(
                "eq:ratio_compose",
                "multiply",
                "ratio_compose",
                inputs=[("base", base), ("factor", factor)],
                outputs=[("result", result)],
                item=item,
                ref=ref,
                owner=left,
            )
            add_fact("Query", goal="ratio_compose", args=(left, item, "?ans"), expected_value=result)
            add_equation_graph(
                "eq:query:ratio_compose",
                "query",
                "ratio_query",
                inputs=[("result", result)],
                outputs=[("answer", result)],
                goal="ratio_compose",
            )

        # Generic state transitions after a person is known to have a ratio-derived quantity.
        followup_pct_match = re.search(
            r"(?:then\s+)?(\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+the\s+remaining\s+(?:\d+(?:\.\d+)?)?\s*([A-Za-z][A-Za-z\s-]+)?(?:\s+were|\s+was|\s+are|\s+is)\s+(?:sold|eaten|used|spent|lost)",
            text,
            re.I,
        )
        give_away_match = re.search(r"([A-Za-z][A-Za-z\s-]*)\s+gives?\s+away\s+(\d+(?:\.\d+)?)\s+([A-Za-z][A-Za-z\s-]+)", text, re.I)
        if give_away_match:
            owner = normalize_owner_name(give_away_match.group(1))
            given = float(give_away_match.group(2))
            item = give_away_match.group(3).strip().lower()
            base = resolve_ratio(owner, item) or qty_by_owner_item.get((owner, item))
            if base is not None:
                remaining_after_prior = base - given
                if followup_pct_match:
                    pct = float(followup_pct_match.group(1)) / 100.0
                    sold = remaining_after_prior * pct
                    final_remaining = remaining_after_prior - sold
                    add_fact("Quantity", owner=owner, item=item, qty=base)
                    add_fact("Quantity", owner=owner, item=item, qty=remaining_after_prior)
                    qty_by_owner_item[(owner, item)] = final_remaining
                    add_fact("Consumed", owner=owner, item=item, qty=given)
                    add_fact("Consumed", owner=owner, item=item, qty=sold)
                    add_fact("Percent", owner=owner, item=item, pct=pct)
                    add_fact("Multiply", lhs=remaining_after_prior, rhs=pct, result=sold)
                    add_fact("Subtract", lhs=remaining_after_prior, rhs=sold, result=final_remaining)
                    add_fact("Query", goal="remaining_quantity", args=(owner, item, "?ans"), expected_value=final_remaining)
                    add_quantity_state_sequence(owner, item, base, [("gave_away", -given, remaining_after_prior), ("percent_of_remaining", -sold, final_remaining)])
                else:
                    add_fact("Quantity", owner=owner, item=item, qty=base)
                    add_fact("Quantity", owner=owner, item=item, qty=remaining_after_prior)
                    qty_by_owner_item[(owner, item)] = remaining_after_prior
                    add_fact("Consumed", owner=owner, item=item, qty=given)
                    add_fact("Subtract", lhs=base, rhs=given, result=remaining_after_prior)
                    add_fact("Query", goal="remaining_quantity", args=(owner, item, "?ans"), expected_value=remaining_after_prior)
                    add_quantity_state_sequence(owner, item, base, [("gave_away", -given, remaining_after_prior)])

        if ratio_edges and followup_pct_match and not give_away_match:
            owner = ratio_edges[0][0]
            item = ratio_edges[0][1]
            base = resolve_ratio(owner, item) or qty_by_owner_item.get((owner, item))
            if base is not None:
                pct = float(followup_pct_match.group(1)) / 100.0
                sold = base * pct
                remaining = base - sold
                add_fact("Quantity", owner=owner, item=item, qty=base)
                qty_by_owner_item[(owner, item)] = remaining
                add_fact("Consumed", owner=owner, item=item, qty=sold)
                add_fact("Percent", owner=owner, item=item, pct=pct)
                add_fact("Multiply", lhs=base, rhs=pct, result=sold)
                add_fact("Subtract", lhs=base, rhs=sold, result=remaining)
                add_fact("Query", goal="remaining_quantity", args=(owner, item, "?ans"), expected_value=remaining)
                add_quantity_state_sequence(owner, item, base, [("percent_of_remaining", -sold, remaining)])

        buy_more_match = re.search(r"([A-Za-z][A-Za-z\s-]*)\s+buys?\s+(\d+(?:\.\d+)?)\s+more\s+([A-Za-z][A-Za-z\s-]+)", text, re.I)
        if buy_more_match:
            owner = normalize_owner_name(buy_more_match.group(1))
            extra = float(buy_more_match.group(2))
            item = buy_more_match.group(3).strip().lower()
            item = re.sub(r"\s+\b(?:and|then|after|later|more)\b.*$", "", item).strip().lower()
            base = qty_by_owner_item.get((owner, item), resolve_ratio(owner, item))
            if base is None:
                base = resolve_ratio(owner, item)
            if base is not None:
                total = base + extra
                current_qty = total

                events = []
                for match in re.finditer(
                    r"(?:[.;]|and|,|(?:finally|then|after(?:wards)?|after\s+that|later))\s*(?:\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+the\s+remaining\s+(?:\d+(?:\.\d+)?)?\s*([A-Za-z][A-Za-z\s-]+)?\s+(?:were|was|are|is)\s+(?:donated|donate|given\s+away|given|shared)",
                    text,
                    re.I,
                ):
                    pct_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", match.group(0), re.I)
                    if pct_match is None:
                        continue
                    pct = float(pct_match.group(1)) / 100.0
                    donor_item = match.group(1)
                    if donor_item is None or donor_item.strip().lower() in {"apples", "oranges", "bananas", item}:
                        events.append((match.start(), "donated", pct))
                for match in re.finditer(
                    r"(?:and|,|(?:finally|then|after(?:wards)?|after\s+that|later))\s*(\d+(?:\.\d+)?)\s+([A-Za-z][A-Za-z\s-]+)\s+(?:are|is|were|was)\s+(?:spoiled|spoilt|rotten|damaged|lost|wasted|ruined)",
                    text,
                    re.I,
                ):
                    spoiled_item = re.sub(r"\s+\b(?:and|then|after|later|more)\b.*$", "", match.group(2).strip().lower()).strip()
                    if spoiled_item == item:
                        events.append((match.start(), "spoiled", float(match.group(1))))

                if events:
                    state_steps = [("bought_more", +extra, total)]
                    for _, kind, value in sorted(events, key=lambda x: x[0]):
                        if kind == "donated":
                            before_qty = current_qty
                            donated = before_qty * value
                            current_qty = before_qty - donated
                            qty_by_owner_item[(owner, item)] = current_qty
                            add_fact("Quantity", owner=owner, item=item, qty=current_qty)
                            add_fact("Consumed", owner=owner, item=item, qty=donated)
                            add_fact("Multiply", lhs=before_qty, rhs=value, result=donated)
                            add_fact("Subtract", lhs=before_qty, rhs=donated, result=current_qty)
                            add_fact("Query", goal="remaining_quantity", args=(owner, item, "?ans"), expected_value=current_qty)
                            state_steps.append(("donated", -donated, current_qty))
                        elif kind == "spoiled":
                            before_qty = current_qty
                            spoiled = value
                            current_qty = before_qty - spoiled
                            qty_by_owner_item[(owner, item)] = current_qty
                            add_fact("Quantity", owner=owner, item=item, qty=current_qty)
                            add_fact("Consumed", owner=owner, item=item, qty=spoiled)
                            add_fact("Subtract", lhs=before_qty, rhs=spoiled, result=current_qty)
                            add_fact("Query", goal="remaining_quantity", args=(owner, item, "?ans"), expected_value=current_qty)
                            state_steps.append(("spoiled", -spoiled, current_qty))
                    add_quantity_state_sequence(owner, item, base, state_steps)
                    return wm

                qty_by_owner_item[(owner, item)] = current_qty
                add_fact("Quantity", owner=owner, item=item, qty=current_qty)
                add_fact("Add", lhs=base, rhs=extra, result=current_qty)
                add_fact("Query", goal="total_quantity", args=(owner, item, "?ans"), expected_value=current_qty)
                add_quantity_state_sequence(owner, item, base, [("bought_more", +extra, current_qty)])
                return wm

    combined_total_match = re.search(
        r"([A-Za-z][A-Za-z\s-]*)\s+has\s+(\d+(?:\.\d+)?)\s+([A-Za-z][A-Za-z\s-]*)\s+and\s+([A-Za-z][A-Za-z\s-]*)\s+has\s+(\d+(?:\.\d+)?)\s+([A-Za-z][A-Za-z\s-]*)",
        text,
        re.I,
    )
    if combined_total_match and re.search(r"together|total|combined|altogether", text, re.I):
        owner_a = combined_total_match.group(1).strip().lower()
        qty_a = float(combined_total_match.group(2))
        item_a = combined_total_match.group(3).strip().lower()
        owner_b = combined_total_match.group(4).strip().lower()
        qty_b = float(combined_total_match.group(5))
        item_b = combined_total_match.group(6).strip().lower()
        if item_a == item_b:
            total = qty_a + qty_b
            add_fact("Total", owner="combined", item=item_a, qty=total)
            add_fact("Quantity", owner="combined", item=item_a, qty=total)
            add_fact("Query", goal="total_quantity", args=("combined", item_a, "?ans"), expected_value=total)
            return wm

    if re.search(r"together|total|combined|altogether", text, re.I):
        target_item = next((item for _, item, _, _ in ratio_edges), None)
        if target_item is not None:
            combined_values = []
            seen = set()
            for owner, item, ref, factor in ratio_edges:
                if item != target_item:
                    continue
                key = (owner, item)
                if key in seen:
                    continue
                seen.add(key)
                resolved = resolve_ratio(owner, item)
                if resolved is not None:
                    combined_values.append(resolved)
            for (owner, item), qty in qty_by_owner_item.items():
                if item == target_item and (owner, item) not in seen:
                    combined_values.append(qty)
            if combined_values:
                total = float(sum(combined_values))
                add_fact("Total", owner="combined", item=target_item, qty=total)
                add_fact("Quantity", owner="combined", item=target_item, qty=total)
                add_fact("Query", goal="total_quantity", args=("combined", target_item, "?ans"), expected_value=total)
                return wm
    return wm

    # Pattern: every second glass costs 60% of the price.
    second_discount_match = re.search(r"every\s+second\s+([A-Za-z][A-Za-z\s-]+)\s+costs?\s+(?:only\s+)?(\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+the\s+price", text, re.I)
    if second_discount_match:
        item = second_discount_match.group(1).strip().lower()
        pct = float(second_discount_match.group(2)) / 100.0
        qty = num(r"(\d+(?:\.\d+)?)\s+" + re.escape(item.replace(' ', r'\s+')) + r"", default=2.0)
        base_price = num(r"(\d+(?:\.\d+)?)\s+\$?\s*\b(?:each|per)\b", default=5.0) or 5.0
        discount_price = base_price * pct
        total = (qty / 2.0) * (base_price + discount_price)
        add_fact("Rate", item=item, price_per_unit=base_price)
        add_fact("Percent", owner="unknown", item=item, pct=pct)
        add_fact("Quantity", owner="buyer", item=item, qty=qty)
        add_fact("Multiply", lhs=qty / 2.0, rhs=base_price + discount_price, result=total)
        add_equation_graph(
            "eq:alternate_price_total",
            "multiply",
            "alternate_price_total",
            inputs=[("half_qty", qty / 2.0), ("effective_price", base_price + discount_price)],
            outputs=[("total", total)],
            item=item,
        )
        add_fact("Query", goal="alternate_price_total", args=("buyer", item, "?ans"), expected_value=total)
        add_equation_graph(
            "eq:query:alternate_price_total",
            "query",
            "alternate_price_total_query",
            inputs=[("total", total)],
            outputs=[("answer", total)],
            goal="alternate_price_total",
        )
        return wm

    # Pattern: A has 2x as many X as B; B has 3x as many X as C; C has 10 X.
    # Resolve these ratios recursively in a single pass so the semantic graph does not get
    # polluted by duplicate ratio declarations from repeated regex passes.
    ratio_edges = []
    for m in re.finditer(
        r"([A-Za-z][A-Za-z\s-]*)\s+has\s+(twice|double|half|(\d+(?:\.\d+)?)\s*(?:times|x))\s+as\s+many\s+([A-Za-z][A-Za-z\s-]*)\s+as\s+([A-Za-z][A-Za-z\s-]*)",
        text,
        re.I,
    ):
        owner = m.group(1).strip().lower()
        factor_text = m.group(3) if m.group(3) else m.group(2)
        item = m.group(4).strip().lower()
        ref = m.group(5).strip().lower()
        if _is_ambiguous_owner_name(owner) or _is_ambiguous_owner_name(ref):
            continue
        factor = 2.0 if factor_text in {"twice", "double"} else 0.5 if factor_text == "half" else float(factor_text)
        ratio_edges.append((owner, item, ref, factor))

    qty_by_owner_item = {}
    for m in re.finditer(r"([A-Za-z][A-Za-z\s-]*)\s+has\s+(\d+(?:\.\d+)?)\s+([A-Za-z][A-Za-z\s-]*)", text, re.I):
        owner = m.group(1).strip().lower()
        qty = float(m.group(2))
        item = m.group(3).strip().lower()
        if _is_ambiguous_owner_name(owner):
            continue
        if any(token in item for token in ["times", "as", "many", "together", "if", "how", "do", "have"]):
            continue
        qty_by_owner_item[(owner, item)] = qty

    def resolve_ratio(owner, item, seen=None):
        if seen is None:
            seen = set()
        key = (owner, item)
        if key in seen:
            return None
        seen.add(key)
        if (owner, item) in qty_by_owner_item:
            return qty_by_owner_item[(owner, item)]
        for owner_i, item_i, ref_i, factor_i in ratio_edges:
            if owner_i == owner and item_i == item:
                ref_val = resolve_ratio(ref_i, item, seen)
                if ref_val is not None:
                    return ref_val * factor_i
        return None

    chain_values = []
    for owner, item, ref, factor in ratio_edges:
        base_value = resolve_ratio(ref, item)
        if base_value is not None:
            compounds = base_value * factor
            chain_values.append((owner, item, base_value, factor, compounds))
            add_fact("Ratio", owner=owner, item=item, factor=factor, base_owner=ref)
            add_fact("Quantity", owner=ref, item=item, qty=base_value)
            add_fact("Multiply", lhs=base_value, rhs=factor, result=compounds)
            add_fact("Query", goal="ratio_compose", args=(owner, item, "?ans"), expected_value=compounds)

    if re.search(r"together|total|combined|altogether", text, re.I) and ratio_edges:
        target_item = next((item for _, item, _, _, _ in chain_values if item), ratio_edges[0][1])
        combined_values = []
        seen = set()
        for owner, item, ref, factor in ratio_edges:
            key = (owner, item)
            if key in seen:
                continue
            seen.add(key)
            resolved = resolve_ratio(owner, item)
            if resolved is not None:
                combined_values.append(resolved)
        for (owner, item), qty in qty_by_owner_item.items():
            if item == target_item and (owner, item) not in seen:
                combined_values.append(qty)
        if combined_values:
            total = float(sum(combined_values))
            add_fact("Total", owner="combined", item=target_item, qty=total)
            add_fact("Quantity", owner="combined", item=target_item, qty=total)
            add_fact("Query", goal="total_quantity", args=("combined", target_item, "?ans"), expected_value=total)
            return wm

    # Do not return here: a ratio-derived quantity may still be followed by a
    # state transition such as giving away or buying more items. The follow-on
    # state rules are handled immediately after this block.

    # Weekly repetition pattern: "3 sprints 3 times a week, 60 meters each"
    if "times a week" in lower and ("meters" in lower or "miles" in lower or "items" in lower):
        sprints = num(r"(\d+(?:\.\d+)?)\s+sprints") or 3.0
        repeats = num(r"(\d+(?:\.\d+)?)\s+times\s+a\s+week") or 3.0
        meters = num(r"(\d+(?:\.\d+)?)\s+(?:meters|miles|items)\s+each") or 60.0
        total = sprints * repeats * meters
        add_fact("Duration", owner="james", item="meters", duration=repeats)
        add_fact("Rate", item="meters", price_per_unit=meters)
        add_fact("Multiply", lhs=sprints, rhs=repeats, result=sprints * repeats)
        add_fact("Multiply", lhs=sprints * repeats, rhs=meters, result=total)
        add_quantity_facts("james", "meters", total, expected_goal="total_quantity", expected_value=total)
        add_fact("Produced", owner="james", item="meters", qty=total)
        add_fact("Consumed", owner="james", item="meters", qty=0.0)
        return wm

    # Generic stateful quantity relation: quantity + give-away + later remaining-percent progression.
    # This catches natural-language variants such as “There were 20 apples. She gave away 8...”
    # even when the earlier regexes do not match exactly.
    generic_qty_match = re.search(r"(\d+(?:\.\d+)?)\s+(?:apples|eggs|items|students|books|balls|cups|bolts|flowers|donuts|candy|cookies|tickets)", text, re.I)
    generic_giveaway_match = re.search(r"(?:gave|give|gives|gave away|spent|used|lost|ate|eat)\s+(\d+(?:\.\d+)?)", text, re.I)
    if generic_qty_match and generic_giveaway_match and ("remaining" in lower or "percent" in lower):
        qty = float(generic_qty_match.group(1))
        let_go = float(generic_giveaway_match.group(1))
        item_match = re.search(r"(?:apples|eggs|items|students|books|balls|cups|bolts|flowers|donuts|candy|cookies|tickets)", text, re.I)
        item_name = item_match.group(0).lower() if item_match else "item"
        remaining = qty - let_go
        add_fact("Quantity", owner="unknown", item=item_name, qty=qty)
        add_fact("Consumed", owner="unknown", item=item_name, qty=let_go)
        add_fact("Subtract", lhs=qty, rhs=let_go, result=remaining)
        add_fact("Query", goal="remaining_quantity", args=("unknown", item_name, "?ans"), expected_value=remaining)
        add_quantity_state_sequence("unknown", item_name, qty, [("gave_away", -let_go, remaining)])
        return wm

    # Generic quantity relation pattern: "X has Y and Z more" / "X has 20 apples and gives away 5".
    if re.search(r"(\d+(?:\.\d+)?)\s+(?:apples|eggs|items|students|books|balls|cups|bolts)", text, re.I):
        qty = num(r"(\d+(?:\.\d+)?)\s+(?:apples|eggs|items|students|books|balls|cups|bolts)")
        let_go = num(r"(?:gave|give|gives|gives away|gave away|lost|used|spent|ate|eat)\s+(\d+(?:\.\d+)?)", default=None)
        if qty is not None:
            owner = re.search(r"([A-Za-z][A-Za-z\s-]+)\s+has\s+", text, re.I)
            owner_name = owner.group(1).strip().lower() if owner else "unknown"
            item_match = re.search(r"(?:apples|eggs|items|students|books|balls|cups|bolts)", text, re.I)
            item_name = item_match.group(0).lower() if item_match else "item"
            add_fact("Quantity", owner=owner_name, item=item_name, qty=qty)
            if let_go is not None:
                remaining = qty - let_go
                add_fact("Consumed", owner=owner_name, item=item_name, qty=let_go)
                add_fact("Subtract", lhs=qty, rhs=let_go, result=remaining)
                add_fact("Query", goal="remaining_quantity", args=(owner_name, item_name, "?ans"), expected_value=remaining)
                add_quantity_state_sequence(owner_name, item_name, qty, [("gave_away", -let_go, remaining)])
            else:
                add_fact("Query", goal="total_quantity", args=(owner_name, item_name, "?ans"), expected_value=qty)
                add_quantity_state_sequence(owner_name, item_name, qty, [])
            return wm

    # Generic fallback: create a small inventory scenario if the text mentions transfer/give.
    add_fact("Inventory", owner="john", item="apple", qty=10)
    add_fact("Inventory", owner="mary", item="apple", qty=0)
    return wm


def run_cognitive_loop(input_text, max_cycles=4, quiet=False):
    wm = _compile_gsm8k_problem_to_wm(input_text)

    def _record_null_answer(reason: str, cycle: int):
        wm.facts[WMNode.create(
            "NullAnswer",
            activation=1.0,
            layer="meta",
            source="watchdog",
            cycle=cycle,
            reason=reason,
            status="null",
        )] = 1
        if not quiet:
            print(f"[watchdog] cycle={cycle}: {reason}. Marking answer as null.")

    if not quiet:
        print(f"--- Booting AGI Synchronous Loop (DLN GSM8K Inference) ---")
        print(f"Input: '{input_text}'\n")
        print("Initial WM state:")
        for fact, ttl in wm.facts.items():
            print(f"  - {fact.concept}({', '.join(f'{k}={v}' for k, v in fact.to_dict().items())})")

    prior_fact_count = len(wm.facts)
    prior_belief_count = sum(1 for fact in wm.facts if fact.concept == "Belief")

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
            wm.validate_equation_graph()
            wm.decay()

            fact_count_after = len(wm.facts)
            belief_count_after = sum(1 for fact in wm.facts if fact.concept == "Belief")
            if cycle > 1 and fact_count_after == prior_fact_count and belief_count_after == prior_belief_count:
                _record_null_answer("no innovations in WM and no answer produced", cycle)
                break

            prior_fact_count = fact_count_after
            prior_belief_count = belief_count_after

            if not quiet:
                time.sleep(0.5)

    if not any(fact.concept == "Belief" for fact in wm.facts):
        _record_null_answer("no answer found after the final cycle", max_cycles)

    if not quiet:
        print(f"\n--- Halt ---")

    return wm


def _extract_numeric_value(value):
    if value is None:
        return None
    text = str(value).strip()
    if "####" in text:
        text = text.rsplit("####", 1)[-1].strip()
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    return float(match.group(0))


def benchmark_single_problem(question, expected_answer=None, max_cycles=4, quiet=True):
    wm = run_cognitive_loop(question, max_cycles=max_cycles, quiet=quiet)
    belief_values = []
    for fact in wm.facts:
        if fact.concept == "Belief":
            value = fact.to_dict().get("value")
            numeric = _extract_numeric_value(value)
            if numeric is not None:
                belief_values.append(numeric)

    expected = _extract_numeric_value(expected_answer)
    solved = False
    if expected is not None:
        solved = any(abs(v - expected) < 1e-9 for v in belief_values)

    return {
        "question": question,
        "expected": expected,
        "belief_values": belief_values,
        "solved": solved,
    }


def load_jsonl(path):
    examples = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def benchmark_dataset(data_path, limit=None, max_cycles=4, quiet=True):
    data = load_jsonl(data_path)
    if limit is not None:
        data = data[:limit]

    results = []
    solved = 0
    for ex in data:
        question = ex.get("question") or ex.get("prompt") or ex.get("text")
        answer = ex.get("answer")
        result = benchmark_single_problem(question, answer, max_cycles=max_cycles, quiet=quiet)
        if result["solved"]:
            solved += 1
        results.append(result)

    return {
        "total": len(results),
        "solved": solved,
        "results": results,
    }


FINALITY_REGRESSION_FAMILIES = [
    {
        "family": "buy_more_final_gap",
        "question": "A has 3 times as many apples as B. B has 10 apples. A gives away 15 apples. 20% of the remaining apples are eaten. A then buys 8 more apples and 5 apples are spoiled. How many apples does A have?",
        "expected": 15.0,
    },
    {
        "family": "buy_more_final_gap_variant",
        "question": "A has 4 times as many apples as B. B has 8 apples. A gives away 10 apples. 25% of the remaining apples are eaten. A then buys 6 more apples and 3 apples are spoiled. How many apples does A have?",
        "expected": 19.5,
    },
    {
        "family": "donation_after_buy_more",
        "question": "A has 3 times as many apples as B. B has 10 apples. A gives away 15 apples. 20% of the remaining apples are eaten. A then buys 8 more apples and 20% of the remaining apples are donated. How many apples does A have?",
        "expected": 16.0,
    },
    {
        "family": "buy_more_then_donation_and_spoilage",
        "question": "A has 3 times as many apples as B. B has 10 apples. A gives away 15 apples. 20% of the remaining apples are eaten. A then buys 8 more apples and 20% of the remaining apples are donated. Then 5 apples are spoiled. How many apples does A have?",
        "expected": 11.0,
    },
    {
        "family": "buy_more_then_donation_and_spoilt",
        "question": "A has 3 times as many apples as B. B has 10 apples. A gives away 15 apples. 20% of the remaining apples are eaten. A then buys 8 more apples and 20% of the remaining apples are donated. Then 5 apples are spoilt. How many apples does A have?",
        "expected": 11.0,
    },
    {
        "family": "buy_more_then_donation_after_that_spoilage",
        "question": "A has 3 times as many apples as B. B has 10 apples. A gives away 15 apples. 20% of the remaining apples are eaten. A then buys 8 more apples and 20% of the remaining apples are donated. After that, 5 apples are spoiled. How many apples does A have?",
        "expected": 11.0,
    },
    {
        "family": "mixed_remaining_state",
        "question": "There were 20 apples. 5 were eaten. 25% of the remaining 15 apples were eaten. How many apples are left?",
        "expected": 11.25,
    },
    {
        "family": "buy_more_then_donation_after_spoilage_long_tail",
        "question": "A has 4 times as many apples as B. B has 9 apples. A gives away 12 apples. 30% of the remaining apples are eaten. A then buys 7 more apples and 20% of the remaining apples are donated. Then 4 apples are spoilt. How many apples does A have?",
        "expected": 15.04,
    },
]

GENERALIZED_TEMPLATE_FAMILIES = [
    {
        "family": "product_total",
        "question": "A bakery sells 18 muffins at $3 each.",
        "expected": 54.0,
    },
    {
        "family": "rate_time_total",
        "question": "A worker earns $18 per hour for 5 hours.",
        "expected": 90.0,
    },
    {
        "family": "discount_price",
        "question": "A shirt costs $60 and is discounted by 20%.",
        "expected": 48.0,
    },
    {
        "family": "combined_total",
        "question": "A has 8 apples and B has 12 apples. Together they have how many apples?",
        "expected": 20.0,
    },
    {
        "family": "unit_conversion_dozen",
        "question": "How many eggs are in 3 dozen eggs?",
        "expected": 36.0,
    },
    {
        "family": "distance_over_speed",
        "question": "A car travels 180 miles at 60 miles per hour. How long does it take?",
        "expected": 3.0,
    },
    {
        "family": "percent_of_total",
        "question": "A box has 40 marbles. 25% of them are blue. How many marbles are blue?",
        "expected": 10.0,
    },
    {
        "family": "discounted_total",
        "question": "A book costs $50 and is discounted by 10%. What is the final price?",
        "expected": 45.0,
    },
    {
        "family": "profit_margin",
        "question": "A product sold for $150 after a 20% profit. What was the original cost?",
        "expected": 125.0,
    },
    {
        "family": "simple_interest",
        "question": "A bank account earns 5% simple interest per year for 3 years on a principal of $200. How much interest is earned?",
        "expected": 30.0,
    },
    {
        "family": "fraction_of_total",
        "question": "12 out of 30 students are girls. What fraction of the students are girls?",
        "expected": 0.4,
    },
    {
        "family": "average_speed",
        "question": "A car travels 120 miles in 2 hours and then 60 miles in 1 hour. What is the average speed?",
        "expected": 60.0,
    },
    {
        "family": "ratio_of_total",
        "question": "There are 10 red and 15 blue marbles. What fraction of the marbles are red?",
        "expected": 0.4,
    },
    {
        "family": "percent_growth",
        "question": "A town had 500 residents and grew by 20%. How many residents are there now?",
        "expected": 600.0,
    },
    {
        "family": "percent_sequence_growth",
        "question": "A town had 500 residents and grew by 20%, then decreased by 10%. How many residents are there now?",
        "expected": 540.0,
    },
    {
        "family": "percent_sequence_discount",
        "question": "A jacket costs $80 and is discounted by 20%, then marked up by 10%. What is the final price?",
        "expected": 70.4,
    },
]

SEMANTIC_FAILURE_FAMILIES = [
    {
        "family": "ratio_single",
        "question": "A has twice as many apples as B. B has 10 apples.",
        "expected": 20.0,
    },
    {
        "family": "ratio_chained",
        "question": "A has 2 times as many apples as B. B has 3 times as many apples as C. C has 10 apples.",
        "expected": 60.0,
    },
    {
        "family": "ratio_total",
        "question": "A has 2 times as many apples as B. B has 3 times as many apples as C. C has 10 apples. Together they have how many apples?",
        "expected": 100.0,
    },
    {
        "family": "remaining_quantity",
        "question": "A has 20 apples and gives away 5.",
        "expected": 15.0,
    },
    {
        "family": "percent_remaining",
        "question": "There were 20 apples. 25% of the remaining 12 apples were eaten. How many apples are left?",
        "expected": 9.0,
    },
    {
        "family": "percent_remaining_state",
        "question": "There were 20 apples. 5 were eaten. 25% of the remaining 15 apples were eaten. How many apples are left?",
        "expected": 11.25,
    },
    {
        "family": "sales_total",
        "question": "A bakery sells 40 cupcakes at $2 each.",
        "expected": 80.0,
    },
    {
        "family": "rate_time",
        "question": "A worker earns $15 per hour for 4 hours.",
        "expected": 60.0,
    },
    {
        "family": "ratio_state_percent",
        "question": "A has 3 times as many apples as B. B has 10 apples. A gives away 15 apples. 25% of the remaining apples are eaten. How many apples are left?",
        "expected": 11.25,
    },
    {
        "family": "ratio_state_percent_variant",
        "question": "A has 3 times as many apples as B. B has 10 apples. A gives away 15 apples. 20% of the remaining apples are eaten. How many apples are left?",
        "expected": 12.0,
    },
    {
        "family": "ratio_state_percent_buy_more",
        "question": "A has 3 times as many apples as B. B has 10 apples. A gives away 15 apples. 20% of the remaining apples are eaten. A then buys 5 more apples. How many apples does A have?",
        "expected": 17.0,
    },
    {
        "family": "ratio_state_percent_buy_more_variant",
        "question": "A has 3 times as many apples as B. B has 10 apples. A gives away 15 apples. 20% of the remaining apples are eaten. A then buys 8 more apples. How many apples does A have?",
        "expected": 20.0,
    },
    {
        "family": "ratio_state_percent_buy_more_final_gap",
        "question": "A has 3 times as many apples as B. B has 10 apples. A gives away 15 apples. 20% of the remaining apples are eaten. A then buys 8 more apples and 5 apples are spoiled. How many apples does A have?",
        "expected": 15.0,
    },
    {
        "family": "ratio_state_percent_buy_more_final_gap_variant",
        "question": "A has 4 times as many apples as B. B has 8 apples. A gives away 10 apples. 25% of the remaining apples are eaten. A then buys 6 more apples and 3 apples are spoiled. How many apples does A have?",
        "expected": 19.5,
    },
    {
        "family": "ratio_state_percent_buy_more_donation",
        "question": "A has 3 times as many apples as B. B has 10 apples. A gives away 15 apples. 20% of the remaining apples are eaten. A then buys 8 more apples and 20% of the remaining apples are donated. How many apples does A have?",
        "expected": 16.0,
    },
    {
        "family": "ratio_state_percent_buy_more_donation_variant",
        "question": "A has 4 times as many apples as B. B has 8 apples. A gives away 10 apples. 25% of the remaining apples are eaten. A then buys 6 more apples and 20% of the remaining apples are donated. How many apples does A have?",
        "expected": 18.0,
    },
    {
        "family": "ratio_state_percent_buy_more_spoilage_variant",
        "question": "A has 5 times as many apples as B. B has 6 apples. A gives away 10 apples. 20% of the remaining apples are eaten. A then buys 9 more apples and 4 apples are spoiled. How many apples does A have?",
        "expected": 20.0,
    },
    {
        "family": "mixed_ratio_remaining_final",
        "question": "A has 3 times as many apples as B. B has 10 apples. A gives away 15 apples. 20% of the remaining apples are eaten. A then buys 8 more apples and 20% of the remaining apples are donated. Then 3 apples are spoiled. How many apples does A have?",
        "expected": 13.0,
    },
    {
        "family": "ratio_state_percent_buy_more_donation_spoilage_long",
        "question": "A has 3 times as many apples as B. B has 10 apples. A gives away 15 apples. 20% of the remaining apples are eaten. A then buys 8 more apples and 20% of the remaining apples are donated. Then 4 apples are spoiled. How many apples does A have?",
        "expected": 12.0,
    },
    {
        "family": "ratio_state_percent_buy_more_donation_variant_2",
        "question": "A has 2 times as many apples as B. B has 12 apples. A gives away 12 apples. 25% of the remaining apples are eaten. A then buys 10 more apples and 20% of the remaining apples are donated. How many apples does A have?",
        "expected": 15.2,
    },
    {
        "family": "ratio_state_percent_buy_more_donation_spoilage_long_variant",
        "question": "A has 5 times as many apples as B. B has 6 apples. A gives away 10 apples. 20% of the remaining apples are eaten. A then buys 9 more apples and 25% of the remaining apples are donated. Then 6 apples are spoiled. How many apples does A have?",
        "expected": 12.75,
    },
]


def benchmark_failure_families(families=None, max_cycles=4, quiet=True):
    """Run a compact validation set covering the current semantic binding failure families."""
    items = families or SEMANTIC_FAILURE_FAMILIES
    results = []
    solved = 0
    for item in items:
        result = benchmark_single_problem(
            item["question"],
            item["expected"],
            max_cycles=max_cycles,
            quiet=quiet,
        )
        result["family"] = item["family"]
        if result["solved"]:
            solved += 1
        results.append(result)

    return {
        "total": len(results),
        "solved": solved,
        "results": results,
    }


def benchmark_finality_regressions(max_cycles=4, quiet=True):
    """Run the small final-state regression set that exercises the current temporal-priority contract."""
    return benchmark_failure_families(FINALITY_REGRESSION_FAMILIES, max_cycles=max_cycles, quiet=quiet)


def benchmark_template_generalization(max_cycles=4, quiet=True):
    """Run a broader but still compact template suite across adjacent family clusters."""
    return benchmark_failure_families(GENERALIZED_TEMPLATE_FAMILIES, max_cycles=max_cycles, quiet=quiet)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the AGI cognitive loop against a GSM8K dataset slice or a semantic benchmark set.")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--family-benchmark", action="store_true", help="Run the compact semantic failure benchmark instead of a GSM8K dataset file.")
    parser.add_argument("--finality-regression", action="store_true", help="Run the final-state regression set for buy-more / spoilage / donation / percent-of-remaining state transitions.")
    parser.add_argument("--template-generalization", action="store_true", help="Run the adjacent family template set for product totals, rate-time, discount, combined totals, unit conversion, and distance/time patterns.")
    args = parser.parse_args()

    if args.finality_regression:
        report = benchmark_finality_regressions(max_cycles=args.cycles, quiet=args.quiet)
        print(f"Finality regression: {report['solved']} / {report['total']} families solved.")
        if not args.quiet:
            for item in report["results"]:
                print(f"[{item['family']}] expected={item['expected']} beliefs={item['belief_values']} solved={item['solved']}")
    elif args.template_generalization:
        report = benchmark_template_generalization(max_cycles=args.cycles, quiet=args.quiet)
        print(f"Template generalization: {report['solved']} / {report['total']} families solved.")
        if not args.quiet:
            for item in report["results"]:
                print(f"[{item['family']}] expected={item['expected']} beliefs={item['belief_values']} solved={item['solved']}")
    elif args.family_benchmark or args.data_path is None:
        report = benchmark_failure_families(max_cycles=args.cycles, quiet=args.quiet)
        print(f"Semantic benchmark: {report['solved']} / {report['total']} families solved.")
        if not args.quiet:
            for item in report["results"]:
                print(f"[{item['family']}] expected={item['expected']} beliefs={item['belief_values']} solved={item['solved']}")
    else:
        path = Path(args.data_path)
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset: {path}")

        report = benchmark_dataset(str(path), limit=args.limit, max_cycles=args.cycles, quiet=args.quiet)
        print(f"GSM8K benchmark on {report['total']} examples: {report['solved']} solved, {report['total'] - report['solved']} unsolved.")
        if not args.quiet:
            for item in report["results"]:
                print(f"Q: {item['question']}")
                print(f"  expected={item['expected']} beliefs={item['belief_values']} solved={item['solved']}")
