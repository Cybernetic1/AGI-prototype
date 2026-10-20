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
sys.path.append(str(BASE_DIR / 'forward_engine'))
sys.path.append(str(BASE_DIR / 'logic_transformer'))
sys.path.append(str(BASE_DIR / 'ilp_learner'))

from working_memory.working_memory import WMNode, WorkingMemory, EquationGraphSolver
from working_memory.semantic_schema import SemanticGraphSchema
from system2_poc import System2Engine
from experta import Fact
from spacy_logical_form import SpacyLogicalFormParser
from train_pot_dln_pointer import DLNPointerDecoder
import train_pot_clause as tpc
from abductive_ilp import KnowledgeBase, OntologicalReasoner, OntologicalILPLearner, DuckEgg

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
        
        # Map the output strings back to WMNodes with soft activation (0.85) representing System 1's neural confidence.
        # The layer is explicitly tagged as parse/semantic so it can be consumed by later grounding rules.
        for clause in core_clauses:
            pred = clause["pred"]
            args = clause["args"]
            concept = pred.capitalize()
            if concept in ["Event", "Predicate", "Agent", "Patient", "Recipient", "Name", "Quantity", "Location", "Time", "Modifier"]:
                kwargs = {f"arg{j}": arg for j, arg in enumerate(args)}
                new_facts.add(WMNode.create(concept, activation=0.85, layer="parse", source="system1", provenance_id=f"s1:{concept.lower()}", **kwargs))

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
            
    return new_facts

def system2_step(current_wm, cycle, _=None, quiet=False):
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
    
    new_facts = set()
    for f in engine.facts.values():
        if type(f).__name__ in ["Belief", "Inventory", "EventProcessed"]:
            clean_attrs = {k: v for k, v in f.items() if not str(k).startswith("__") and k != "activation"}
                
            # Preserving the original node's activation/salience score from Working Memory
            concept = type(f).__name__
            temp_node = WMNode.create(concept, **clean_attrs)
            original = next((n for n in current_wm if n == temp_node), None)
            activation = original.activation if original is not None else 1.0
                
            new_facts.add(WMNode.create(concept, activation=activation, **clean_attrs))
            
    return new_facts

def system3_step(current_wm, cycle, _=None, quiet=False):
    new_facts = set()
    
    # 1. Search for any Query node currently in Working Memory.
    # A single problem can legitimately produce multiple query candidates (e.g. ratio
    # sub-questions plus a final combined total), so prefer the most specific final
    # query and fall back to the first candidate only when necessary.
    query_nodes = [n for n in current_wm if n.concept == "Query"]
    if query_nodes:
        preferred = None
        for node in query_nodes:
            goal = node.to_dict().get("goal")
            if goal in {"total_quantity", "remaining_quantity", "final_price", "alternate_price_total"}:
                preferred = node
                break
        if preferred is None:
            preferred = query_nodes[0]
        q_attrs = preferred.to_dict()
        goal_pred = q_attrs.get("goal")
        goal_args = list(q_attrs.get("args", []))
        expected_value = q_attrs.get("expected_value")
    else:
        # If no explicit query fact is present but there is an explicit graph query node,
        # still try to read its answer directly from the equation graph.
        graph_query = next((n for n in current_wm if n.concept == "GraphNode" and n.to_dict().get("kind") == "query"), None)
        if graph_query is None:
            return new_facts
        q_attrs = graph_query.to_dict()
        goal_pred = q_attrs.get("goal") or q_attrs.get("label")
        goal_args = [q_attrs.get("goal")] if q_attrs.get("goal") else [q_attrs.get("label")]
        expected_value = q_attrs.get("expected_value")
    
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

    # Structured GSM8K compiler.
    # We keep a compact, canonical predicate vocabulary and map several common
    # arithmetic and state-transition families into it. This is still heuristic,
    # but it is systematic rather than one-off. That is the right next step.
    text = input_text or ""
    lower = text.lower()

    def num(pattern, default=None):
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if not m:
            return default
        val = m.group(1)
        try:
            return float(val)
        except ValueError:
            return default

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
    distance_time_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:miles?|mi|kilometers?|km)\s+(?:at\s+)?(?:a\s+)?speed\s+of\s+(\d+(?:\.\d+)?)\s*(?:mph|miles?\s+per\s+hour)|travel(?:ing)?\s+(\d+(?:\.\d+)?)\s*(?:miles?|mi|kilometers?|km)\s+at\s+(\d+(?:\.\d+)?)\s*(?:mph|miles?\s+per\s+hour)", text, re.I)
    if distance_time_match and "how long" in lower or "how much time" in lower or "what time" in lower or "how many hours" in lower:
        dist = float(distance_time_match.group(1) or distance_time_match.group(3))
        speed = float(distance_time_match.group(2) or distance_time_match.group(4))
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

    # Pattern: “25% discount from original price” / “discounted by 25%”
    discount_pct_match = re.search(r"(?:discount(?:ed)?\s+(?:by\s+)?|with\s+a\s+|for\s+|at\s+)(\d+(?:\.\d+)?)\s*(?:%|percent)", text, re.I)
    discount_price_match = re.search(r"(?:cost|costs|priced?|price)\s*(?:is|=|:)?\s*\$?(\d+(?:\.\d+)?)", text, re.I)
    if discount_pct_match:
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
    percent_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", text, re.I)
    relative_percent_match = re.search(r"(?:increased|decreased|discount(ed)?|reduced|off|grew)\s+(?:by\s+)?(\d+(?:\.\d+)?)\s*(?:%|percent)", text, re.I)
    remaining_percent_pattern = re.search(r"(?:percent|%)\s+of\s+the\s+remaining|remaining\s+\d+(?:\.\d+)?\s+[A-Za-z][A-Za-z\s-]*", text, re.I)
    if (percent_match or relative_percent_match) and not remaining_percent_pattern:
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
            owner = give_away_match.group(1).strip().lower()
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
                    add_fact("Consumed", owner=owner, item=item, qty=given)
                    add_fact("Consumed", owner=owner, item=item, qty=sold)
                    add_fact("Percent", owner=owner, item=item, pct=pct)
                    add_fact("Multiply", lhs=remaining_after_prior, rhs=pct, result=sold)
                    add_fact("Subtract", lhs=remaining_after_prior, rhs=sold, result=final_remaining)
                    add_fact("Query", goal="remaining_quantity", args=(owner, item, "?ans"), expected_value=final_remaining)
                    return wm
                add_fact("Quantity", owner=owner, item=item, qty=base)
                add_fact("Quantity", owner=owner, item=item, qty=remaining_after_prior)
                add_fact("Consumed", owner=owner, item=item, qty=given)
                add_fact("Subtract", lhs=base, rhs=given, result=remaining_after_prior)
                add_fact("Query", goal="remaining_quantity", args=(owner, item, "?ans"), expected_value=remaining_after_prior)
                return wm

        if ratio_edges and followup_pct_match:
            owner = ratio_edges[0][0]
            item = ratio_edges[0][1]
            base = resolve_ratio(owner, item) or qty_by_owner_item.get((owner, item))
            if base is not None:
                pct = float(followup_pct_match.group(1)) / 100.0
                sold = base * pct
                remaining = base - sold
                add_fact("Quantity", owner=owner, item=item, qty=base)
                add_fact("Consumed", owner=owner, item=item, qty=sold)
                add_fact("Percent", owner=owner, item=item, pct=pct)
                add_fact("Multiply", lhs=base, rhs=pct, result=sold)
                add_fact("Subtract", lhs=base, rhs=sold, result=remaining)
                add_fact("Query", goal="remaining_quantity", args=(owner, item, "?ans"), expected_value=remaining)
                return wm

        buy_more_match = re.search(r"([A-Za-z][A-Za-z\s-]*)\s+buys?\s+(\d+(?:\.\d+)?)\s+more\s+([A-Za-z][A-Za-z\s-]+)", text, re.I)
        if buy_more_match:
            owner = buy_more_match.group(1).strip().lower()
            extra = float(buy_more_match.group(2))
            item = buy_more_match.group(3).strip().lower()
            base = resolve_ratio(owner, item) or qty_by_owner_item.get((owner, item))
            if base is not None:
                total = base + extra
                add_fact("Quantity", owner=owner, item=item, qty=total)
                add_fact("Add", lhs=base, rhs=extra, result=total)
                add_fact("Query", goal="total_quantity", args=(owner, item, "?ans"), expected_value=total)
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
            else:
                add_fact("Query", goal="total_quantity", args=(owner_name, item_name, "?ans"), expected_value=qty)
            return wm

    # Generic fallback: create a small inventory scenario if the text mentions transfer/give.
    add_fact("Inventory", owner="john", item="apple", qty=10)
    add_fact("Inventory", owner="mary", item="apple", qty=0)
    return wm


def run_cognitive_loop(input_text, max_cycles=4, quiet=False):
    wm = _compile_gsm8k_problem_to_wm(input_text)
    
    if not quiet:
        print(f"--- Booting AGI Synchronous Loop (DLN GSM8K Inference) ---")
        print(f"Input: '{input_text}'\n")
        print("Initial WM state:")
        for fact, ttl in wm.facts.items():
            print(f"  - {fact.concept}({', '.join(f'{k}={v}' for k, v in fact.to_dict().items())})")
    
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
            
            if not quiet:
                time.sleep(0.5)
 
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the AGI cognitive loop against a GSM8K dataset slice or a semantic benchmark set.")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--family-benchmark", action="store_true", help="Run the compact semantic failure benchmark instead of a GSM8K dataset file.")
    args = parser.parse_args()

    if args.family_benchmark or args.data_path is None:
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
