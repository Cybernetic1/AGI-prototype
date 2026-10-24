from __future__ import annotations

import argparse
import re
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util

if __package__ in (None, ""):
    package_dir = Path(__file__).resolve().parent
    module_path = package_dir / "logic_transformer.py"
    spec = importlib.util.spec_from_file_location("logic_transformer_pkg_logic", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    LogicTransformer = module.LogicTransformer

    parser_path = package_dir / "spacy_logical_form.py"
    parser_spec = importlib.util.spec_from_file_location("logic_transformer_pkg_parser", parser_path)
    parser_module = importlib.util.module_from_spec(parser_spec)
    parser_spec.loader.exec_module(parser_module)
    SpacyLogicalFormParser = parser_module.SpacyLogicalFormParser
else:
    from .logic_transformer import LogicTransformer
    from .spacy_logical_form import SpacyLogicalFormParser

from working_memory.working_memory import WMNode, WorkingMemory

random.seed(7)
torch.manual_seed(7)

OWNER_IDS = {"bob": 1.0 / 3.0, "alice": 2.0 / 3.0}
ITEM_IDS = {"apples": 1.0 / 3.0, "oranges": 2.0 / 3.0, "apples": 1.0 / 3.0}
REL_IDS = {"has": 0.0, "lose": 1.0 / 3.0, "query": 2.0 / 3.0, "quantity": 1.0 / 6.0}
PROP_DIM = 8
MAX_PROPS = 3
NULL_ERROR_PENALTY = 1e6


def encode_relation(rel_name: str, owner: str, item: str, value: float):
    owner_key = owner.lower()
    item_key = item.lower()
    if owner_key not in OWNER_IDS:
        owner_key = "bob"
    if item_key not in ITEM_IDS:
        item_key = "apples"
    return [
        REL_IDS[rel_name],
        OWNER_IDS[owner_key],
        ITEM_IDS[item_key],
        float(value) / 50.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]


def encode_wme(owner: str, item: str, quantity: float):
    owner_key = owner.lower()
    item_key = item.lower()
    if owner_key not in OWNER_IDS:
        owner_key = "bob"
    if item_key not in ITEM_IDS:
        item_key = "apples"
    return [
        OWNER_IDS[owner_key],
        ITEM_IDS[item_key],
        1.0,
        0.0,
        float(quantity) / 50.0,
        0.0,
        float(quantity) / 50.0,
        0.0,
    ]


def build_example() -> Tuple[List[List[float]], List[float]]:
    owner = random.choice(["bob", "alice"])
    item = random.choice(["apples", "oranges"])
    start = random.randint(5, 30)
    lost = random.randint(1, 12)
    final = start - lost
    props = [
        encode_relation("has", owner, item, start),
        encode_relation("lose", owner, item, lost),
        encode_relation("query", owner, item, 0.0),
    ]
    target = encode_wme(owner, item, final)
    return props, target


def build_dataset(n: int):
    return [build_example() for _ in range(n)]


def discretize_slot(value: float, options: Dict[str, float]) -> str:
    return min(options, key=lambda symbol: abs(value - options[symbol]))


def discretize_proposition(vec: List[float]) -> Dict[str, object]:
    owner = discretize_slot(float(vec[1]), OWNER_IDS)
    item = discretize_slot(float(vec[2]), ITEM_IDS)
    rel = discretize_slot(float(vec[0]), REL_IDS)
    qty = max(0.0, float(vec[3]) * 50.0)
    return {"owner": owner, "item": item, "rel": rel, "qty": round(qty, 2)}


def proposition_to_wme(proposition: Dict[str, object]) -> WMNode:
    rel = proposition["rel"]
    owner = proposition["owner"]
    item = proposition["item"]
    qty = proposition["qty"]

    if rel == "has":
        return WMNode.create("Has", activation=0.82, layer="semantic", source="lt-proposal", owner=owner, item=item, qty=float(qty))
    if rel == "lose":
        return WMNode.create("Lose", activation=0.82, layer="semantic", source="lt-proposal", owner=owner, item=item, qty=float(qty))
    if rel == "query":
        return WMNode.create("Query", activation=0.82, layer="semantic", source="lt-proposal", owner=owner, item=item, goal="remaining_quantity")
    return WMNode.create("Quantity", activation=0.82, layer="semantic", source="lt-proposal", owner=owner, item=item, qty=float(qty))


def null_answer(reason: str, penalty: float = NULL_ERROR_PENALTY):
    return WMNode.create(
        "NullAnswer",
        activation=1.0,
        layer="meta",
        source="watchdog",
        reason=reason,
        status="null",
        error_penalty=float(penalty),
    )


def parse_lf(text: str):
    parser = SpacyLogicalFormParser()
    lf = parser.parse(text)
    print("[LF parsed by spaCy]:")
    for line in lf.render().splitlines()[:20]:
        print("  ", line)
    return lf


def lf_to_vector_proposals(text: str) -> Iterable[List[float]]:
    parse_lf(text)
    vectors: List[List[float]] = []
    patterns = [
        (r"\b([A-Z][a-z]+)\s+has\s+(\d+(?:\.\d+)?)\s+([A-Za-z]+)\b", "has"),
        (r"\b([A-Z][a-z]+)\s+lost\s+(\d+(?:\.\d+)?)\s+([A-Za-z]+)\b", "lose"),
        (r"\b([A-Z][a-z]+)\s+has\s+(\d+(?:\.\d+)?)\s+([A-Za-z]+)\s+and\s+lost\s+(\d+(?:\.\d+)?)\s+([A-Za-z]+)\b", "query"),
    ]
    for pattern, rel in patterns:
        for m in re.finditer(pattern, text, re.I):
            if rel == "has":
                owner, qty, item = m.groups()
                vectors.append(encode_relation("has", owner.lower(), item.lower(), float(qty)))
            elif rel == "lose":
                owner, qty, item = m.groups()
                vectors.append(encode_relation("lose", owner.lower(), item.lower(), float(qty)))
            else:
                owner, qty_start, item, qty_lost, item_2 = m.groups()
                vectors.append(encode_relation("query", owner.lower(), item.lower(), 0.0))
    return vectors


def _as_wm_batch(vec_or_props):
    arr = torch.tensor(vec_or_props, dtype=torch.float32)
    if arr.dim() == 2:
        return arr.unsqueeze(0)
    return arr.unsqueeze(0)


def run_pipeline_demo(text: str, train_size: int = 200, val_size: int = 40, epochs: int = 20):
    print("--- LT + spaCy LF + WM injection pipeline ---", flush=True)
    lf_vectors = lf_to_vector_proposals(text)
    if not lf_vectors:
        wm = WorkingMemory()
        wm.facts[null_answer("no valid LF-to-semantic proposal was produced")] = 1
        print("\nNo valid semantic proposals produced from LF. Injected null answer with large penalty:", NULL_ERROR_PENALTY, flush=True)
        return wm

    train_set = build_dataset(train_size)
    val_set = build_dataset(val_size)
    model = LogicTransformer(
        prop_length=PROP_DIM,
        max_props=MAX_PROPS,
        output_dim=8,
        num_rules=12,
        num_premises=2,
        var_slots=3,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for props, target in train_set:
            pred = model(_as_wm_batch(props))
            loss = loss_fn(pred, torch.tensor([target], dtype=torch.float32))
            opt.zero_grad(); loss.backward(); opt.step()
        if epoch % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for props, target in val_set:
                    pred = model(_as_wm_batch(props))
                    val_losses.append(loss_fn(pred, torch.tensor([target], dtype=torch.float32)).item())
                val_loss = sum(val_losses) / len(val_losses)
            print(f"epoch={epoch:03d} val_mse={val_loss:.4f}", flush=True)

    wm = WorkingMemory()
    for vec in lf_vectors:
        with torch.no_grad():
            semantic_vec = model(_as_wm_batch([vec])).squeeze(0).tolist()
        proposal = discretize_proposition(semantic_vec)
        print("LF vector -> semantic proposal:", proposal, flush=True)
        wm.facts[proposition_to_wme(proposal)] = 5

    if not any(f.concept in {"Has", "Lose", "Query", "Quantity"} for f in wm.facts):
        wm.facts[null_answer("LT emitted no valid symbolic WMEs after discretization", penalty=NULL_ERROR_PENALTY)] = 1
        print("\nNo valid symbolic WME survived conversion. Injected null answer with penalty:", NULL_ERROR_PENALTY, flush=True)
    else:
        print("\nInjected semantic WMEs into WM for the symbolic engine:", flush=True)
        for fact, ttl in wm.facts.items():
            print(f"  - {fact.concept}({', '.join(f'{k}={v}' for k, v in fact.to_dict().items())}) ttl={ttl}", flush=True)

    return wm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="Bob has 10 apples and lost 3 apples. How many apples does Bob have now?")
    parser.add_argument("--train-size", type=int, default=200)
    parser.add_argument("--val-size", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    wm = run_pipeline_demo(args.text, train_size=args.train_size, val_size=args.val_size, epochs=args.epochs)
    print("\nFinal WM summary:")
    for fact, ttl in wm.facts.items():
        if fact.concept in {"Has", "Lose", "Query", "NullAnswer"}:
            print(f"  - {fact.concept}({', '.join(f'{k}={v}' for k, v in fact.to_dict().items())}) ttl={ttl}")

    predicate_atoms = wm.to_predicate_atoms()
    print("\nCanonical predicate-logic view (arity <= 2):")
    for atom in predicate_atoms[:12]:
        print(f"  - {atom}")
