from __future__ import annotations

import random
from pathlib import Path

import torch
import torch.nn as nn

from logic_transformer import LogicTransformer

random.seed(7)
torch.manual_seed(7)

NUM_OWNERS = 6
NUM_ITEMS = 5
MAX_PROPS = 3
PROP_DIM = 8


def encode_fact(rel_kind: str, owner_a: int, owner_b: int, item: int, value: float) -> list[float]:
    rel = {"has": 0.0, "ratio": 1.0, "query": 2.0}[rel_kind]
    # Fixed 8-D proposition schema for the ratio-only curriculum:
    # [relation_kind, owner_a, owner_b, item, value, pad1, pad2, pad3]
    return [
        rel,
        float(owner_a) / max(1, NUM_OWNERS - 1),
        float(owner_b) / max(1, NUM_OWNERS - 1),
        float(item) / max(1, NUM_ITEMS - 1),
        float(value),
        0.0,
        0.0,
        0.0,
    ]


def build_ratio_example():
    owner_a = random.randrange(NUM_OWNERS)
    owner_b = random.randrange(NUM_OWNERS)
    item = random.randrange(NUM_ITEMS)
    base = random.randint(2, 40)
    factor = random.choice([0.5, 1.5, 2.0, 3.0, 4.0])
    target = base * factor

    props = [
        encode_fact("has", owner_b, owner_b, item, base),
        encode_fact("ratio", owner_a, owner_b, item, factor),
        encode_fact("query", owner_a, owner_b, item, 0.0),
    ]
    return props, target


def build_dataset(n: int = 2000):
    dataset = []
    for _ in range(n):
        dataset.append(build_ratio_example())
    return dataset


def train_demo():
    print("--- Ratio-only LT diagnostic ---")
    train_set = build_dataset(400)
    val_set = build_dataset(100)

    model = LogicTransformer(
        prop_length=PROP_DIM,
        max_props=MAX_PROPS,
        output_dim=1,
        num_rules=12,
        num_premises=2,
        var_slots=3,
    )
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    for epoch in range(120):
        model.train()
        total_loss = 0.0
        for props, target in train_set:
            wm = torch.tensor([props], dtype=torch.float32)
            y = torch.tensor([[float(target)]], dtype=torch.float32)
            opt.zero_grad()
            pred = model(wm)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg_train = total_loss / len(train_set)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for props, target in val_set:
                wm = torch.tensor([props], dtype=torch.float32)
                y = torch.tensor([[float(target)]], dtype=torch.float32)
                pred = model(wm)
                val_loss += loss_fn(pred, y).item()
        avg_val = val_loss / len(val_set)
        if avg_val < best_val:
            best_val = avg_val
        if epoch % 20 == 0 or epoch == 119:
            print(f"epoch={epoch:03d} train_mse={avg_train:.4f} val_mse={avg_val:.4f}")

    model.eval()
    rules_path = Path(__file__).with_name("ratio_rules.txt")
    saved_path = model.save_rules(rules_path)
    print(f"\nSaved learned rules to {saved_path}")
    print("\nLearned rule summary:")
    model.print_rules()

    sample_props, sample_target = random.choice(val_set)
    with torch.no_grad():
        pred = model(torch.tensor([sample_props], dtype=torch.float32)).item()
    print("\nSample hold-out case:")
    print("propositions:", sample_props)
    print(f"pred={pred:.4f} target={sample_target:.4f} abs_err={abs(pred-sample_target):.4f}")

    errs = []
    with torch.no_grad():
        for props, target in val_set:
            pred = model(torch.tensor([props], dtype=torch.float32)).item()
            errs.append(abs(pred - target))
    mae = sum(errs) / len(errs)
    print(f"mean_abs_error={mae:.4f}")
    print("\nConclusion: a single-family ratio curriculum is the minimal clean baseline. If the original LT model can solve this clean task, then the task is not fundamentally impossible; if it fails, the issue is deeper than mixed-family arithmetic complexity and we should focus on the rule grammar rather than broad sweep diversity.")


if __name__ == "__main__":
    train_demo()
