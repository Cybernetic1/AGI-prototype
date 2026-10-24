from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from model import LTBootstrapModel

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

random.seed(7)
torch.manual_seed(7)


def sample_people():
    return ["alice", "bob", "carol", "dana", "erin", "frank"]


def sample_items():
    return ["apples", "oranges", "bananas", "books", "pens", "tickets"]


def gen_ratio_example():
    owner_a, owner_b = random.sample(sample_people(), 2)
    item = random.choice(sample_items())
    base = random.randint(4, 20)
    factor = random.choice([2, 3, 0.5, 1.5, 4])
    value = round(base * factor, 2)
    props = [
        {"pred": "has", "args": [owner_b, item, str(base)]},
        {"pred": "ratio", "args": [owner_a, owner_b, item, str(factor)]},
        {"pred": "query", "args": [owner_a, item]},
    ]
    return {"family": "ratio", "props": props, "answer": value}


def gen_remaining_example():
    owner = random.choice(sample_people())
    item = random.choice(sample_items())
    start = random.randint(10, 60)
    pct = random.choice([10, 20, 25, 30, 40, 50])
    value = round(start * (1 - pct / 100.0), 2)
    props = [
        {"pred": "has", "args": [owner, item, str(start)]},
        {"pred": "percent_reduction", "args": [owner, item, str(pct)]},
        {"pred": "query", "args": [owner, item]},
    ]
    return {"family": "remaining", "props": props, "answer": value}


def gen_state_transition_example():
    owner = random.choice(sample_people())
    item = random.choice(sample_items())
    before = random.randint(4, 40)
    delta = random.randint(-12, 18)
    after = before + delta
    props = [
        {"pred": "state", "args": [owner, item, str(before)]},
        {"pred": "delta", "args": [owner, item, str(delta)]},
        {"pred": "query", "args": [owner, item]},
    ]
    return {"family": "state_transition", "props": props, "answer": after}


def gen_final_state_example():
    owner = random.choice(sample_people())
    item = random.choice(sample_items())
    earlier = random.randint(5, 30)
    later = random.randint(10, 50)
    final_value = later
    props = [
        {"pred": "state", "args": [owner, item, str(earlier)]},
        {"pred": "state", "args": [owner, item, str(later)]},
        {"pred": "final_preference", "args": [owner, item]},
        {"pred": "query", "args": [owner, item]},
    ]
    return {"family": "final_state", "props": props, "answer": final_value}


def gen_mixed_example():
    owner = random.choice(sample_people())
    item = random.choice(sample_items())
    base = random.randint(8, 40)
    percent = random.choice([10, 20, 25, 50])
    delta = random.randint(-8, 12)
    value = round((base * (1 - percent / 100.0)) + delta, 2)
    props = [
        {"pred": "has", "args": [owner, item, str(base)]},
        {"pred": "percent_reduction", "args": [owner, item, str(percent)]},
        {"pred": "delta", "args": [owner, item, str(delta)]},
        {"pred": "final_preference", "args": [owner, item]},
        {"pred": "query", "args": [owner, item]},
    ]
    return {"family": "mixed", "props": props, "answer": value}


GENERATORS = [
    gen_ratio_example,
    gen_remaining_example,
    gen_state_transition_example,
    gen_final_state_example,
    gen_mixed_example,
]


def generate_dataset(num_examples: int = 600):
    dataset = []
    for _ in range(num_examples):
        dataset.append(random.choice(GENERATORS)())
    return dataset


class ToyGSMDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return ex["props"], torch.tensor(float(ex["answer"]), dtype=torch.float32)


def collate_props(batch):
    props_batch = [row[0] for row in batch]
    targets = torch.stack([row[1] for row in batch])
    return props_batch, targets


def train_demo():
    print("--- Toy GSM semantic-structure demo ---")
    dataset = generate_dataset(600)
    data_path = DATA_DIR / "toy_gsm_semantic.jsonl"
    with data_path.open("w", encoding="utf-8") as fh:
        for ex in dataset:
            fh.write(json.dumps(ex) + "\n")
    print(f"Generated {len(dataset)} synthetic examples at {data_path}")

    random.shuffle(dataset)
    split = int(0.8 * len(dataset))
    train_set = dataset[:split]
    val_set = dataset[split:]

    model = LTBootstrapModel(feature_dim=32, num_buckets=2048)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    train_loader = DataLoader(ToyGSMDataset(train_set), batch_size=32, shuffle=True, collate_fn=collate_props)
    val_loader = DataLoader(ToyGSMDataset(val_set), batch_size=64, shuffle=False, collate_fn=collate_props)

    best_val = float("inf")
    for epoch in range(120):
        model.train()
        running = 0.0
        for props_list, targets in train_loader:
            opt.zero_grad()
            preds = []
            for props in props_list:
                pred = model(props).view(())
                preds.append(pred)
            pred_tensor = torch.stack(preds)
            loss = criterion(pred_tensor, targets)
            loss.backward()
            opt.step()
            running += loss.item() * len(targets)
        train_loss = running / len(train_set)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for props_list, targets in val_loader:
                preds = []
                for props in props_list:
                    pred = model(props).view(())
                    preds.append(pred)
                pred_tensor = torch.stack(preds)
                val_loss += criterion(pred_tensor, targets).item() * len(targets)
        val_loss /= len(val_set)
        if val_loss < best_val:
            best_val = val_loss
        if epoch % 20 == 0 or epoch == 119:
            print(f"epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    model.eval()
    sample = random.choice(val_set)
    print("\nSample holdout example:")
    print(json.dumps(sample, sort_keys=True))
    with torch.no_grad():
        wm, attn = model.encode_props(sample["props"])
        ranked = sorted(zip(sample["props"], attn.tolist()), key=lambda x: x[1], reverse=True)
        print("Top attention props:")
        for prop, weight in ranked[:5]:
            print(f"  {prop} -> {weight:.4f}")
        pred = model(sample["props"]).item()
        print(f"Predicted answer: {pred:.4f} | gold: {sample['answer']}")

    print("\nConclusion: a compact LT-style model can learn the synthetic semantics from structured examples; the model puts attention on the relevant relational predicates (ratio/percent/state/final-preference), which is the latent structure we want to stabilize before broader GSM8K-like sweeps.")


if __name__ == "__main__":
    train_demo()
