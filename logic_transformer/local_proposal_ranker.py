from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_PATH = REPO_ROOT / "ilp_learner" / "synthetic_local_train.jsonl"
DEFAULT_HOLDOUT_PATH = REPO_ROOT / "logic_transformer" / "proposal_holdout.jsonl"
DEFAULT_MODEL_PATH = REPO_ROOT / "logic_transformer" / "models" / "local_proposal_ranker.pt"
DOMAIN_KEYWORDS = {
    "ratio": {"ratio", "times", "twice", "half", "double", "triple", "factor"},
    "remaining": {"remaining", "left", "leftover", "after", "gave", "away", "spent", "used"},
    "state": {"had", "bought", "bought", "gave", "donated", "spoil", "remains", "now", "final"},
    "percent": {"percent", "%", "of", "remaining", "donated", "consumed"},
    "buy": {"bought", "buy", "more", "purchase", "purchased"},
    "donation": {"donated", "donation", "gifted"},
    "owner": {"owner", "has", "owns", "does", "have"},
    "item": {"apples", "oranges", "bananas"},
}


def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[%_.,;:!?()\[\]{}\-+/=]", " ", text)
    return [tok for tok in text.split() if tok]


class ProposalRanker(nn.Module):
    def __init__(self, vocab: Sequence[str]):
        super().__init__()
        self.vocab = list(vocab)
        self.index = {word: i for i, word in enumerate(self.vocab)}
        self.linear = nn.Linear(len(self.vocab), 1, bias=True)

    def features(self, text: str) -> torch.Tensor:
        vec = torch.zeros(len(self.vocab), dtype=torch.float32)
        for token in tokenize(text):
            idx = self.index.get(token)
            if idx is not None:
                vec[idx] += 1.0
        return vec

    def forward(self, text: str) -> torch.Tensor:
        return torch.sigmoid(self.linear(self.features(text)))

    @torch.no_grad()
    def score(self, question: str, proposal: str) -> float:
        score = float(self.forward(f"{question} {proposal}").squeeze().item())
        return min(max(score, 0.0), 1.0)

    @classmethod
    def load_or_train(
        cls,
        train_path: Path = DEFAULT_TRAIN_PATH,
        model_path: Path = DEFAULT_MODEL_PATH,
        epochs: int = 400,
        lr: float = 0.05,
    ) -> "ProposalRanker":
        if model_path.exists():
            return load_local_proposal_model(model_path)
        return train_local_proposal_model(train_path, model_path, epochs, lr)


def build_vocab(records: Iterable[Dict]) -> List[str]:
    vocab: set[str] = set()
    for record in records:
        text = f"{record.get('question','')} {record.get('label','')} {record.get('owner','')} {record.get('item','')}"
        vocab.update(tokenize(text))
        for keywords in DOMAIN_KEYWORDS.values():
            vocab.update(keywords)
    return sorted(vocab)


def generate_positive_candidate(record: Dict) -> str:
    label = str(record.get("label", "unknown"))
    owner = str(record.get("owner", "A"))
    item = str(record.get("item", "apples"))
    answer = record.get("answer", 0)
    if label == "ratio_owner_binding":
        base_owner = str(record.get("base_owner", "B" if owner != "B" else "C"))
        return f"ratio(owner={owner}, base_owner={base_owner}, item={item}, factor=3)"
    if label == "percent_of_remaining":
        return f"percent(owner={owner}, item={item}, pct=0.5); remaining(owner={owner}, item={item}, qty={answer})"
    if label == "buy_more_state_update":
        return f"final_state(owner={owner}, item={item}, qty={answer})"
    if label == "final_state_precedence":
        return f"final_state(owner={owner}, item={item}, qty={answer})"
    if label == "donation_after_buy_more":
        return f"final_state(owner={owner}, item={item}, qty={answer})"
    if label == "remaining_after_subtraction":
        return f"remaining(owner={owner}, item={item}, qty={answer})"
    return f"final_state(owner={owner}, item={item}, qty={answer})"


def generate_negative_candidate(record: Dict) -> str:
    label = str(record.get("label", "unknown"))
    owner = str(record.get("owner", "A"))
    item = str(record.get("item", "apples"))
    base_owner = str(record.get("base_owner", "B" if owner != "B" else "C"))
    if label == "ratio_owner_binding":
        return f"ratio(owner={owner}, base_owner={owner}, item={item}, factor=2)"
    if label == "percent_of_remaining":
        return f"ratio(owner={owner}, base_owner={base_owner}, item={item}, factor=2)"
    if label == "buy_more_state_update":
        return f"ratio(owner={owner}, base_owner={base_owner}, item={item}, factor=1)"
    if label == "final_state_precedence":
        return f"remaining(owner={owner}, item={item}, qty=0)"
    if label == "donation_after_buy_more":
        return f"ratio(owner={owner}, base_owner={base_owner}, item={item}, factor=4)"
    return f"ratio(owner={owner}, base_owner={base_owner}, item={item}, factor=2)"


def load_records(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    return records


def make_training_pairs(records: Sequence[Dict]) -> Tuple[List[str], List[int]]:
    texts: List[str] = []
    labels: List[int] = []
    for record in records:
        question = str(record.get("question", ""))
        pos = generate_positive_candidate(record)
        neg = generate_negative_candidate(record)
        texts.append(f"{question} {pos}")
        labels.append(1)
        texts.append(f"{question} {neg}")
        labels.append(0)
    return texts, labels


def train_local_proposal_model(
    train_path: Path = DEFAULT_TRAIN_PATH,
    output_path: Path = DEFAULT_MODEL_PATH,
    epochs: int = 400,
    lr: float = 0.05,
) -> ProposalRanker:
    records = load_records(train_path)
    vocab = build_vocab(records)
    texts, labels = make_training_pairs(records)

    model = ProposalRanker(vocab)
    torch.manual_seed(0)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    features = []
    targets = []
    for text, label in zip(texts, labels):
        features.append(model.features(text))
        targets.append(float(label))

    X = torch.stack(features)
    y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model.linear(X)
        loss = criterion(torch.sigmoid(logits), y)
        loss.backward()
        optimizer.step()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"vocab": model.vocab, "state_dict": model.state_dict()}, output_path)
    return model


def load_local_proposal_model(model_path: Path = DEFAULT_MODEL_PATH) -> ProposalRanker:
    if not model_path.exists():
        raise FileNotFoundError(f"No model checkpoint found at {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu")
    model = ProposalRanker(checkpoint["vocab"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def evaluate_model(model: ProposalRanker, records: Sequence[Dict]) -> float:
    texts, labels = make_training_pairs(records)
    correct = 0
    for text, label in zip(texts, labels):
        pred = 1 if model.score("", text) >= 0.5 else 0
        correct += int(pred == label)
    return correct / max(len(labels), 1)


def evaluate_holdout(model: ProposalRanker, holdout_path: Path = DEFAULT_HOLDOUT_PATH) -> float:
    if not holdout_path.exists():
        raise FileNotFoundError(f"Missing holdout dataset: {holdout_path}")
    records = load_records(holdout_path)
    return evaluate_model(model, records)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train/score a lightweight semantic-proposal ranker.")
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--holdout-path", type=Path, default=DEFAULT_HOLDOUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.05)
    args = parser.parse_args()

    model = train_local_proposal_model(
        train_path=args.train_path,
        output_path=args.output,
        epochs=args.epochs,
        lr=args.lr,
    )
    records = load_records(args.train_path)
    accuracy = evaluate_model(model, records)
    holdout_accuracy = evaluate_holdout(model, args.holdout_path)
    print(f"proposal_ranker_accuracy={accuracy:.4f}")
    print(f"proposal_ranker_holdout_accuracy={holdout_accuracy:.4f}")
    print(f"saved_model={args.output}")
