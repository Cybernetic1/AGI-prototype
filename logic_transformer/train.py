"""Unified canonical LT trainer for semantic continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from model import LTSemanticContinuationModel
from preprocess_gsm8k import prepare_tinystories_continuation_file


TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+|[!?.,]")


def _dataset_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
    return rows


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())


def build_vocabulary(rows: list[dict]) -> dict[str, int]:
    vocabulary = {"<pad>": 0, "<unk>": 1}
    for row in rows:
        for field in ("context", "candidate_a", "candidate_b"):
            for token in tokenize(row.get(field, "")):
                if token not in vocabulary:
                    vocabulary[token] = len(vocabulary)
    return vocabulary


def encode_text(text: str, vocabulary: dict[str, int], length: int) -> list[int]:
    ids = [vocabulary.get(token, vocabulary["<unk>"]) for token in tokenize(text)]
    ids = ids[:length]
    return ids + [0] * (length - len(ids))


def _save_checkpoint(
    path: Path,
    model: LTSemanticContinuationModel,
    optimizer: torch.optim.Optimizer,
    vocabulary: dict[str, int],
    *,
    epoch: int,
    data_path: Path,
    metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "vocabulary": vocabulary,
        "feature_dim": model.feature_dim,
        "symbol_dim": model.symbol_dim,
        "num_rules": model.num_rules,
        "max_sentence_tokens": model.max_sentence_tokens,
        "completed_epoch": int(epoch),
        "data_path": str(data_path.resolve()),
        "data_sha256": _dataset_fingerprint(data_path),
        "last_metrics": metrics,
        "task": "semantic-continuation",
        "model": "LTSemanticContinuationModel",
        "saved_at": time.time(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    print(f"Saved checkpoint to {path} (completed_epoch={epoch})", flush=True)


def lexical_continuation_accuracy(rows: list[dict]) -> tuple[float, int]:
    correct = 0
    scored = 0
    for row in rows:
        context = set(tokenize(row.get("context", "")))
        score_a = len(context & set(tokenize(row.get("candidate_a", ""))))
        score_b = len(context & set(tokenize(row.get("candidate_b", ""))))
        if score_a == score_b:
            continue
        predicted = 0 if score_a > score_b else 1
        correct += int(predicted == int(row["label"]))
        scored += 1
    return correct / max(1, scored), scored


def _batch(rows: list[dict], vocabulary: dict[str, int], length: int):
    context = torch.tensor(
        [encode_text(row["context"], vocabulary, length) for row in rows],
        dtype=torch.long,
    )
    candidate_a = torch.tensor(
        [encode_text(row["candidate_a"], vocabulary, length) for row in rows],
        dtype=torch.long,
    )
    candidate_b = torch.tensor(
        [encode_text(row["candidate_b"], vocabulary, length) for row in rows],
        dtype=torch.long,
    )
    labels = torch.tensor([int(row["label"]) for row in rows], dtype=torch.long)
    return context, candidate_a, candidate_b, labels


def evaluate(model, rows, vocabulary, batch_size):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            context, candidate_a, candidate_b, labels = _batch(
                rows[start : start + batch_size],
                vocabulary,
                model.max_sentence_tokens,
            )
            logits = model(context, candidate_a, candidate_b)
            total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
            total_correct += int((logits.argmax(dim=-1) == labels).sum().item())
            total += len(labels)
    return total_loss / max(1, total), total_correct / max(1, total)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the canonical vectorized LT semantic-continuation model")
    parser.add_argument("--task", choices=["semantic-continuation"], default="semantic-continuation")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--tinystories-source", default=None)
    parser.add_argument("--prepared-output", default="runtime/tinystories/semantic_continuation_10k.jsonl")
    parser.add_argument("--max-stories", type=int, default=10_000)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20, help="Additional epochs to run.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-rules", type=int, default=16)
    parser.add_argument("--symbol-dim", type=int, default=64)
    parser.add_argument("--checkpoint", default="checkpoints/lt_semantic_continuation.pt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-data-change", action="store_true")
    parser.add_argument("--max-minutes", type=float, default=0.0)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--holdout-ratio", type=float, default=0.1)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    if args.tinystories_source:
        data_path = Path(args.prepared_output)
        pairs = prepare_tinystories_continuation_file(
            args.tinystories_source,
            data_path,
            max_stories=args.max_stories,
            validation_ratio=args.holdout_ratio,
            seed=args.seed,
        )
        print(f"Prepared {len(pairs)} pairs into {data_path}", flush=True)
    elif args.data_path:
        data_path = Path(args.data_path)
    else:
        raise ValueError("provide --data-path or --tinystories-source")

    rows = load_rows(data_path)
    train_rows = [row for row in rows if row.get("split") == "train"]
    validation_rows = [row for row in rows if row.get("split") == "validation"]
    if not train_rows or not validation_rows:
        raise ValueError("data must contain train and validation splits")
    if args.max_pairs > 0:
        train_rows = train_rows[: max(1, int(args.max_pairs * 0.9))]
        validation_rows = validation_rows[: max(1, int(args.max_pairs * 0.1))]

    vocabulary = build_vocabulary(rows)
    model = LTSemanticContinuationModel(
        len(vocabulary),
        symbol_dim=args.symbol_dim,
        num_rules=args.num_rules,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint_path = Path(args.checkpoint)
    completed_epoch = 0

    if args.resume:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if checkpoint.get("model") != "LTSemanticContinuationModel":
            raise ValueError("checkpoint is not a canonical LT semantic-continuation model")
        if checkpoint.get("vocabulary") != vocabulary:
            raise ValueError("checkpoint vocabulary differs; retrain with the same prepared data")
        if checkpoint.get("num_rules") != model.num_rules:
            raise ValueError("checkpoint rule count differs from --num-rules")
        if checkpoint.get("symbol_dim") != model.symbol_dim:
            raise ValueError("checkpoint symbol dimension differs from --symbol-dim")
        if checkpoint.get("data_sha256") != _dataset_fingerprint(data_path) and not args.allow_data_change:
            raise ValueError("dataset changed since checkpoint; use --allow-data-change deliberately")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        completed_epoch = int(checkpoint.get("completed_epoch", 0))
        print(f"Resumed at completed_epoch={completed_epoch}", flush=True)

    lexical_accuracy, lexical_count = lexical_continuation_accuracy(validation_rows)
    print(
        f"Task: semantic-continuation | model_parameters={sum(p.numel() for p in model.parameters()):,} "
        f"| rules={model.num_rules} | vocab={len(vocabulary)} "
        f"| train={len(train_rows)} | validation={len(validation_rows)}",
        flush=True,
    )
    print(
        f"Random baseline=0.500 | lexical-overlap baseline={lexical_accuracy:.3f} "
        f"({lexical_count} non-tied pairs)",
        flush=True,
    )

    started_at = time.monotonic()
    for epoch in range(completed_epoch, completed_epoch + args.epochs):
        random.shuffle(train_rows)
        model.train()
        running_loss = 0.0
        seen = 0
        timed_out = False
        for start in range(0, len(train_rows), args.batch_size):
            if args.max_minutes > 0 and (time.monotonic() - started_at) >= args.max_minutes * 60:
                timed_out = True
                print(f"Time limit reached after {seen} examples in epoch {epoch + 1}.", flush=True)
                break
            batch_rows = train_rows[start : start + args.batch_size]
            context, candidate_a, candidate_b, labels = _batch(
                batch_rows, vocabulary, model.max_sentence_tokens
            )
            logits = model(context, candidate_a, candidate_b)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_loss += loss.item() * len(batch_rows)
            seen += len(batch_rows)
        if timed_out:
            break
        train_loss = running_loss / max(1, seen)
        eval_loss, pair_accuracy = evaluate(
            model, validation_rows, vocabulary, args.batch_size
        )
        metrics = {
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "pair_accuracy": pair_accuracy,
            "elapsed_sec": time.monotonic() - started_at,
        }
        print(
            f"Epoch {epoch + 1:02d} | train_loss={train_loss:.4f} "
            f"| eval_loss={eval_loss:.4f} | pair_acc={pair_accuracy:.3f}",
            flush=True,
        )
        _save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            vocabulary,
            epoch=epoch + 1,
            data_path=data_path,
            metrics=metrics,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
