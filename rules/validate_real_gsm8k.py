#!/usr/bin/env python3
"""Lightweight validation against real GSM8K examples.

This script is intended to stay cheap and periodic: it samples a fixed set of
real examples from the local GSM8K test export, runs the current cognitive loop,
and reports more than a single solve rate. The goal is to expose where the model
fails (ratios, remaining quantities, percent updates, etc.) so progress can be
tracked across iterations without spending time on the full suite.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from cognitive_loop import benchmark_single_problem, _extract_numeric_value


def discover_dataset() -> Path:
    candidates = [
        BASE_DIR / "logic_transformer" / "data" / "gsm8k_test_lt.jsonl",
        BASE_DIR / "logic_transformer" / "data" / "gsm8k_test.jsonl",
        BASE_DIR / "logic_transformer" / "data" / "gsm8k" / "main_test.jsonl",
        BASE_DIR / "logic_transformer" / "data" / "gsm8k_tiny.jsonl",
        Path("/home/yky/misc-programs/AGI-experiments") / "gsm8k-dln-demo" / "data" / "gsm8k" / "main_test.jsonl",
        Path("/home/yky/misc-programs/AGI-experiments") / "data" / "processed" / "gsm8k_test.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not locate a real GSM8K test export. Expected one of:\n"
        + "\n".join(str(p) for p in candidates)
    )


def load_examples(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue
            question = ex.get("question") or ex.get("text") or ex.get("prompt")
            answer = ex.get("answer")
            if question and answer:
                rows.append({
                    "question": str(question),
                    "answer": str(answer),
                    "id": ex.get("id", "unknown"),
                })
    return rows


def question_tags(question: str) -> list[str]:
    lower = question.lower()
    tags: list[str] = []
    if any(word in lower for word in ["percent", "%", "discount", "increase", "decrease", "more than", "less than"]):
        tags.append("percent")
    if any(word in lower for word in ["times as many", "twice", "double", "ratio", "half as many", "three times"]):
        tags.append("ratio")
    if any(word in lower for word in ["remaining", "left", "leftover", "after", "then"]):
        tags.append("remaining")
    if any(word in lower for word in ["per hour", "per minute", "for 4 hours", "earns", "costs", "sold at", "each"]):
        tags.append("rate_time")
    if any(word in lower for word in ["altogether", "in all", "total", "combined", "together"]):
        tags.append("total")
    if any(word in lower for word in ["how many more", "how much more", "difference", "less than", "fewer than"]):
        tags.append("comparison")
    if any(word in lower for word in ["profit", "sold", "buys", "bought", "sells", "price", "cost", "dollar"]):
        tags.append("commerce")
    if not tags:
        tags.append("general")
    return tags


def answer_bucket(answer_value: float | None) -> str:
    if answer_value is None:
        return "unknown"
    if abs(answer_value) < 10:
        return "small"
    if abs(answer_value) < 100:
        return "medium"
    if abs(answer_value) < 1000:
        return "large"
    return "very_large"


def summarize_example(index: int, example: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    expected = _extract_numeric_value(example["answer"])
    belief_values = result.get("belief_values") or []
    numeric_values = []
    for value in belief_values:
        parsed = _extract_numeric_value(value)
        if parsed is not None:
            numeric_values.append(float(parsed))
    tags = question_tags(example["question"])
    answer_generated = bool(result.get("answer_generated"))
    wme_signal = float(result.get("intermediate_wme_signal") or 0.0)
    return {
        "index": index,
        "id": example.get("id"),
        "tags": tags,
        "question": example["question"],
        "expected": expected,
        "belief_values": numeric_values,
        "solved": bool(result.get("solved")),
        "answer_generated": answer_generated,
        "intermediate_wme_signal": wme_signal,
        "numeric_fact_count": int(result.get("numeric_fact_count") or 0),
        "max_activation": float(result.get("max_activation") or 0.0),
        "answer_bucket": answer_bucket(expected),
    }


def validate_real_gsm8k(sample_size: int = 20, seed: int = 0, max_cycles: int = 4, quiet: bool = False, json_out: str | None = None) -> dict[str, Any]:
    dataset_path = discover_dataset()
    examples = load_examples(dataset_path)
    if not examples:
        raise ValueError(f"No examples loaded from dataset: {dataset_path}")
    if sample_size <= 0:
        sampled = examples
    else:
        sample_size = min(sample_size, len(examples))
        rng = random.Random(seed)
        sampled = rng.sample(examples, sample_size)

    rows: list[dict[str, Any]] = []
    solved = 0
    tag_counter: Counter[str] = Counter()
    tag_solved: Counter[str] = Counter()
    tag_generated: Counter[str] = Counter()
    bucket_counter: Counter[str] = Counter()
    bucket_solved: Counter[str] = Counter()
    bucket_generated: Counter[str] = Counter()

    for idx, example in enumerate(sampled, start=1):
        result = benchmark_single_problem(
            example["question"],
            example["answer"],
            max_cycles=max_cycles,
            quiet=True,
        )
        row = summarize_example(idx, example, result)
        rows.append(row)
        if row["solved"]:
            solved += 1
        for tag in row["tags"]:
            tag_counter[tag] += 1
            if row["answer_generated"]:
                tag_generated[tag] += 1
            if row["solved"]:
                tag_solved[tag] += 1
        bucket = row["answer_bucket"]
        bucket_counter[bucket] += 1
        if row["answer_generated"]:
            bucket_generated[bucket] += 1
        if row["solved"]:
            bucket_solved[bucket] += 1

        if not quiet:
            tag_text = ", ".join(row["tags"])
            belief_text = ", ".join(f"{v:.6g}" for v in row["belief_values"]) if row["belief_values"] else "[]"
            print(f"[{idx:02d}] tags={tag_text:20s} expected={row['expected']} beliefs=[{belief_text}] generated={row['answer_generated']} wme_signal={row['intermediate_wme_signal']:.2f} solved={row['solved']}")

    summary = {
        "dataset": str(dataset_path),
        "sample_size": len(rows),
        "seed": seed,
        "max_cycles": max_cycles,
        "solved": solved,
        "unsolved": len(rows) - solved,
        "solve_rate": (solved / len(rows)) if rows else 0.0,
        "answer_generation_rate": sum(1 for r in rows if r["answer_generated"]) / len(rows) if rows else 0.0,
        "by_tag": {
            k: {
                "count": v,
                "solved": tag_solved.get(k, 0),
                "answer_generated": tag_generated.get(k, 0),
            }
            for k, v in sorted(tag_counter.items())
        },
        "by_answer_bucket": {
            k: {
                "count": v,
                "solved": bucket_solved.get(k, 0),
                "answer_generated": bucket_generated.get(k, 0),
            }
            for k, v in sorted(bucket_counter.items())
        },
        "results": rows,
    }

    if json_out:
        out_path = Path(json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nReal GSM8K validation summary")
    print(f"dataset={dataset_path}")
    print(f"sample_size={len(rows)} solved={solved}/{len(rows)} solve_rate={summary['solve_rate']:.2%}")
    print(f"answer_generation_rate={summary['answer_generation_rate']:.2%}")
    print("by_tag:")
    for tag, stats in summary["by_tag"].items():
        count = stats["count"]
        solved_count = stats["solved"]
        generated_count = stats["answer_generated"]
        print(f"  - {tag}: count={count}, generated={generated_count}, solved={solved_count}, solve_rate={((solved_count / count) if count else 0.0):.2%}")
    print("by_answer_bucket:")
    for bucket, stats in summary["by_answer_bucket"].items():
        count = stats["count"]
        solved_count = stats["solved"]
        generated_count = stats["answer_generated"]
        print(f"  - {bucket}: count={count}, generated={generated_count}, solved={solved_count}, solve_rate={((solved_count / count) if count else 0.0):.2%}")

    if rows:
        avg_signal = sum(r["intermediate_wme_signal"] for r in rows) / len(rows)
        max_signal = max(r["intermediate_wme_signal"] for r in rows)
        generated_rows = [r for r in rows if r["answer_generated"]]
        if generated_rows:
            avg_generated_signal = sum(r["intermediate_wme_signal"] for r in generated_rows) / len(generated_rows)
            max_generated_signal = max(r["intermediate_wme_signal"] for r in generated_rows)
            print(f"\nHeuristic WME signal: avg_total={avg_signal:.2f}, max_total={max_signal:.2f}, avg_generated={avg_generated_signal:.2f}, max_generated={max_generated_signal:.2f}, generated_count={len(generated_rows)}")
        else:
            print(f"\nHeuristic WME signal: avg_total={avg_signal:.2f}, max_total={max_signal:.2f}, generated_count=0")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lightweight validation pass against real GSM8K examples.")
    parser.add_argument("--sample-size", type=int, default=20, help="How many real examples to sample from the GSM8K test split.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling the validation set.")
    parser.add_argument("--cycles", type=int, default=4, help="Maximum cognitive loop cycles per example.")
    parser.add_argument("--json-out", default="", help="Optional path to write JSON summary for tracking.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-example output; keep only aggregate summaries.")
    args = parser.parse_args()
    validate_real_gsm8k(
        sample_size=args.sample_size,
        seed=args.seed,
        max_cycles=args.cycles,
        quiet=args.quiet,
        json_out=args.json_out,
    )


if __name__ == "__main__":
    main()
