#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/yky/misc-programs/.venv/bin/python")
if not PYTHON.exists():
    PYTHON = Path(sys.executable)


def main() -> None:
    corpus = REPO_ROOT / "rules" / "reports" / "pot_corpus_smoke.jsonl"
    model = REPO_ROOT / "rules" / "reports" / "pot_proposer_model_smoke.json"
    subprocess.run(
        [
            str(PYTHON),
            str(REPO_ROOT / "rules" / "export_pot_corpus.py"),
            "--limit",
            "12",
            "--output",
            str(corpus),
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )
    subprocess.run(
        [
            str(PYTHON),
            str(REPO_ROOT / "rules" / "train_pot_proposer.py"),
            "--data",
            str(corpus),
            "--output",
            str(model),
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )
    payload = json.loads(model.read_text(encoding="utf-8"))
    metrics = payload.get("metrics") or {}
    assert float(metrics.get("accuracy", 0.0)) >= 0.5, "expected at least basic separability in smoke corpus"
    assert float(metrics.get("f1", 0.0)) >= 0.5, "expected at least basic F1 in smoke corpus"
    print("PoT proposer validation passed.")


if __name__ == "__main__":
    main()
