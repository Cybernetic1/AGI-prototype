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


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def main() -> None:
    out = REPO_ROOT / "rules" / "reports" / "pot_corpus_smoke.jsonl"
    subprocess.run(
        [
            str(PYTHON),
            str(REPO_ROOT / "rules" / "export_pot_corpus.py"),
            "--limit",
            "10",
            "--output",
            str(out),
        ],
        cwd=str(REPO_ROOT),
        check=True,
    )
    rows = _read_jsonl(out)
    assert rows, "expected non-empty PoT corpus export"
    pos = [r for r in rows if int(r.get("label", -1)) == 1]
    neg = [r for r in rows if int(r.get("label", -1)) == 0]
    assert pos, "expected positive rows"
    assert neg, "expected negative rows"
    assert all("program" in r and isinstance(r["program"], dict) for r in rows), "each row needs a program payload"
    assert all("steps" in r["program"] for r in rows), "each program needs steps"
    print("PoT corpus validation passed.")


if __name__ == "__main__":
    main()
