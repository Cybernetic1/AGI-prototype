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
    out = REPO_ROOT / "rules" / "reports" / "tinystories_symbolic_compact_smoke.jsonl"
    subprocess.run(
        [str(PYTHON), str(REPO_ROOT / "rules" / "export_tinystories_symbolic_set.py"), "--output", str(out)],
        check=True,
        cwd=str(REPO_ROOT),
    )
    rows = _read_jsonl(out)
    assert rows, "expected non-empty TinyStories symbolic set"
    required = {"family", "question", "expected", "symbolic_targets"}
    for row in rows:
        missing = required.difference(row.keys())
        assert not missing, f"missing fields: {missing}"
        targets = row.get("symbolic_targets") or {}
        assert isinstance(targets.get("states"), list) and targets.get("states"), "states must be non-empty"
        query = targets.get("query") or {}
        assert query.get("goal"), "query.goal required"
    print("TinyStories symbolic curation validation passed.")


if __name__ == "__main__":
    main()
