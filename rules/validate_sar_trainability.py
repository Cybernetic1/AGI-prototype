#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
PREFERRED_PYTHON = Path("/home/yky/misc-programs/.venv/bin/python")
if PREFERRED_PYTHON.exists():
    PYTHON = str(PREFERRED_PYTHON)


def main() -> None:
    data_path = REPO_ROOT / "rules" / "reports" / "sar_training_data_smoke.jsonl"
    cfg_path = REPO_ROOT / "rules" / "reports" / "sar_trained_config_smoke.json"

    subprocess.run(
        [
            PYTHON,
            str(REPO_ROOT / "rules" / "export_sar_training_data.py"),
            "--suite",
            "combined",
            "--limit",
            "6",
            "--output",
            str(data_path),
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )
    subprocess.run(
        [
            PYTHON,
            str(REPO_ROOT / "rules" / "train_sar_config.py"),
            "--data",
            str(data_path),
            "--output",
            str(cfg_path),
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )

    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    config = payload.get("config") or {}
    metrics = payload.get("metrics") or {}
    assert bool(config.get("enabled")) is True, "trained SAR config must be enabled"
    assert float(metrics.get("accuracy", 0.0)) > 0.0, "accuracy must be computed"
    assert float(metrics.get("f1", 0.0)) > 0.0, "f1 must be computed"
    print("SAR trainability validation passed.")


if __name__ == "__main__":
    main()
