from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ilp_learner.metarule_loop import export_synthetic_training_set


def main() -> int:
    parser = argparse.ArgumentParser(description="Export synthetic metarule training data with positives and negatives.")
    parser.add_argument("--out", default=str(REPO_ROOT / "ilp_learner" / "synthetic_metarule_train.jsonl"))
    args = parser.parse_args()

    count = export_synthetic_training_set(args.out)
    print(f"Wrote {count} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
