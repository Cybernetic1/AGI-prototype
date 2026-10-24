#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Print saved LT rules without retraining.")
    parser.add_argument("path", nargs="?", default=str(Path(__file__).with_name("ratio_rules.txt")))
    args = parser.parse_args()
    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(f"No saved rules found at {path}. Run ratio_single_family_lt_demo.py first.")
    print(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
