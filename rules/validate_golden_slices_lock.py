#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate GSM golden slice lock against widening contract.")
    parser.add_argument("--lock", default="rules/bootstrap_golden_slices_v1.json")
    parser.add_argument("--contract", default="")
    parser.add_argument("--output", default="rules/reports/golden_slices_lock_eval_v1.json")
    args = parser.parse_args()

    lock = _load_json(Path(args.lock))
    contract_path = str(args.contract or lock.get("contract_path") or "").strip()
    if not contract_path:
        raise ValueError("contract path is required (via --contract or lock.contract_path)")
    contract = _load_json(Path(contract_path))

    slices = contract.get("slices")
    if not isinstance(slices, list):
        raise ValueError("contract.slices must be a list")
    by_id = {str(s.get("id")): s for s in slices if isinstance(s, dict) and s.get("id")}

    missing = []
    golden_families = []
    for sid in lock.get("golden_slices", []):
        row = by_id.get(str(sid))
        if row is None:
            missing.append(str(sid))
            continue
        fams = [str(f) for f in (row.get("families") or [])]
        golden_families.append({"slice_id": str(sid), "suite": row.get("suite"), "family_count": len(fams), "families": fams})

    ok = len(missing) == 0 and bool(golden_families)
    payload = {
        "lock_id": lock.get("lock_id"),
        "lock_path": args.lock,
        "contract_path": contract_path,
        "ok": ok,
        "missing_slices": missing,
        "golden_slices": golden_families,
        "seed_set": list(lock.get("seeds") or []),
        "profiles": list(lock.get("profiles") or []),
        "promotion_policy": dict(lock.get("promotion_policy") or {}),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved golden-lock validation to {out}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
