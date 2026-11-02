from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_utility_state(path: str | Path) -> dict[str, Any]:
    src = Path(path)
    if not src.exists():
        return {
            "meta": {"total_updates": 0, "updated_at": _utc_now()},
            "rules": {},
        }
    payload = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {"meta": {"total_updates": 0, "updated_at": _utc_now()}, "rules": {}}
    payload.setdefault("meta", {})
    payload.setdefault("rules", {})
    payload["meta"].setdefault("total_updates", 0)
    payload["meta"].setdefault("updated_at", _utc_now())
    return payload


def save_utility_state(path: str | Path, state: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    state.setdefault("meta", {})
    state["meta"]["updated_at"] = _utc_now()
    out.write_text(json.dumps(state, indent=2), encoding="utf-8")


def utility_components(state: dict[str, Any], signature: str) -> dict[str, float]:
    rules = state.get("rules", {}) if isinstance(state.get("rules"), dict) else {}
    row = rules.get(signature, {}) if isinstance(rules.get(signature), dict) else {}
    count = float(row.get("count", 0.0) or 0.0)
    total_reward = float(row.get("total_reward", 0.0) or 0.0)
    value = (total_reward / count) if count > 0 else 0.0
    total_updates = float((state.get("meta", {}) or {}).get("total_updates", 0.0) or 0.0)
    exploration = math.sqrt(max(0.0, math.log(total_updates + 1.0) / (count + 1.0)))
    bonus = (0.25 * value) + (0.10 * exploration)
    return {
        "count": count,
        "value": value,
        "exploration": exploration,
        "bonus": bonus,
    }


def apply_reward_updates(state: dict[str, Any], updates: list[dict[str, Any]]) -> dict[str, Any]:
    rules = state.setdefault("rules", {})
    meta = state.setdefault("meta", {})
    total_updates = int(meta.get("total_updates", 0) or 0)
    applied = 0
    for update in updates:
        signature = str(update.get("signature") or "").strip()
        if not signature:
            continue
        reward = float(update.get("reward", 0.0) or 0.0)
        row = rules.get(signature)
        if not isinstance(row, dict):
            row = {
                "count": 0,
                "total_reward": 0.0,
                "avg_reward": 0.0,
                "last_reward": 0.0,
                "last_updated": _utc_now(),
            }
        row["count"] = int(row.get("count", 0) or 0) + 1
        row["total_reward"] = float(row.get("total_reward", 0.0) or 0.0) + reward
        row["avg_reward"] = float(row["total_reward"]) / float(row["count"])
        row["last_reward"] = reward
        row["last_updated"] = _utc_now()
        rules[signature] = row
        total_updates += 1
        applied += 1
    meta["total_updates"] = total_updates
    meta["updated_at"] = _utc_now()
    return {"applied": applied, "total_updates": total_updates}
