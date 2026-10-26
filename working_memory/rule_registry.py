from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


_ALLOWED_STATUS = {"proposed", "approved", "active", "retracted"}
_ALLOWED_TRANSITIONS = {
    "proposed": {"approved", "retracted"},
    "approved": {"active", "retracted"},
    "active": {"retracted"},
    "retracted": {"approved"},
}
_ALLOWED_MUTABILITY = {"locked", "priority_only", "fully_mutable"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RuleRegistry:
    path: Path

    @classmethod
    def default(cls) -> "RuleRegistry":
        path = Path(__file__).resolve().parents[1] / "rules" / "nl_rule_registry.yaml"
        return cls(path)

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {"rules": []}
        payload = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        if "rules" not in payload or not isinstance(payload["rules"], list):
            payload["rules"] = []
        return payload

    def _save(self, payload: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")

    def _find_entry(self, payload: Dict[str, Any], rule_id: str) -> Optional[Dict[str, Any]]:
        for entry in payload.get("rules", []):
            if entry.get("rule_id") == rule_id:
                return entry
        return None

    def register_proposal(self, *, rule_id: str, raw_rule: str, source: str, compiled_payload: Dict[str, Any], evidence: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = self._load()
        now = _utc_now()
        entry = self._find_entry(payload, rule_id)
        fingerprint = {
            "family": compiled_payload.get("family"),
            "condition": compiled_payload.get("condition"),
            "effect": compiled_payload.get("effect"),
            "priority": compiled_payload.get("priority"),
        }

        default_provenance = "teacher" if source in {"human", "teacher", "rule_registry"} else "empirical"
        default_mutability = "locked" if default_provenance == "teacher" else "priority_only"
        if entry is None:
            entry = {
                "rule_id": rule_id,
                "version": 1,
                "status": "proposed",
                "created_at": now,
                "updated_at": now,
                "source": source,
                "provenance": default_provenance,
                "mutability": default_mutability,
                "raw_rule": raw_rule,
                "compiled": compiled_payload,
                "fingerprint": fingerprint,
                "evidence": evidence or {},
                "accounting": {
                    "applied": 0,
                    "wins": 0,
                    "defeats": 0,
                    "recall_accepted": 0,
                    "recall_rejected": 0,
                    "last_event_at": now,
                },
                "history": [{"at": now, "event": "proposed", "source": source}],
            }
            payload["rules"].append(entry)
        else:
            if entry.get("fingerprint") != fingerprint:
                entry["version"] = int(entry.get("version", 1)) + 1
            entry["updated_at"] = now
            entry["source"] = source
            entry["raw_rule"] = raw_rule
            entry["compiled"] = compiled_payload
            entry["fingerprint"] = fingerprint
            entry["evidence"] = evidence or entry.get("evidence") or {}
            entry["provenance"] = entry.get("provenance", default_provenance)
            entry["mutability"] = entry.get("mutability", default_mutability)
            if entry["mutability"] not in _ALLOWED_MUTABILITY:
                entry["mutability"] = default_mutability
            accounting = entry.setdefault("accounting", {})
            accounting.setdefault("applied", 0)
            accounting.setdefault("wins", 0)
            accounting.setdefault("defeats", 0)
            accounting.setdefault("recall_accepted", 0)
            accounting.setdefault("recall_rejected", 0)
            accounting.setdefault("last_event_at", now)
            if entry.get("status") not in _ALLOWED_STATUS:
                entry["status"] = "proposed"
            entry.setdefault("history", []).append({"at": now, "event": "updated", "source": source})

        self._save(payload)
        return entry

    def set_mutability(self, rule_id: str, mutability: str, *, reviewer: str = "human", reason: str = "") -> Dict[str, Any]:
        target = str(mutability).strip().lower()
        if target not in _ALLOWED_MUTABILITY:
            raise ValueError(f"unsupported mutability: {mutability}")
        payload = self._load()
        entry = self._find_entry(payload, rule_id)
        if entry is None:
            raise KeyError(f"rule not found: {rule_id}")
        now = _utc_now()
        current = str(entry.get("mutability", "priority_only")).lower()
        entry["mutability"] = target
        entry["updated_at"] = now
        entry.setdefault("history", []).append(
            {
                "at": now,
                "event": "mutability_change",
                "from": current,
                "to": target,
                "reviewer": reviewer,
                "reason": reason,
            }
        )
        self._save(payload)
        return entry

    def transition(self, rule_id: str, new_status: str, *, reviewer: str = "human", reason: str = "") -> Dict[str, Any]:
        target = str(new_status).strip().lower()
        if target not in _ALLOWED_STATUS:
            raise ValueError(f"unsupported status: {new_status}")

        payload = self._load()
        entry = self._find_entry(payload, rule_id)
        if entry is None:
            raise KeyError(f"rule not found: {rule_id}")

        current = str(entry.get("status", "proposed")).lower()
        if target != current:
            allowed = _ALLOWED_TRANSITIONS.get(current, set())
            if target not in allowed:
                raise ValueError(f"invalid transition: {current} -> {target}")

        now = _utc_now()
        entry["status"] = target
        entry["updated_at"] = now
        entry.setdefault("history", []).append(
            {
                "at": now,
                "event": "status_change",
                "from": current,
                "to": target,
                "reviewer": reviewer,
                "reason": reason,
            }
        )
        self._save(payload)
        return entry

    def get(self, rule_id: str) -> Optional[Dict[str, Any]]:
        payload = self._load()
        return self._find_entry(payload, rule_id)

    def list_rules(self) -> List[Dict[str, Any]]:
        payload = self._load()
        return list(payload.get("rules", []))

    def record_accounting_event(
        self,
        rule_id: str,
        *,
        applied: int = 0,
        wins: int = 0,
        defeats: int = 0,
        recall_accepted: int = 0,
        recall_rejected: int = 0,
        event: str = "accounting_update",
        details: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        payload = self._load()
        entry = self._find_entry(payload, rule_id)
        if entry is None:
            return None
        now = _utc_now()
        accounting = entry.setdefault("accounting", {})
        accounting["applied"] = int(accounting.get("applied", 0)) + int(applied)
        accounting["wins"] = int(accounting.get("wins", 0)) + int(wins)
        accounting["defeats"] = int(accounting.get("defeats", 0)) + int(defeats)
        accounting["recall_accepted"] = int(accounting.get("recall_accepted", 0)) + int(recall_accepted)
        accounting["recall_rejected"] = int(accounting.get("recall_rejected", 0)) + int(recall_rejected)
        accounting["last_event_at"] = now
        entry["updated_at"] = now
        hist = {"at": now, "event": event}
        if details:
            hist["details"] = details
        entry.setdefault("history", []).append(hist)
        self._save(payload)
        return entry
