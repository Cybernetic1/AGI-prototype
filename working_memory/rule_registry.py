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
_ALLOWED_RUNTIME_AUTHORITY = {"full", "proposal_only"}
_ALLOWED_ACTIVATION_POLICY = {"none", "nl_compiler_validator", "proposal_only"}


def _is_teacher_llm_source(source: str) -> bool:
    normalized = str(source or "").strip().lower().replace("-", "_")
    return normalized in {"teacher_llm", "bootstrap_teacher_llm"}


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
            return {"rules": [], "teacher_proposals": []}
        payload = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        if "rules" not in payload or not isinstance(payload["rules"], list):
            payload["rules"] = []
        if "teacher_proposals" not in payload or not isinstance(payload["teacher_proposals"], list):
            payload["teacher_proposals"] = []
        return payload

    def _save(self, payload: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")

    def _find_entry(self, payload: Dict[str, Any], rule_id: str) -> Optional[Dict[str, Any]]:
        for entry in payload.get("rules", []):
            if entry.get("rule_id") == rule_id:
                return entry
        return None

    def register_proposal(
        self,
        *,
        rule_id: str,
        raw_rule: str,
        source: str,
        compiled_payload: Dict[str, Any],
        evidence: Optional[Dict[str, Any]] = None,
        audit: Optional[Dict[str, Any]] = None,
        proposal_kind: str = "rule",
        runtime_authority: Optional[str] = None,
        activation_policy: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = self._load()
        now = _utc_now()
        entry = self._find_entry(payload, rule_id)
        fingerprint = {
            "family": compiled_payload.get("family"),
            "condition": compiled_payload.get("condition"),
            "effect": compiled_payload.get("effect"),
            "priority": compiled_payload.get("priority"),
        }

        normalized_source = str(source or "").strip().lower()
        if normalized_source in {"human", "teacher", "rule_registry"}:
            default_provenance = "teacher"
        elif _is_teacher_llm_source(normalized_source):
            default_provenance = "teacher_llm"
        else:
            default_provenance = "empirical"
        default_mutability = "locked" if default_provenance in {"teacher", "teacher_llm"} else "priority_only"
        desired_runtime_authority = str(runtime_authority or ("proposal_only" if _is_teacher_llm_source(normalized_source) else "full")).strip().lower()
        if desired_runtime_authority not in _ALLOWED_RUNTIME_AUTHORITY:
            desired_runtime_authority = "full"
        desired_activation_policy = str(
            activation_policy
            or (
                "proposal_only"
                if desired_runtime_authority == "proposal_only"
                else "nl_compiler_validator"
                if normalized_source in {"human", "teacher", "rule_registry", "validator", "ltm-validator", "file"}
                else "none"
            )
        ).strip().lower()
        if desired_activation_policy not in _ALLOWED_ACTIVATION_POLICY:
            desired_activation_policy = "none"

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
                "proposal_kind": proposal_kind,
                "runtime_authority": desired_runtime_authority,
                "activation_policy": desired_activation_policy,
                "evidence": evidence or {},
                "audit": audit or {},
                "approval": {
                    "status": "proposed",
                    "reviewer": "",
                    "reason": "",
                    "reviewed_at": now,
                },
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
            merged_audit = dict(entry.get("audit") or {})
            if audit:
                merged_audit.update(audit)
            entry["audit"] = merged_audit
            entry["provenance"] = entry.get("provenance", default_provenance)
            entry["mutability"] = entry.get("mutability", default_mutability)
            entry["proposal_kind"] = str(entry.get("proposal_kind") or proposal_kind)
            current_runtime_authority = str(entry.get("runtime_authority") or desired_runtime_authority).strip().lower()
            if current_runtime_authority not in _ALLOWED_RUNTIME_AUTHORITY:
                current_runtime_authority = desired_runtime_authority
            entry["runtime_authority"] = current_runtime_authority
            current_activation_policy = str(entry.get("activation_policy") or desired_activation_policy).strip().lower()
            if current_activation_policy not in _ALLOWED_ACTIVATION_POLICY:
                current_activation_policy = desired_activation_policy
            entry["activation_policy"] = current_activation_policy
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

    def _activation_gate(self, entry: Dict[str, Any]) -> tuple[bool, List[str]]:
        runtime_authority = str(entry.get("runtime_authority", "full")).strip().lower()
        activation_policy = str(entry.get("activation_policy", "none")).strip().lower()
        if runtime_authority == "proposal_only" or activation_policy == "proposal_only":
            return False, ["proposal-only entries cannot become runtime-active"]
        if activation_policy == "nl_compiler_validator":
            from working_memory.nl_rule_injection import NLRuleCompiler

            ok, issues = NLRuleCompiler.validate(entry.get("compiled") or {})
            if not ok:
                return False, issues
        return True, []

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
            if target == "active":
                gate_ok, gate_issues = self._activation_gate(entry)
                if not gate_ok:
                    raise ValueError(f"activation gate failed for {rule_id}: {'; '.join(gate_issues)}")

        now = _utc_now()
        entry["status"] = target
        entry["updated_at"] = now
        entry["approval"] = {
            "status": target,
            "reviewer": reviewer,
            "reason": reason,
            "reviewed_at": now,
        }
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

    def append_teacher_proposal_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        payload = self._load()
        now = _utc_now()
        events = payload.setdefault("teacher_proposals", [])
        record = dict(event or {})
        if not record.get("proposal_id"):
            record["proposal_id"] = f"teacher-proposal-{len(events) + 1:06d}"
        record.setdefault("logged_at", now)
        events.append(record)
        self._save(payload)
        return record

    def list_teacher_proposals(self) -> List[Dict[str, Any]]:
        payload = self._load()
        return list(payload.get("teacher_proposals", []))

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
