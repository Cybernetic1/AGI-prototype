from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

_EPHEMERAL_META_CONCEPTS = {
    "RecallCandidate",
    "RecallDecision",
    "ConsolidationReport",
    "RulePriorityUpdate",
    "NullAnswer",
}
_RULE_META_CONCEPTS = {
    "Rule",
    "RuleLifecycle",
    "PreferredStateRule",
    "RemainingRule",
    "StateTransitionRule",
    "DefeasibleMeta",
}
_MAX_LTM_STORE_BYTES = 2_000_000


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def _safe_attrs(attrs: Dict[str, Any]) -> Dict[str, Any]:
    clean: Dict[str, Any] = {}
    for key, value in attrs.items():
        if _is_scalar(value):
            clean[str(key)] = value
        else:
            clean[str(key)] = str(value)
    return clean


def _tokenize_fact(concept: str, attrs: Dict[str, Any]) -> List[str]:
    tokens = [str(concept).lower()]
    for key in ("owner", "item", "goal", "label", "event_kind", "family", "rule_id", "status"):
        value = attrs.get(key)
        if value is None:
            continue
        tokens.append(f"{key}:{str(value).lower()}")
    return tokens


@dataclass
class LongTermMemory:
    path: Path

    @classmethod
    def default(cls) -> "LongTermMemory":
        return cls(Path(__file__).resolve().parents[1] / "rules" / "ltm_store.yaml")

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {"episodes": [], "semantic_memory": [], "rule_memory": []}
        try:
            if self.path.stat().st_size > _MAX_LTM_STORE_BYTES:
                # Defensive fast path: avoid loading pathological oversized stores in
                # interactive runtime loops; maintenance can compact/rotate later.
                return {"episodes": [], "semantic_memory": [], "rule_memory": [], "oversized_store": True}
        except OSError:
            pass
        payload = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        payload.setdefault("episodes", [])
        payload.setdefault("semantic_memory", [])
        payload.setdefault("rule_memory", [])
        return payload

    def _save(self, payload: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")

    def _episode_id(self, cues: List[str], propositions: List[Dict[str, Any]], created_at: str) -> str:
        digest = hashlib.sha1()
        digest.update(created_at.encode("utf-8"))
        digest.update("|".join(sorted(cues)).encode("utf-8"))
        digest.update(str(len(propositions)).encode("utf-8"))
        return f"ep_{digest.hexdigest()[:12]}"

    def capture_episode(
        self,
        facts: Iterable[Any],
        *,
        reward: float = 0.0,
        outcome: str = "",
        extra_cues: Optional[List[str]] = None,
        max_propositions: int = 500,
    ) -> Dict[str, Any]:
        propositions: List[Dict[str, Any]] = []
        rules: List[Dict[str, Any]] = []
        cue_tokens: List[str] = list(extra_cues or [])

        for node in facts:
            concept = getattr(node, "concept", None)
            if concept is None:
                continue
            attrs = getattr(node, "to_dict", lambda: {})()
            if not isinstance(attrs, dict):
                attrs = dict(attrs)
            clean_attrs = _safe_attrs(attrs)
            cue_tokens.extend(_tokenize_fact(str(concept), clean_attrs))

            layer = str(getattr(node, "layer", "reasoning"))
            # Keep recall memory compact and stable by excluding transient bookkeeping WMEs.
            if str(concept) in _EPHEMERAL_META_CONCEPTS:
                continue
            if layer == "meta" and str(concept) not in _RULE_META_CONCEPTS:
                continue

            record = {
                "concept": str(concept),
                "attrs": clean_attrs,
                "layer": layer,
                "source": getattr(node, "source", "unknown"),
                "activation": float(getattr(node, "activation", 1.0)),
            }
            propositions.append(record)
            if str(concept) in {"Rule", "RuleLifecycle", "PreferredStateRule", "RemainingRule", "StateTransitionRule"}:
                rules.append(
                    {
                        "rule_id": clean_attrs.get("rule_id"),
                        "family": clean_attrs.get("family"),
                        "status": clean_attrs.get("status"),
                        "priority": clean_attrs.get("priority"),
                    }
                )
            if len(propositions) >= max(10, int(max_propositions)):
                break

        cues = sorted({c for c in cue_tokens if c})
        created_at = _utc_now()
        interestingness = min(1.0, max(0.0, 0.35 + (0.25 if abs(float(reward)) > 0 else 0.0) + min(0.4, len(cues) / 80.0)))

        episode = {
            "episode_id": self._episode_id(cues, propositions, created_at),
            "created_at": created_at,
            "updated_at": created_at,
            "outcome": outcome,
            "reward": float(reward),
            "interestingness": interestingness,
            "recall_count": 0,
            "archived": False,
            "cues": cues,
            "propositions": propositions,
            "rules": rules,
        }

        payload = self._load()
        payload["episodes"].append(episode)
        self._save(payload)
        return episode

    def sync_rule_registry(self, rules: List[Dict[str, Any]]) -> None:
        payload = self._load()
        seen = {}
        for r in payload.get("rule_memory", []):
            rid = r.get("rule_id")
            if rid:
                seen[rid] = r
        now = _utc_now()
        for entry in rules:
            rid = entry.get("rule_id")
            if not rid:
                continue
            compiled = entry.get("compiled") or {}
            seen[rid] = {
                "rule_id": rid,
                "status": entry.get("status", "proposed"),
                "version": int(entry.get("version", 1)),
                "family": compiled.get("family"),
                "condition": compiled.get("condition"),
                "effect": compiled.get("effect"),
                "priority": compiled.get("priority"),
                "updated_at": entry.get("updated_at", now),
            }
        payload["rule_memory"] = sorted(seen.values(), key=lambda x: str(x.get("rule_id")))
        self._save(payload)

    @staticmethod
    def _cue_similarity(query: List[str], candidate: List[str]) -> float:
        q = set(query)
        c = set(candidate)
        if not q or not c:
            return 0.0
        return float(len(q & c)) / float(len(q | c))

    def associative_recall(
        self,
        cues: List[str],
        *,
        top_k: int = 5,
        min_score: float = 0.1,
        max_total_propositions: int = 300,
        max_episode_propositions: int = 2000,
    ) -> Dict[str, Any]:
        payload = self._load()
        episodes = payload.get("episodes", [])
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for ep in episodes:
            if ep.get("archived"):
                continue
            if len(ep.get("propositions", [])) > int(max_episode_propositions):
                continue
            score = self._cue_similarity(cues, ep.get("cues", []))
            score += 0.1 * float(ep.get("interestingness", 0.0))
            if score >= min_score:
                scored.append((score, ep))
        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [ep for _, ep in scored[:max(1, top_k)]]

        propositions: List[Dict[str, Any]] = []
        seen = set()
        for ep in selected:
            for p in ep.get("propositions", []):
                key = (p.get("concept"), tuple(sorted((p.get("attrs") or {}).items())))
                if key in seen:
                    continue
                seen.add(key)
                propositions.append(p)
                if len(propositions) >= int(max_total_propositions):
                    break
            ep["recall_count"] = int(ep.get("recall_count", 0)) + 1
            ep["updated_at"] = _utc_now()
            if len(propositions) >= int(max_total_propositions):
                break
        self._save(payload)

        return {
            "cues": cues,
            "episodes": [
                {"episode_id": ep.get("episode_id"), "interestingness": ep.get("interestingness"), "outcome": ep.get("outcome")}
                for ep in selected
            ],
            "propositions": propositions,
            "rules": payload.get("rule_memory", []),
        }

    def consolidate(self, *, min_support: int = 2) -> Dict[str, Any]:
        payload = self._load()
        episodes = [ep for ep in payload.get("episodes", []) if not ep.get("archived")]
        support: Dict[str, Dict[str, Any]] = {}

        for ep in episodes:
            for prop in ep.get("propositions", []):
                attrs = prop.get("attrs", {})
                signature = (
                    str(prop.get("concept")),
                    str(attrs.get("owner")),
                    str(attrs.get("item")),
                    str(attrs.get("goal")),
                    str(attrs.get("label")),
                )
                key = "|".join(signature)
                entry = support.setdefault(
                    key,
                    {
                        "proposition": prop,
                        "support": 0,
                        "avg_reward_sum": 0.0,
                    },
                )
                entry["support"] += 1
                entry["avg_reward_sum"] += float(ep.get("reward", 0.0))

        semantic_memory: List[Dict[str, Any]] = []
        for key, entry in support.items():
            if int(entry["support"]) < int(min_support):
                continue
            semantic_memory.append(
                {
                    "signature": key,
                    "support": int(entry["support"]),
                    "avg_reward": float(entry["avg_reward_sum"]) / float(entry["support"]),
                    "proposition": entry["proposition"],
                    "updated_at": _utc_now(),
                }
            )
        payload["semantic_memory"] = semantic_memory
        self._save(payload)
        return {"semantic_count": len(semantic_memory)}

    def record_quality_snapshot(
        self,
        *,
        suite: str,
        metrics: Dict[str, float],
        tag: str = "quality",
        max_snapshots: int = 20,
    ) -> Dict[str, Any]:
        payload = self._load()
        snapshots = payload.setdefault("governance_snapshots", {})
        suite_key = str(suite or "default")
        entry = {
            "snapshot_id": f"snap_{len(snapshots.get(suite_key, [])) + 1}_{_utc_now().replace(':', '').replace('-', '').replace('.', '')}",
            "tag": str(tag),
            "created_at": _utc_now(),
            "metrics": {str(k): float(v) for k, v in (metrics or {}).items()},
        }
        suite_list = snapshots.setdefault(suite_key, [])
        suite_list.append(entry)
        if len(suite_list) > int(max_snapshots):
            suite_list[:] = suite_list[-int(max_snapshots):]
        payload["governance_latest"] = entry
        self._save(payload)
        return entry

    def check_quality_drift(
        self,
        *,
        suite: str,
        current_metrics: Dict[str, float],
        quality_metric: str = "symbolic_validity",
        drift_threshold: float = 0.05,
    ) -> Dict[str, Any]:
        payload = self._load()
        snapshots = (payload.get("governance_snapshots") or {}).get(str(suite), [])
        if not snapshots:
            return {"drift_detected": False, "baseline": None, "current": current_metrics, "delta": 0.0}

        metric_values = [
            float(snapshot.get("metrics", {}).get(quality_metric, 0.0))
            for snapshot in snapshots
            if quality_metric in (snapshot.get("metrics") or {})
        ]
        baseline = max(metric_values) if metric_values else float((snapshots[-1].get("metrics") or {}).get(quality_metric, 0.0))
        current = float((current_metrics or {}).get(quality_metric, baseline))
        delta = baseline - current
        drift_detected = delta > float(drift_threshold)
        return {
            "drift_detected": bool(drift_detected),
            "metric": quality_metric,
            "baseline": baseline,
            "current": current,
            "delta": delta,
            "threshold": float(drift_threshold),
            "last_snapshot_id": snapshots[-1].get("snapshot_id"),
        }

    def rollback_if_drift(
        self,
        *,
        suite: str,
        current_metrics: Dict[str, float],
        quality_metric: str = "symbolic_validity",
        drift_threshold: float = 0.05,
    ) -> Dict[str, Any]:
        report = self.check_quality_drift(
            suite=suite,
            current_metrics=current_metrics,
            quality_metric=quality_metric,
            drift_threshold=drift_threshold,
        )
        if not report["drift_detected"]:
            return {"rolled_back": False, "report": report}

        payload = self._load()
        snapshots = (payload.get("governance_snapshots") or {}).get(str(suite), [])
        anchor = None
        for candidate in reversed(snapshots):
            metric = float((candidate.get("metrics") or {}).get(quality_metric, 0.0))
            if metric >= float(report["baseline"]) - float(drift_threshold):
                anchor = candidate
                break
        payload["rollback_anchor"] = anchor or snapshots[-1]
        payload["last_drift_report"] = report
        self._save(payload)
        return {"rolled_back": True, "anchor": anchor or snapshots[-1], "report": report}

    def forget(self, *, threshold: float = 0.2, keep_last: int = 50) -> Dict[str, Any]:
        payload = self._load()
        episodes = payload.get("episodes", [])
        if not episodes:
            return {"archived": 0, "kept": 0}

        # Keep newest N episodes regardless of retention score.
        sorted_idx = sorted(range(len(episodes)), key=lambda i: str(episodes[i].get("created_at", "")), reverse=True)
        keep_set = set(sorted_idx[:max(0, keep_last)])
        archived = 0
        for idx, ep in enumerate(episodes):
            if idx in keep_set:
                ep["archived"] = False
                continue
            if ep.get("archived"):
                continue
            retention = (
                float(ep.get("interestingness", 0.0))
                + 0.08 * float(ep.get("recall_count", 0))
                + min(0.2, 0.05 * len(ep.get("rules", [])))
                + min(0.2, 0.02 * len(ep.get("cues", [])))
            )
            if retention < threshold:
                ep["archived"] = True
                archived += 1
        self._save(payload)
        return {"archived": archived, "kept": len(episodes) - archived}
