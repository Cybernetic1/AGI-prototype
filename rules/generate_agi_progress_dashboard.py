#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import select
import sys
import termios
import tty
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_RULE_CATEGORIES = [
    "physical_inventory",
    "quantitative_arithmetic",
    "temporal_event",
    "causal_intervention",
    "social_intent",
    "epistemic_defeasible",
]

_CATEGORY_KEYWORDS = {
    "physical_inventory": {
        "inventory", "has", "remaining", "state", "state_transition", "owner", "item", "transfer", "buy", "sell", "give",
    },
    "quantitative_arithmetic": {
        "add", "subtract", "multiply", "divide", "percent", "ratio", "total", "quantity", "price", "cost", "rate", "unit",
    },
    "temporal_event": {
        "temporal", "event", "before", "after", "later", "earlier", "duration", "time", "order",
    },
    "causal_intervention": {
        "cause", "effect", "delta", "change", "intervention", "trigger", "impact",
    },
    "social_intent": {
        "social", "agent", "recipient", "beneficiary", "intend", "goal", "promise", "commit",
    },
    "epistemic_defeasible": {
        "belief", "default", "defeasible", "revise", "contradiction", "exception", "prefer", "uncertain", "abduce",
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def _resolve_widening_report(cli_path: str) -> Path:
    if cli_path:
        path = Path(cli_path)
        return path if path.is_absolute() else (REPO_ROOT / path)
    candidates = [
        REPO_ROOT / "rules/reports/agi_bootstrap_curriculum_widening_v2.json",
        REPO_ROOT / "rules/reports/agi_bootstrap_curriculum_widening_v1.json",
    ]
    for item in candidates:
        if item.exists():
            return item
    raise FileNotFoundError("no widening report found (expected v2 or v1 under rules/reports)")


def _get_profile(report: dict[str, Any], profile_name: str) -> dict[str, Any]:
    for profile in report.get("profiles", []):
        if isinstance(profile, dict) and profile.get("profile") == profile_name:
            return profile
    raise KeyError(f"profile '{profile_name}' not found")


def _extract_slice_report(slice_row: dict[str, Any]) -> dict[str, Any] | None:
    baseline_report = slice_row.get("baseline_report")
    if isinstance(baseline_report, dict):
        return baseline_report
    path_val = slice_row.get("baseline_report_path")
    if not path_val:
        return None
    path = Path(str(path_val))
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        return None
    return _load_json(path)


def _slice_gate_score(slice_row: dict[str, Any]) -> float:
    gates = slice_row.get("gate_results")
    if not isinstance(gates, list) or not gates:
        return 0.0
    passed = sum(1 for g in gates if isinstance(g, dict) and bool(g.get("passed")))
    return 100.0 * (passed / len(gates))


def _slice_metrics(slice_row: dict[str, Any], profile_name: str = "hybrid") -> dict[str, float] | None:
    report = _extract_slice_report(slice_row)
    if not isinstance(report, dict):
        return None
    profile = _get_profile(report, profile_name)
    solve = float(profile["capability"]["solve_rate"]) * 100.0
    symbolic = float(profile["symbolic_validity"]["symbolic_valid_rate"]) * 100.0
    gate = _slice_gate_score(slice_row)
    return {
        "solve_rate_pct": solve,
        "symbolic_validity_pct": symbolic,
        "stability_pct": gate,
        "rule_quality_pct": (0.6 * symbolic) + (0.4 * gate),
    }


def _slice_category(
    widening: dict[str, Any],
    *,
    category_id: str,
    label: str,
    slice_ids: list[str],
    description: str,
) -> dict[str, Any]:
    slice_rows = [
        row
        for row in widening.get("slices", [])
        if isinstance(row, dict) and str(row.get("slice_id")) in set(slice_ids)
    ]
    coverage_pct = 100.0 * (len(slice_rows) / max(1, len(slice_ids)))

    metrics = [_slice_metrics(row) for row in slice_rows]
    usable = [m for m in metrics if isinstance(m, dict)]
    if not usable:
        return {
            "id": category_id,
            "label": label,
            "description": description,
            "sources": slice_ids,
            "coverage_pct": round(coverage_pct, 2),
            "symbolic_validity_pct": 0.0,
            "solve_utility_pct": 0.0,
            "rule_quality_pct": 0.0,
            "stability_pct": 0.0,
            "progress_pct": 0.0,
            "notes": "No usable slice baseline reports were found for this category.",
        }

    symbolic = mean(m["symbolic_validity_pct"] for m in usable)
    utility = mean(m["solve_rate_pct"] for m in usable)
    quality = mean(m["rule_quality_pct"] for m in usable)
    stability = mean(m["stability_pct"] for m in usable)
    progress = (
        (0.2 * coverage_pct)
        + (0.25 * symbolic)
        + (0.25 * utility)
        + (0.15 * quality)
        + (0.15 * stability)
    )
    return {
        "id": category_id,
        "label": label,
        "description": description,
        "sources": slice_ids,
        "coverage_pct": round(coverage_pct, 2),
        "symbolic_validity_pct": round(symbolic, 2),
        "solve_utility_pct": round(utility, 2),
        "rule_quality_pct": round(quality, 2),
        "stability_pct": round(stability, 2),
        "progress_pct": round(progress, 2),
        "notes": "",
    }


def _modal_context_category() -> dict[str, Any]:
    required = [
        REPO_ROOT / "working_memory/semantic_schema.py",
        REPO_ROOT / "working_memory/working_memory.py",
        REPO_ROOT / "working_memory/nl_rule_injection.py",
        REPO_ROOT / "tests/test_modal_context_guards.py",
        REPO_ROOT / "tests/test_modal_rule_validation.py",
    ]
    present = sum(1 for p in required if p.exists())
    coverage_pct = 100.0 * (present / len(required))
    # This category is currently implementation-accounted rather than benchmark-accounted.
    symbolic = coverage_pct
    utility = 40.0
    quality = 0.6 * coverage_pct + 0.4 * symbolic
    stability = 80.0 if present == len(required) else 50.0
    progress = (
        (0.2 * coverage_pct)
        + (0.25 * symbolic)
        + (0.25 * utility)
        + (0.15 * quality)
        + (0.15 * stability)
    )
    return {
        "id": "modal_context_reasoning",
        "label": "Modal/context reasoning",
        "description": "Context-indexed facts, accessibility bridges, and modal validation guards.",
        "sources": [str(p.relative_to(REPO_ROOT)) for p in required],
        "coverage_pct": round(coverage_pct, 2),
        "symbolic_validity_pct": round(symbolic, 2),
        "solve_utility_pct": round(utility, 2),
        "rule_quality_pct": round(quality, 2),
        "stability_pct": round(stability, 2),
        "progress_pct": round(progress, 2),
        "notes": "Provisional score based on implemented artifacts/tests; benchmark slices not yet separated.",
    }


def _background_trend(metrics_path: Path) -> dict[str, Any]:
    if not metrics_path.exists():
        return {"available": False}
    rows = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    if not rows:
        return {"available": False}
    solve = [float(r.get("solve_rate", 0.0)) for r in rows if r.get("solve_rate") is not None]
    valid = [float(r.get("symbolic_valid_rate", 0.0)) for r in rows if r.get("symbolic_valid_rate") is not None]
    cost = [float(r.get("wall_time_sec_per_valid_solve", 0.0)) for r in rows if r.get("wall_time_sec_per_valid_solve") is not None]
    tail = cost[-20:] if cost else []
    return {
        "available": True,
        "rows": len(rows),
        "first_timestamp": rows[0].get("timestamp"),
        "last_timestamp": rows[-1].get("timestamp"),
        "solve_rate_avg": round(mean(solve), 6) if solve else 0.0,
        "symbolic_valid_rate_avg": round(mean(valid), 6) if valid else 0.0,
        "cost_per_valid_solve_avg": round(mean(cost), 6) if cost else 0.0,
        "cost_per_valid_solve_last20_avg": round(mean(tail), 6) if tail else 0.0,
    }


def _overall_progress(categories: list[dict[str, Any]]) -> float:
    if not categories:
        return 0.0
    return round(mean(float(c.get("progress_pct", 0.0)) for c in categories), 2)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


def _rule_learning_summary(*, utility_path: Path, registry_path: Path, snapshots_path: Path) -> dict[str, Any]:
    utility = _load_json(utility_path) if utility_path.exists() else {"meta": {}, "rules": {}}
    rules = utility.get("rules") if isinstance(utility.get("rules"), dict) else {}
    tracked = [
        {
            "signature": sig,
            "count": int((row or {}).get("count", 0) or 0),
            "avg_reward": float((row or {}).get("avg_reward", 0.0) or 0.0),
            "last_reward": float((row or {}).get("last_reward", 0.0) or 0.0),
            "last_updated": str((row or {}).get("last_updated") or ""),
        }
        for sig, row in rules.items()
    ]
    tracked.sort(key=lambda r: (r["count"], r["avg_reward"]), reverse=True)
    under_learning = [r for r in tracked if r["count"] < 5]

    registry = _load_yaml(registry_path)
    registry_rules = registry.get("rules") if isinstance(registry.get("rules"), list) else []
    proposal_only = 0
    ilp_rules = 0
    for row in registry_rules:
        if not isinstance(row, dict):
            continue
        if str(row.get("runtime_authority") or "").strip().lower() == "proposal_only":
            proposal_only += 1
        if str(row.get("proposal_kind") or "").strip().lower() == "ilp_rule":
            ilp_rules += 1

    snapshots = []
    if snapshots_path.exists():
        payload = _load_json(snapshots_path)
        history = payload.get("snapshots")
        if isinstance(history, list):
            snapshots = [item for item in history if isinstance(item, dict)]

    return {
        "utility_path": str(utility_path),
        "registry_path": str(registry_path),
        "snapshots_path": str(snapshots_path),
        "utility_total_updates": int((utility.get("meta", {}) or {}).get("total_updates", 0) or 0),
        "tracked_rule_count": len(tracked),
        "rules_under_learning_count": len(under_learning),
        "registry_rule_count": len(registry_rules),
        "proposal_only_rule_count": proposal_only,
        "ilp_rule_count": ilp_rules,
        "top_rules": tracked[:20],
        "snapshots": snapshots[-50:],
    }


def _capture_rule_snapshot(*, summary: dict[str, Any], snapshots_path: Path, label: str = "") -> dict[str, Any]:
    current = {"snapshots": []}
    if snapshots_path.exists():
        existing = _load_json(snapshots_path)
        if isinstance(existing, dict):
            current = existing
        current.setdefault("snapshots", [])
    snapshot = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "tracked_rule_count": int(summary.get("tracked_rule_count", 0) or 0),
        "rules_under_learning_count": int(summary.get("rules_under_learning_count", 0) or 0),
        "utility_total_updates": int(summary.get("utility_total_updates", 0) or 0),
        "top_rules": list(summary.get("top_rules", []))[:20],
    }
    current["snapshots"].append(snapshot)
    snapshots_path.parent.mkdir(parents=True, exist_ok=True)
    snapshots_path.write_text(json.dumps(current, indent=2), encoding="utf-8")
    return snapshot


def _classify_rule_category(*, text: str, family: str = "", preferred: str = "") -> str:
    preferred_norm = str(preferred or "").strip().lower()
    if preferred_norm in EXPECTED_RULE_CATEGORIES:
        return preferred_norm

    haystack = f"{family} {text}".lower()
    scores: dict[str, int] = {}
    for cat, keys in _CATEGORY_KEYWORDS.items():
        score = sum(1 for k in keys if k in haystack)
        if score > 0:
            scores[cat] = score
    if not scores:
        return "unclassified"
    return sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=True)[0][0]


def _rule_inventory_summary(
    *,
    registry_path: Path,
    metarules_path: Path,
    semantic_ar_path: Path,
    rule_shape_path: Path,
    utility_path: Path,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []

    registry = _load_yaml(registry_path)
    for row in registry.get("rules", []) if isinstance(registry.get("rules"), list) else []:
        if not isinstance(row, dict):
            continue
        compiled = row.get("compiled") if isinstance(row.get("compiled"), dict) else {}
        family = str(compiled.get("family") or row.get("proposal_kind") or "")
        text = str(compiled.get("rule_text") or row.get("raw_rule") or "")
        category = _classify_rule_category(text=text, family=family)
        entries.append(
            {
                "source": "registry",
                "category": category,
                "signature": str(row.get("rule_id") or compiled.get("rule_id") or family or "registry-rule"),
                "status": str(row.get("status") or "unknown"),
            }
        )

    metarules = _load_json(metarules_path) if metarules_path.exists() else {}
    for row in metarules.get("rules", []) if isinstance(metarules.get("rules"), list) else []:
        if not isinstance(row, dict):
            continue
        family = str(row.get("family") or "")
        name = str(row.get("name") or family or "metarule")
        text = f"{row.get('description') or ''} {row.get('calc') or ''}"
        category = _classify_rule_category(text=text, family=family)
        entries.append(
            {
                "source": "metarules",
                "category": category,
                "signature": name,
                "status": "snapshot",
            }
        )

    semantic_ar = _load_json(semantic_ar_path) if semantic_ar_path.exists() else {}
    seen_seed_rules: set[tuple[str, str]] = set()
    for ex in semantic_ar.get("examples", []) if isinstance(semantic_ar.get("examples"), list) else []:
        if not isinstance(ex, dict):
            continue
        cat = str(ex.get("category") or "").strip().lower()
        for rule in ex.get("seed_rules", []) if isinstance(ex.get("seed_rules"), list) else []:
            rule_text = str(rule or "").strip()
            if not rule_text:
                continue
            key = (cat, rule_text)
            if key in seen_seed_rules:
                continue
            seen_seed_rules.add(key)
            category = _classify_rule_category(text=rule_text, preferred=cat)
            entries.append(
                {
                    "source": "semantic_ar_seed",
                    "category": category,
                    "signature": rule_text[:120],
                    "status": "template",
                }
            )

    by_source: dict[str, int] = {}
    by_category: dict[str, int] = {}
    category_examples: dict[str, list[str]] = {}
    for row in entries:
        src = str(row.get("source") or "unknown")
        cat = str(row.get("category") or "unclassified")
        by_source[src] = by_source.get(src, 0) + 1
        by_category[cat] = by_category.get(cat, 0) + 1
        category_examples.setdefault(cat, [])
        if len(category_examples[cat]) < 3:
            category_examples[cat].append(str(row.get("signature") or ""))

    shape_report = _load_json(rule_shape_path) if rule_shape_path.exists() else {}
    allocated_shapes = {
        str(item.get("shape"))
        for item in shape_report.get("rule_shapes", [])
        if isinstance(item, dict) and item.get("shape")
    }
    observed_required_shapes = set()
    for ex in semantic_ar.get("examples", []) if isinstance(semantic_ar.get("examples"), list) else []:
        if not isinstance(ex, dict):
            continue
        for shape in ex.get("required_shapes", []) if isinstance(ex.get("required_shapes"), list) else []:
            if str(shape).strip():
                observed_required_shapes.add(str(shape).strip())

    utility = _load_json(utility_path) if utility_path.exists() else {}
    utility_rules = utility.get("rules", {}) if isinstance(utility.get("rules"), dict) else {}

    missing_categories = [cat for cat in EXPECTED_RULE_CATEGORIES if by_category.get(cat, 0) <= 0]
    unclassified = int(by_category.get("unclassified", 0))

    return {
        "total_rules_observed": len(entries),
        "by_source": dict(sorted(by_source.items())),
        "by_category": dict(sorted(by_category.items())),
        "category_examples": category_examples,
        "missing_categories": missing_categories,
        "unclassified_count": unclassified,
        "allocated_shapes": sorted(allocated_shapes),
        "observed_required_shapes": sorted(observed_required_shapes),
        "missing_shapes_for_observed_requirements": sorted(observed_required_shapes - allocated_shapes),
        "unused_allocated_shapes": sorted(allocated_shapes - observed_required_shapes),
        "tracked_utility_rules": len(utility_rules),
    }


def _read_single_key() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ready, _, _ = select.select([sys.stdin], [], [], 0.1)
        if not ready:
            return ""
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _normalize_semantic_graph(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {"nodes": [], "edges": []}
    nodes = raw.get("nodes") or []
    edges = raw.get("edges") or []

    if isinstance(nodes, dict):
        normalized_nodes = []
        for node_id, node in nodes.items():
            payload = dict(node) if isinstance(node, dict) else {}
            payload.setdefault("id", str(node_id))
            payload.setdefault("label", str(node_id))
            payload.setdefault("kind", payload.get("kind", "entity"))
            normalized_nodes.append(payload)
        nodes = normalized_nodes

    normalized_edges = []
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source = edge.get("source") or edge.get("from")
        target = edge.get("target") or edge.get("to")
        if source is None or target is None:
            continue
        item = {
            "from": str(source),
            "to": str(target),
            "label": str(edge.get("relation") or edge.get("label") or "related"),
            "arrows": "to",
            "color": {"color": "#4b5563"},
        }
        for key in ("id", "title", "value"):
            if key in edge:
                item[key] = edge[key]
        normalized_edges.append(item)

    return {"nodes": nodes, "edges": normalized_edges}


def _render_html(payload: dict[str, Any]) -> str:
    rows = []
    for cat in payload.get("categories", []):
        bar = f"{cat['progress_pct']:.1f}%"
        rows.append(
            f"<tr>"
            f"<td><b>{cat['label']}</b><br><small>{cat['description']}</small></td>"
            f"<td>{cat['coverage_pct']:.1f}%</td>"
            f"<td>{cat['symbolic_validity_pct']:.1f}%</td>"
            f"<td>{cat['solve_utility_pct']:.1f}%</td>"
            f"<td>{cat['rule_quality_pct']:.1f}%</td>"
            f"<td>{cat['stability_pct']:.1f}%</td>"
            f"<td>{cat.get('confidence_pct', 0.0):.1f}%</td>"
            f"<td><div class='bar'><span style='width:{cat['progress_pct']:.1f}%'></span></div><small>{bar}</small></td>"
            f"</tr>"
        )
    trend = payload.get("background_trend", {})
    rule_learning = payload.get("rule_learning", {})
    rule_inventory = payload.get("rule_inventory", {}) if isinstance(payload.get("rule_inventory"), dict) else {}
    snapshots = rule_learning.get("snapshots", []) if isinstance(rule_learning.get("snapshots"), list) else []
    trend_html = (
        f"<li>Rows: {trend.get('rows')}</li>"
        f"<li>Solve avg: {trend.get('solve_rate_avg')}</li>"
        f"<li>Symbolic-valid avg: {trend.get('symbolic_valid_rate_avg')}</li>"
        f"<li>Cost/valid avg: {trend.get('cost_per_valid_solve_avg')}</li>"
        f"<li>Cost/valid last20 avg: {trend.get('cost_per_valid_solve_last20_avg')}</li>"
    ) if trend.get("available") else "<li>No background loop metrics available.</li>"
    top_rows = []
    for row in (rule_learning.get("top_rules") or [])[:12]:
        top_rows.append(
            f"<tr><td><code>{str(row.get('signature',''))[:20]}...</code></td>"
            f"<td>{int(row.get('count', 0) or 0)}</td>"
            f"<td>{float(row.get('avg_reward', 0.0) or 0.0):.3f}</td>"
            f"<td>{float(row.get('last_reward', 0.0) or 0.0):.3f}</td></tr>"
        )
    snapshot_rows = []
    for snap in reversed(snapshots[-20:]):
        snapshot_rows.append(
            f"<tr><td>{snap.get('captured_at','')}</td>"
            f"<td>{snap.get('label','')}</td>"
            f"<td>{snap.get('tracked_rule_count',0)}</td>"
            f"<td>{snap.get('rules_under_learning_count',0)}</td>"
            f"<td>{snap.get('utility_total_updates',0)}</td></tr>"
        )
    source_rows = []
    for src, count in (rule_inventory.get("by_source") or {}).items():
        source_rows.append(f"<tr><td>{src}</td><td>{int(count)}</td></tr>")
    category_rows = []
    for cat, count in (rule_inventory.get("by_category") or {}).items():
        examples = ", ".join((rule_inventory.get("category_examples") or {}).get(cat, [])[:2])
        category_rows.append(
            f"<tr><td>{cat}</td><td>{int(count)}</td><td><small>{examples}</small></td></tr>"
        )
    missing_categories = rule_inventory.get("missing_categories") or []
    missing_shapes = rule_inventory.get("missing_shapes_for_observed_requirements") or []

    semantic_graph = _normalize_semantic_graph(payload.get("semantic_graph"))
    semantic_nodes = json.dumps(semantic_graph.get("nodes", []), ensure_ascii=False)
    semantic_edges = json.dumps(semantic_graph.get("edges", []), ensure_ascii=False)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AGI Prototype Progress Dashboard</title>
  <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #111; }}
    h1 {{ margin-bottom: 4px; }}
    .meta {{ color: #555; margin-bottom: 20px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    th {{ background: #f5f5f5; text-align: left; }}
    .bar {{ background: #eee; height: 10px; border-radius: 5px; overflow: hidden; margin-bottom: 4px; }}
    .bar > span {{ display: block; height: 100%; background: #3b82f6; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px; margin: 16px 0; }}
    #semantic-graph {{ height: 540px; border: 1px solid #d1d5db; border-radius: 8px; overflow: hidden; }}
  </style>
</head>
<body>
  <h1>AGI Progress Dashboard</h1>
  <div class="meta">Generated at {payload['generated_at']} | Overall progress: <b>{payload['overall_progress_pct']:.1f}%</b></div>

  <div class="card">
    <h3>Category progress map</h3>
    <table>
      <thead>
        <tr>
          <th>Category</th>
          <th>Coverage</th>
          <th>Symbolic validity</th>
          <th>Solve utility</th>
          <th>Rule quality</th>
          <th>Stability</th>
          <th>Confidence</th>
          <th>Progress</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </div>

  <div class="card">
    <h3>Semantic graph view</h3>
    <div id="semantic-graph"></div>
    <script>
      const semanticNodes = {semantic_nodes};
      const semanticEdges = {semantic_edges};
      const container = document.getElementById('semantic-graph');
      const nodes = new vis.DataSet(semanticNodes.map((node) => ({{
        id: String(node.id),
        label: String(node.label || node.id),
        title: String(node.kind || 'node'),
        shape: node.shape || 'box',
        color: node.color || {{ background: '#dbeafe', border: '#2563eb', highlight: {{ background: '#bfdbfe', border: '#1d4ed8' }} }},
        font: {{ color: '#111827', face: 'sans-serif' }}
      }})));
      const edges = new vis.DataSet(semanticEdges.map((edge) => ({{
        from: String(edge.from || edge.source),
        to: String(edge.to || edge.target),
        label: String(edge.label || ''),
        arrows: edge.arrows || 'to',
        color: edge.color || {{ color: '#4b5563' }},
        font: {{ align: 'middle', color: '#374151' }}
      }})));
      const network = new vis.Network(container, {{ nodes, edges }}, {{
        physics: {{ enabled: true, solver: 'forceAtlas2Based', stabilization: {{ iterations: 200 }} }},
        interaction: {{ hover: true, tooltipDelay: 150 }},
        edges: {{ smooth: {{ type: 'dynamic' }} }},
        layout: {{ improvedLayout: true }},
        nodes: {{ borderWidth: 1.5 }}
      }});
      network.fit();
    </script>
  </div>

  <div class="card">
    <h3>Background loop trend (from metrics.jsonl)</h3>
    <ul>{trend_html}</ul>
  </div>

  <div class="card">
    <h3>Rule learning monitor</h3>
    <ul>
      <li>Tracked rules in utility state: <b>{rule_learning.get('tracked_rule_count', 0)}</b></li>
      <li>Rules under learning (count &lt; 5): <b>{rule_learning.get('rules_under_learning_count', 0)}</b></li>
      <li>Registry proposal-only rules: <b>{rule_learning.get('proposal_only_rule_count', 0)}</b></li>
      <li>Registry ILP rules: <b>{rule_learning.get('ilp_rule_count', 0)}</b></li>
      <li>Utility total updates: <b>{rule_learning.get('utility_total_updates', 0)}</b></li>
    </ul>
    <h4>Top tracked rules</h4>
    <table>
      <thead><tr><th>Signature</th><th>Count</th><th>Avg reward</th><th>Last reward</th></tr></thead>
      <tbody>{''.join(top_rows) if top_rows else "<tr><td colspan='4'>No rule utility entries yet.</td></tr>"}</tbody>
    </table>
    <h4>Snapshots (press <code>s</code> in keypress mode)</h4>
    <table>
      <thead><tr><th>Captured at</th><th>Label</th><th>Tracked</th><th>Under learning</th><th>Total updates</th></tr></thead>
      <tbody>{''.join(snapshot_rows) if snapshot_rows else "<tr><td colspan='5'>No snapshots yet.</td></tr>"}</tbody>
    </table>
  </div>

  <div class="card">
    <h3>Rule inventory overview</h3>
    <ul>
      <li>Total observed rules across sources: <b>{rule_inventory.get('total_rules_observed', 0)}</b></li>
      <li>Unclassified rules: <b>{rule_inventory.get('unclassified_count', 0)}</b></li>
      <li>Tracked utility rule IDs: <b>{rule_inventory.get('tracked_utility_rules', 0)}</b></li>
      <li>Missing expected categories: <b>{', '.join(missing_categories) if missing_categories else 'none'}</b></li>
      <li>Missing shapes for observed requirements: <b>{', '.join(missing_shapes) if missing_shapes else 'none'}</b></li>
    </ul>
    <h4>By source</h4>
    <table>
      <thead><tr><th>Source</th><th>Count</th></tr></thead>
      <tbody>{''.join(source_rows) if source_rows else "<tr><td colspan='2'>No rule sources found.</td></tr>"}</tbody>
    </table>
    <h4>By category</h4>
    <table>
      <thead><tr><th>Category</th><th>Count</th><th>Examples</th></tr></thead>
      <tbody>{''.join(category_rows) if category_rows else "<tr><td colspan='3'>No categorized rules found.</td></tr>"}</tbody>
    </table>
  </div>

  <div class="card">
    <h3>Accounting notes</h3>
    <ul>
      <li>Category progress uses weighted accounting: coverage (20%), symbolic validity (25%), solve utility (25%), rule quality (15%), stability (15%).</li>
      <li>Rule quality is currently a practical proxy derived from symbolic validity + gate stability; this can be replaced later with explicit information-gain/compressivity metrics.</li>
      <li>Each AGI category now uses confidence-weighted estimation: estimated_progress = confidence * measured_proxy + (1-confidence) * conservative prior.</li>
      <li>This prevents narrow-slice 100% scores from being interpreted as near-AGI completion.</li>
    </ul>
  </div>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a basic AGI category progress dashboard (JSON + HTML).")
    parser.add_argument("--baseline-report", default="rules/reports/agi_bootstrap_baseline_v1.json")
    parser.add_argument("--widening-report", default="", help="Optional explicit widening report path.")
    parser.add_argument("--ledger-report", default="rules/reports/agi_bootstrap_unified_contract_ledger_v1.json")
    parser.add_argument("--background-metrics", default="rules/reports/background_loop/metrics.jsonl")
    parser.add_argument("--utility-report", default="rules/reports/ilp_rule_utility_v1.json")
    parser.add_argument("--rule-registry", default="rules/nl_rule_registry.yaml")
    parser.add_argument("--metarules-report", default="rules/reports/metarules_snapshot_v1.json")
    parser.add_argument("--semantic-ar-report", default="rules/reports/semantic_ar_worked_examples_v1.json")
    parser.add_argument("--rule-shape-report", default="rules/reports/rule_shape_allocation_300m_v1.json")
    parser.add_argument("--snapshot-file", default="rules/reports/rule_learning_snapshots_v1.json")
    parser.add_argument("--semantic-graph", default="rules/reports/semantic_graph_example.json", help="Optional semantic graph JSON to visualize in the dashboard (node/edge graph).")
    parser.add_argument("--snapshot-now", action="store_true", help="Capture one rule-learning snapshot while generating dashboard.")
    parser.add_argument("--snapshot-label", default="", help="Optional label for snapshot entries.")
    parser.add_argument("--snapshot-keypress", action="store_true", help="Interactive mode: press 's' to snapshot and refresh dashboard, 'q' to quit.")
    parser.add_argument("--output-json", default="rules/reports/agi_progress_dashboard_v1.json")
    parser.add_argument("--output-html", default="rules/reports/agi_progress_dashboard_v1.html")
    args = parser.parse_args()

    baseline_path = Path(args.baseline_report)
    if not baseline_path.is_absolute():
        baseline_path = REPO_ROOT / baseline_path
    widening_path = _resolve_widening_report(args.widening_report)
    ledger_path = Path(args.ledger_report)
    if not ledger_path.is_absolute():
        ledger_path = REPO_ROOT / ledger_path
    bg_metrics_path = Path(args.background_metrics)
    if not bg_metrics_path.is_absolute():
        bg_metrics_path = REPO_ROOT / bg_metrics_path
    utility_path = Path(args.utility_report)
    if not utility_path.is_absolute():
        utility_path = REPO_ROOT / utility_path
    registry_path = Path(args.rule_registry)
    if not registry_path.is_absolute():
        registry_path = REPO_ROOT / registry_path
    metarules_path = Path(args.metarules_report)
    if not metarules_path.is_absolute():
        metarules_path = REPO_ROOT / metarules_path
    semantic_ar_path = Path(args.semantic_ar_report)
    if not semantic_ar_path.is_absolute():
        semantic_ar_path = REPO_ROOT / semantic_ar_path
    rule_shape_path = Path(args.rule_shape_report)
    if not rule_shape_path.is_absolute():
        rule_shape_path = REPO_ROOT / rule_shape_path
    snapshots_path = Path(args.snapshot_file)
    if not snapshots_path.is_absolute():
        snapshots_path = REPO_ROOT / snapshots_path
    semantic_graph_path = Path(args.semantic_graph)
    if not semantic_graph_path.is_absolute():
        semantic_graph_path = REPO_ROOT / semantic_graph_path

    baseline = _load_json(baseline_path)
    widening = _load_json(widening_path)
    ledger = _load_json(ledger_path) if ledger_path.exists() else {}

    # Narrow measured slices are still useful, but we must not equate them to broad AGI
    # completion. We therefore build AGI-layer categories using confidence-weighted estimates.
    finality = _slice_category(
        widening,
        category_id="finality_state_machine",
        label="Finality/state-machine reasoning",
        slice_ids=["invariant-core-finality"],
        description="Final-state authority and state-transition consistency on GSM-style cases.",
    )
    gsm = _slice_category(
        widening,
        category_id="gsm_template_generalization",
        label="GSM template generalization",
        slice_ids=["adjacent-template-core", "adjacent-template-full"],
        description="Generalization across arithmetic templates under symbolic validity constraints.",
    )
    tiny = _slice_category(
        widening,
        category_id="tinystories_state_tracking",
        label="Narrative state tracking",
        slice_ids=["tinystories-state-tracking-compact"],
        description="Entity/state continuity over compact TinyStories symbolic slices.",
    )
    mixed = _slice_category(
        widening,
        category_id="mixed_narrative_symbolic",
        label="Mixed narrative-symbolic transfer",
        slice_ids=["mixed-narrative-symbolic-compact"],
        description="Cross-domain symbolic grounding in mixed narrative + arithmetic contexts.",
    )
    modal = _modal_context_category()

    def _avg_progress(*items: dict[str, Any]) -> float:
        vals = [float(i.get("progress_pct", 0.0)) for i in items if isinstance(i, dict)]
        return mean(vals) if vals else 0.0

    def _estimate_category(
        *,
        category_id: str,
        label: str,
        description: str,
        measured_proxy_pct: float,
        prior_pct: float,
        confidence_pct: float,
        notes: str,
    ) -> dict[str, Any]:
        c = max(0.0, min(1.0, confidence_pct / 100.0))
        progress = (c * measured_proxy_pct) + ((1.0 - c) * prior_pct)
        # Keep component columns visible as accountability breakdown, but make it
        # explicit that they come from proxy estimates at current maturity.
        return {
            "id": category_id,
            "label": label,
            "description": description,
            "coverage_pct": round(progress, 2),
            "symbolic_validity_pct": round(progress, 2),
            "solve_utility_pct": round(progress, 2),
            "rule_quality_pct": round(progress, 2),
            "stability_pct": round(progress, 2),
            "confidence_pct": round(confidence_pct, 2),
            "progress_pct": round(progress, 2),
            "notes": notes,
        }

    categories = [
        _estimate_category(
            category_id="perceptual_grounding",
            label="Perceptual/grounding rules",
            description="Map language observations to canonical entities/events/quantities/roles/time.",
            measured_proxy_pct=_avg_progress(tiny, mixed),
            prior_pct=25.0,
            confidence_pct=35.0,
            notes="Proxy from TinyStories + mixed narrative slices; still narrow domain coverage.",
        ),
        _estimate_category(
            category_id="type_ontology",
            label="Type/ontology rules",
            description="Category membership, inheritance, and compatibility constraints.",
            measured_proxy_pct=_avg_progress(mixed, tiny),
            prior_pct=22.0,
            confidence_pct=20.0,
            notes="Weak direct measurement; mostly inferred from symbolic parsing behavior.",
        ),
        _estimate_category(
            category_id="compositional_semantics",
            label="Compositional semantic rules",
            description="Clause composition, reference resolution, scope, quantification, negation.",
            measured_proxy_pct=_avg_progress(mixed, tiny),
            prior_pct=20.0,
            confidence_pct=30.0,
            notes="Partially measured via narrative-symbolic slices; missing explicit scope/negation benchmarks.",
        ),
        _estimate_category(
            category_id="state_transition",
            label="State-transition rules",
            description="Actions/events update world state over time.",
            measured_proxy_pct=_avg_progress(finality),
            prior_pct=40.0,
            confidence_pct=65.0,
            notes="Strongly evidenced by invariant-core-finality slices.",
        ),
        _estimate_category(
            category_id="temporal_causal",
            label="Temporal/causal rules",
            description="Before/after, persistence, cause-effect, intervention effects.",
            measured_proxy_pct=_avg_progress(finality, mixed),
            prior_pct=30.0,
            confidence_pct=45.0,
            notes="Some evidence from state order/finality and narrative slices; intervention tests still limited.",
        ),
        _estimate_category(
            category_id="default_defeasible",
            label="Default + defeasible rules",
            description="Normally-true defaults with explicit exception/override priorities.",
            measured_proxy_pct=_avg_progress(modal, finality),
            prior_pct=20.0,
            confidence_pct=25.0,
            notes="Initial support present, but broad defeasible policy coverage is not yet benchmarked.",
        ),
        _estimate_category(
            category_id="non_monotonic_revision",
            label="Non-monotonic revision rules",
            description="Contradiction handling, belief retraction, preference relations, truth maintenance.",
            measured_proxy_pct=_avg_progress(finality, modal),
            prior_pct=25.0,
            confidence_pct=45.0,
            notes="Supported by contradiction/finality machinery; still lacks broad stress tests across domains.",
        ),
        _estimate_category(
            category_id="numeric_structural",
            label="Numeric/structural reasoning rules",
            description="Arithmetic, units, ratios, constraints, graph dependencies.",
            measured_proxy_pct=_avg_progress(gsm, finality),
            prior_pct=35.0,
            confidence_pct=55.0,
            notes="Best-evidenced area (GSM-style slices), but still not full GSM8K benchmark breadth.",
        ),
        _estimate_category(
            category_id="pragmatic_intent",
            label="Pragmatic/intent rules",
            description="Speaker intent, dialogue acts, implicature, goal inference.",
            measured_proxy_pct=_avg_progress(mixed),
            prior_pct=10.0,
            confidence_pct=10.0,
            notes="Not explicitly benchmarked yet; current score intentionally conservative.",
        ),
        _estimate_category(
            category_id="meta_rules_control",
            label="Meta-rules (learning/control)",
            description="When to propose/specialize/retract/activate rules and apply evidence policies.",
            measured_proxy_pct=60.0 if bool(ledger.get("overall_green")) else 45.0,
            prior_pct=30.0,
            confidence_pct=50.0,
            notes="Measured via governance/approval pipeline health and gate behavior.",
        ),
        _estimate_category(
            category_id="planning_action",
            label="Planning/action rules",
            description="Goal decomposition, preconditions/effects, resources, recovery policies.",
            measured_proxy_pct=40.0,
            prior_pct=10.0,
            confidence_pct=10.0,
            notes="Not directly covered by current benchmark suite.",
        ),
        _estimate_category(
            category_id="social_normative",
            label="Social/normative rules",
            description="Conventions, commitments, permissions/prohibitions, safety boundaries.",
            measured_proxy_pct=20.0,
            prior_pct=5.0,
            confidence_pct=5.0,
            notes="No dedicated slices yet; placeholder estimate only.",
        ),
    ]

    semantic_graph = {}
    if semantic_graph_path.exists():
        semantic_graph = _load_json(semantic_graph_path)
    else:
        semantic_graph = {"nodes": [], "edges": []}

    payload = {
        "dashboard_id": "agi-progress-dashboard-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "baseline_report": str(baseline_path),
            "widening_report": str(widening_path),
            "ledger_report": str(ledger_path) if ledger_path.exists() else "",
            "background_metrics": str(bg_metrics_path),
            "utility_report": str(utility_path),
            "rule_registry": str(registry_path),
            "metarules_report": str(metarules_path),
            "semantic_ar_report": str(semantic_ar_path),
            "rule_shape_report": str(rule_shape_path),
            "snapshot_file": str(snapshots_path),
            "semantic_graph": str(semantic_graph_path),
        },
        "ledger_overall_green": bool(ledger.get("overall_green")) if isinstance(ledger, dict) else False,
        "categories": categories,
        "background_trend": _background_trend(bg_metrics_path),
        "semantic_graph": semantic_graph,
        "rule_learning": _rule_learning_summary(
            utility_path=utility_path,
            registry_path=registry_path,
            snapshots_path=snapshots_path,
        ),
        "rule_inventory": _rule_inventory_summary(
            registry_path=registry_path,
            metarules_path=metarules_path,
            semantic_ar_path=semantic_ar_path,
            rule_shape_path=rule_shape_path,
            utility_path=utility_path,
        ),
        "overall_progress_pct": _overall_progress(categories),
    }

    if args.snapshot_now:
        snap = _capture_rule_snapshot(
            summary=payload["rule_learning"],
            snapshots_path=snapshots_path,
            label=str(args.snapshot_label or ""),
        )
        payload["rule_learning"]["snapshots"] = (payload["rule_learning"].get("snapshots") or []) + [snap]

    def _write_payload(current_payload: dict[str, Any]) -> None:
        out_json = Path(args.output_json)
        if not out_json.is_absolute():
            out_json = REPO_ROOT / out_json
        out_html = Path(args.output_html)
        if not out_html.is_absolute():
            out_html = REPO_ROOT / out_html
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_html.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(current_payload, indent=2), encoding="utf-8")
        out_html.write_text(_render_html(current_payload), encoding="utf-8")
        print(f"Saved dashboard JSON to {out_json}")
        print(f"Saved dashboard HTML to {out_html}")

    _write_payload(payload)
    if args.snapshot_keypress:
        print("Snapshot mode active: press 's' to capture a snapshot and refresh dashboard, 'q' to quit.")
        while True:
            key = _read_single_key().lower()
            if not key:
                continue
            if key == "q":
                print("Snapshot mode stopped.")
                break
            if key == "s":
                refreshed = dict(payload)
                refreshed["generated_at"] = datetime.now(timezone.utc).isoformat()
                refreshed["rule_learning"] = _rule_learning_summary(
                    utility_path=utility_path,
                    registry_path=registry_path,
                    snapshots_path=snapshots_path,
                )
                _capture_rule_snapshot(
                    summary=refreshed["rule_learning"],
                    snapshots_path=snapshots_path,
                    label=str(args.snapshot_label or ""),
                )
                refreshed["rule_learning"] = _rule_learning_summary(
                    utility_path=utility_path,
                    registry_path=registry_path,
                    snapshots_path=snapshots_path,
                )
                _write_payload(refreshed)
                payload = refreshed
                print("Snapshot captured.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
