#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SYS2_RULES = REPO_ROOT / "rules" / "sys2_rule_kb.yaml"
DEFEASIBLE_RULES = REPO_ROOT / "rules" / "defeasible_rule_kb.yaml"
NL_RULE_REGISTRY = REPO_ROOT / "rules" / "nl_rule_registry.yaml"
NL_RULE_EXAMPLES = REPO_ROOT / "rules" / "nl_rules.txt"
LTM_STORE = REPO_ROOT / "rules" / "ltm_store.yaml"
LT_RULE_CANDIDATES = [
    REPO_ROOT / "logic_transformer" / "semantic_wme_rules.txt",
    REPO_ROOT / "logic_transformer" / "ratio_rules.txt",
]
RULE_SNAPSHOT_PATHS = [
    REPO_ROOT / "rules" / "reports" / "metarules_snapshot_v1.json",
    REPO_ROOT / "rules" / "reports" / "rule_learning_snapshots_v1.json",
]
SYS3_SEED_FILE = REPO_ROOT / "ilp_learner" / "metagol_nds.py"
SYS3_SEED_FILE_2 = REPO_ROOT / "ilp_learner" / "abductive_ilp.py"
SYS2_EXPERTA_FILE = REPO_ROOT / "forward_engine" / "system2_poc.py"
ANSI_BOLD_YELLOW = "\033[1;93m"
ANSI_BRIGHT_BLUE = "\033[94m"
ANSI_RESET = "\033[0m"


def _use_color() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _bold_yellow(text: str) -> str:
    return f"{ANSI_BOLD_YELLOW}{text}{ANSI_RESET}" if _use_color() else text


def _bright_blue(text: str) -> str:
    return f"{ANSI_BRIGHT_BLUE}{text}{ANSI_RESET}" if _use_color() else text


def _format_rule_text(text: str | None) -> str:
    if not text:
        return "<empty>"
    return str(text).replace(" and ", " ∧ ").replace(chr(10), " ").strip()


def _extract_experta_rules(path: Path) -> list[dict]:
    if not path.exists():
        return []
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    extracted: list[dict] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        has_rule_decorator = False
        patterns: list[str] = []
        action_lines: list[str] = []
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call):
                continue
            if isinstance(dec.func, ast.Name) and dec.func.id == "Rule":
                has_rule_decorator = True
                for arg in dec.args:
                    segment = ast.get_source_segment(src, arg) or ""
                    line = " ".join(segment.split())
                    if line:
                        patterns.append(line)
        if not has_rule_decorator:
            continue

        body_nodes = list(node.body)
        if body_nodes and isinstance(body_nodes[0], ast.Expr):
            value = getattr(body_nodes[0], "value", None)
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                body_nodes = body_nodes[1:]

        for stmt in body_nodes:
            segment = ast.get_source_segment(src, stmt) or ""
            cleaned = " ".join(line.strip() for line in segment.splitlines() if line.strip())
            if cleaned:
                action_lines.append(cleaned)

        extracted.append(
            {
                "name": node.name,
                "doc": (ast.get_docstring(node) or "").strip(),
                "patterns": patterns,
                "actions": action_lines,
            }
        )
    return extracted


def _print_rule_dict(rule: dict, label: str, *, show_body: bool = True) -> None:
    if not isinstance(rule, dict):
        print(f"  - {label}: {rule}")
        return

    name = rule.get("name") or rule.get("rule_id") or rule.get("id") or rule.get("family") or label
    family = rule.get("family") or "generic"
    print(f"- {_bold_yellow(str(name))} [{family}]")

    template = rule.get("template")
    if template:
        print(f"  template: {_bright_blue(_format_rule_text(template))}")

    raw_rule = rule.get("raw_rule")
    if raw_rule:
        print(f"  raw_rule: {_bright_blue(_format_rule_text(raw_rule))}")

    compiled = rule.get("compiled")
    if isinstance(compiled, dict):
        rule_text = compiled.get("rule_text") or compiled.get("template") or compiled.get("condition")
        if rule_text:
            print(f"  compiled_rule: {_bright_blue(_format_rule_text(str(rule_text)))}")
        for key in ("condition", "effect", "family", "priority", "kind"):
            value = compiled.get(key)
            if value not in (None, ""):
                print(f"  compiled.{key}: {value}")

    if show_body:
        for key in ("head", "body", "calc", "target", "description"):
            value = rule.get(key)
            if value not in (None, ""):
                if key in {"calc", "target"}:
                    print(f"  {key}: {_bright_blue(str(value))}")
                elif key == "description":
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {value}")

    notes = rule.get("notes") or []
    if isinstance(notes, list):
        for note in notes:
            print(f"  note: {note}")

    if not any(v for v in (template, raw_rule, compiled, rule.get("head"), rule.get("body"), rule.get("calc"), rule.get("description"), notes)):
        print("  <no concrete rule body recorded>")


def print_sys2_rules(path: Path) -> None:
    if not path.exists():
        print(f"[missing] {path}")
        return

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    if not payload:
        print(f"[empty] {path}")
        return

    print(f"System 2 rule base: {path}")
    print("=" * 72)
    for i, rule in enumerate(payload, start=1):
        _print_rule_dict(rule, f"rule_{i}")
        print()


def print_defeasible_rules(path: Path) -> None:
    if not path.exists():
        print(f"[missing] {path}")
        return
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rules = payload.get("rules") or []
    priorities = payload.get("priorities") or []
    if not rules and not priorities:
        print(f"[empty] {path}")
        return

    print(f"Defeasible rule base: {path}")
    print("=" * 72)
    for rule in rules:
        rid = rule.get("id", "unnamed")
        kind = rule.get("kind", "defeasible")
        expr = str(rule.get("rule", "")).replace(" and ", " ∧ ").replace("=>", "⟹")
        note = rule.get("notes", "")
        print(f"- {_bold_yellow(rid)} [{kind}]")
        print(f"  rule: {_bright_blue(expr)}")
        if note:
            print(f"  note: {note}")
        print()
    if priorities:
        print("Priorities:")
        for rel in priorities:
            print(f"  - {rel}")
        print()


def print_lt_rules(paths: list[Path]) -> None:
    found = False
    print("Logic Transformer saved rule summaries:")
    print("=" * 72)
    for path in paths:
        if not path.exists():
            print(f"[missing] {path}")
            continue
        found = True
        print(f"--- {path.relative_to(REPO_ROOT)} ---")
        print(path.read_text(encoding="utf-8").strip() or "<empty>")
        print()
    if not found:
        print("No LT rule summaries found yet. Run a demo first.")


def print_nl_rule_registry(path: Path) -> None:
    if not path.exists():
        print(f"[missing] {path}")
        return
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rules = payload.get("rules") or []
    teacher_events = payload.get("teacher_proposals") or []
    print(f"NL rule registry: {path}")
    print("=" * 72)
    if not rules:
        print("<empty>")
    else:
        for entry in rules:
            rid = entry.get("rule_id", "unknown")
            status = entry.get("status", "proposed")
            version = entry.get("version", 1)
            compiled = entry.get("compiled") or {}
            family = compiled.get("family", "unknown")
            provenance = entry.get("provenance", "unknown")
            mutability = entry.get("mutability", "priority_only")
            accounting = entry.get("accounting") or {}
            print(f"- {_bold_yellow(rid)} [status={status}, version={version}, family={family}]")
            print(f"  provenance: {provenance}  mutability: {mutability}")
            print(
                "  accounting: "
                f"applied={accounting.get('applied', 0)} "
                f"wins={accounting.get('wins', 0)} "
                f"defeats={accounting.get('defeats', 0)} "
                f"recall+={accounting.get('recall_accepted', 0)} "
                f"recall-={accounting.get('recall_rejected', 0)}"
            )
            print(f"  source: {entry.get('source', 'unknown')}")
            print(f"  updated_at: {entry.get('updated_at', 'n/a')}")
            raw_rule = entry.get("raw_rule")
            if raw_rule:
                print(f"  raw_rule: {_bright_blue(_format_rule_text(raw_rule))}")
            if isinstance(compiled, dict):
                for key in ("rule_text", "condition", "effect", "priority"):
                    value = compiled.get(key)
                    if value not in (None, ""):
                        print(f"  compiled.{key}: {_bright_blue(_format_rule_text(str(value)))}")
            print()

    print(f"teacher_proposals_logged: {len(teacher_events)}")
    for event in teacher_events[-5:]:
        print(
            f"  - {event.get('proposal_id', 'unknown')} "
            f"kind={event.get('proposal_kind', 'unknown')} "
            f"decision={event.get('decision', 'unknown')} "
            f"ab={event.get('ab_arm', 'n/a')}"
        )


def print_nl_rule_examples(path: Path) -> None:
    if not path.exists():
        print(f"[missing] {path}")
        return
    print(f"NL authoring examples: {path}")
    print("=" * 72)
    print(path.read_text(encoding="utf-8").strip() or "<empty>")
    print()


def print_rule_snapshot(path: Path) -> None:
    if not path.exists():
        print(f"[missing] {path}")
        return
    print(f"Rule snapshot: {path}")
    print("=" * 72)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if "rules" in payload and isinstance(payload["rules"], list):
            rules = payload["rules"]
            print(f"count: {len(rules)}")
            for i, rule in enumerate(rules, start=1):
                _print_rule_dict(rule, f"rule_{i}")
                print()
        elif "snapshots" in payload and isinstance(payload["snapshots"], list):
            snapshots = payload["snapshots"]
            print(f"snapshots: {len(snapshots)}")
            for snap in snapshots[:3]:
                label = snap.get("label", "snapshot")
                tracked = snap.get("tracked_rule_count", 0)
                learning = snap.get("rules_under_learning_count", 0)
                print(f"- {label}: tracked={tracked} learning={learning}")
                top_rules = snap.get("top_rules") or []
                if top_rules:
                    for item in top_rules[:5]:
                        print(
                            "    top_rule: "
                            f"signature={item.get('signature', 'unknown')} "
                            f"count={item.get('count', 0)} "
                            f"avg_reward={item.get('avg_reward', 0)}"
                        )
                if not top_rules:
                    print("    <no concrete rule bodies recorded in this snapshot>")
        else:
            print(payload)
    else:
        print(payload)
    print()


def print_ltm_summary(path: Path) -> None:
    if not path.exists():
        print(f"[missing] {path}")
        print("LTM store is not yet persisted as a concrete YAML file; snapshot summaries are therefore partial-only.")
        return
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    episodes = payload.get("episodes") or []
    semantic = payload.get("semantic_memory") or []
    rules = payload.get("rule_memory") or []
    active_episodes = [e for e in episodes if not e.get("archived")]
    print(f"LTM store: {path}")
    print("=" * 72)
    print(f"episodes_total: {len(episodes)}")
    print(f"episodes_active: {len(active_episodes)}")
    print(f"semantic_memory_entries: {len(semantic)}")
    print(f"rule_memory_entries: {len(rules)}")


def print_sys2_experta_rules(path: Path) -> None:
    if not path.exists():
        print(f"[missing] {path}")
        return
    rules = _extract_experta_rules(path)
    print(f"System 2 Experta rules (code-defined): {path}")
    print("=" * 72)
    print(f"count: {len(rules)}")
    for row in rules:
        print(f"- {_bold_yellow(row['name'])} [experta]")
        for pat in row.get("patterns") or []:
            print(f"  when: {_bright_blue(_format_rule_text(pat))}")
        for action in row.get("actions") or []:
            print(f"  action: {_bright_blue(_format_rule_text(action))}")
        doc = str(row.get("doc") or "").strip()
        if doc:
            one_line = " ".join(doc.split())
            print(f"  doc: {one_line}")
        print()


def _count_rule_entries(path: Path) -> tuple[int, int, str]:
    if not path.exists():
        return 0, 0, "missing"
    try:
        if path.suffix.lower() in {".yaml", ".yml"}:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if isinstance(payload, list):
                return len(payload), 0, "list"
            if isinstance(payload, dict):
                rule_count = len(payload.get("rules") or [])
                proposal_count = len(payload.get("teacher_proposals") or [])
                return rule_count, proposal_count, "dict"
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                if "rules" in payload:
                    return len(payload.get("rules") or []), 0, "json_rules"
                if "snapshots" in payload:
                    tracked = []
                    for snap in payload.get("snapshots") or []:
                        tracked.extend(snap.get("top_rules") or [])
                    return len(payload.get("snapshots") or []), len(tracked), "json_snapshots"
    except Exception:
        return 0, 0, "parse_error"
    return 0, 0, "empty"


def print_rule_inventory_summary() -> None:
    sys2_rules, _, _ = _count_rule_entries(SYS2_RULES)
    defeasible_rules, _, _ = _count_rule_entries(DEFEASIBLE_RULES)
    registry_rules, registry_proposals, _ = _count_rule_entries(NL_RULE_REGISTRY)
    metarules, _, _ = _count_rule_entries(RULE_SNAPSHOT_PATHS[0])
    learning_snapshots, learning_candidates, _ = _count_rule_entries(RULE_SNAPSHOT_PATHS[1])
    experta_rules = len(_extract_experta_rules(SYS2_EXPERTA_FILE))
    active_runtime = sys2_rules + defeasible_rules + registry_rules
    total_materialized = sys2_rules + defeasible_rules + registry_rules + metarules + learning_candidates
    print("Rule inventory summary:")
    print("=" * 72)
    print(f"System 2 file-backed rules: {sys2_rules}")
    print(f"Defeasible rules: {defeasible_rules}")
    print(f"NL registry compiled rules: {registry_rules}")
    print(f"NL registry proposal records: {registry_proposals}")
    print(f"Metarule snapshot entries: {metarules}")
    print(f"Learning snapshot tracked groups: {learning_snapshots}")
    print(f"Learning snapshot candidate signatures: {learning_candidates}")
    print(f"System 2 Experta code-defined rules: {experta_rules}")
    print(f"Active runtime KB (file-backed + compiled): {active_runtime}")
    print(f"Operational total incl. Experta code rules: {active_runtime + experta_rules}")
    print(f"Total materialized rule/metarule count: {total_materialized}")
    if not LTM_STORE.exists():
        print("LTM store: missing (0 persisted rule-memory entries)")
    else:
        ltm_rules, _, _ = _count_rule_entries(LTM_STORE)
        print(f"LTM rule-memory entries: {ltm_rules}")
    print()


def print_sys3_rules_summary() -> None:
    print("System 3 rule base: not yet a separate persisted file.")
    print("Current implementation lives in:")
    for path in (SYS3_SEED_FILE, SYS3_SEED_FILE_2):
        print(f"  - {path.relative_to(REPO_ROOT)}")
    print()
    print("It currently consists of:")
    print("  - metarule grammar and family selectors in ilp_learner/metagol_nds.py")
    print("  - default symbolic templates in ilp_learner/abductive_ilp.py")
    print("  - online rule generation in the System 3 reasoning path")
    print("This is a code-defined metarule base, not a dedicated YAML/JSON rule KB yet.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Print persisted AGI rule sets.")
    parser.add_argument("--sys2", action="store_true", help="Print the file-backed Sys-2 rule KB.")
    parser.add_argument("--defeasible", action="store_true", help="Print the defeasible rule KB and priorities.")
    parser.add_argument("--registry", action="store_true", help="Print the NL rule registry with lifecycle status.")
    parser.add_argument("--nl", action="store_true", help="Print authoring examples from the NL rule text file.")
    parser.add_argument("--snapshot", action="store_true", help="Print metarule/rule-learning snapshots in JSON reports.")
    parser.add_argument("--ltm", action="store_true", help="Print a summary of the LTM store.")
    parser.add_argument("--lt", action="store_true", help="Print saved LT rule summaries from logic_transformer/.")
    parser.add_argument("--experta", action="store_true", help="Print code-defined System-2 Experta @Rule methods.")
    parser.add_argument("--sys3", action="store_true", help="Print the current Sys-3 rule base status.")
    parser.add_argument("--all", action="store_true", help="Print all available persisted rule sources.")
    args = parser.parse_args()

    if not any([args.sys2, args.defeasible, args.registry, args.nl, args.snapshot, args.ltm, args.lt, args.experta, args.sys3, args.all]):
        args.all = True

    print_rule_inventory_summary()
    if args.all or args.sys2:
        print_sys2_rules(SYS2_RULES)
        print()
    if args.all or args.defeasible:
        print_defeasible_rules(DEFEASIBLE_RULES)
        print()
    if args.all or args.registry:
        print_nl_rule_registry(NL_RULE_REGISTRY)
        print()
    if args.all or args.nl:
        print_nl_rule_examples(NL_RULE_EXAMPLES)
        print()
    if args.all or args.snapshot:
        for path in RULE_SNAPSHOT_PATHS:
            print_rule_snapshot(path)
    if args.all or args.ltm:
        print_ltm_summary(LTM_STORE)
        print()
    if args.all or args.lt:
        print_lt_rules(LT_RULE_CANDIDATES)
    if args.all or args.experta:
        print_sys2_experta_rules(SYS2_EXPERTA_FILE)
        print()
    if args.all or args.sys3:
        print_sys3_rules_summary()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
