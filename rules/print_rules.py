#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SYS2_RULES = REPO_ROOT / "rules" / "sys2_rule_kb.yaml"
DEFEASIBLE_RULES = REPO_ROOT / "rules" / "defeasible_rule_kb.yaml"
NL_RULE_REGISTRY = REPO_ROOT / "rules" / "nl_rule_registry.yaml"
LTM_STORE = REPO_ROOT / "rules" / "ltm_store.yaml"
LT_RULE_CANDIDATES = [
    REPO_ROOT / "logic_transformer" / "semantic_wme_rules.txt",
    REPO_ROOT / "logic_transformer" / "ratio_rules.txt",
]
SYS3_SEED_FILE = REPO_ROOT / "ilp_learner" / "metagol_nds.py"
SYS3_SEED_FILE_2 = REPO_ROOT / "ilp_learner" / "abductive_ilp.py"


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
    for rule in payload:
        name = rule.get("name", "unnamed")
        family = rule.get("family", "generic")
        template = (rule.get("template") or "").strip()
        notes = rule.get("notes") or []
        pretty = template.replace(" and ", " ∧ ")
        print(f"- {name} [{family}]")
        print(f"  template: {pretty.replace(chr(10), ' ')}")
        for note in notes:
            print(f"    note: {note}")
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
        print(f"- {rid} [{kind}]")
        print(f"  rule: {expr}")
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
    print(f"NL rule registry: {path}")
    print("=" * 72)
    if not rules:
        print("<empty>")
        return
    for entry in rules:
        rid = entry.get("rule_id", "unknown")
        status = entry.get("status", "proposed")
        version = entry.get("version", 1)
        family = (entry.get("compiled") or {}).get("family", "unknown")
        provenance = entry.get("provenance", "unknown")
        mutability = entry.get("mutability", "priority_only")
        accounting = entry.get("accounting") or {}
        print(f"- {rid} [status={status}, version={version}, family={family}]")
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
        print()


def print_ltm_summary(path: Path) -> None:
    if not path.exists():
        print(f"[missing] {path}")
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
    parser.add_argument("--ltm", action="store_true", help="Print a summary of the LTM store.")
    parser.add_argument("--lt", action="store_true", help="Print saved LT rule summaries from logic_transformer/.")
    parser.add_argument("--sys3", action="store_true", help="Print the current Sys-3 rule base status.")
    parser.add_argument("--all", action="store_true", help="Print all available persisted rule sources.")
    args = parser.parse_args()

    if not any([args.sys2, args.defeasible, args.registry, args.ltm, args.lt, args.sys3, args.all]):
        args.all = True

    if args.all or args.sys2:
        print_sys2_rules(SYS2_RULES)
        print()
    if args.all or args.defeasible:
        print_defeasible_rules(DEFEASIBLE_RULES)
        print()
    if args.all or args.registry:
        print_nl_rule_registry(NL_RULE_REGISTRY)
        print()
    if args.all or args.ltm:
        print_ltm_summary(LTM_STORE)
        print()
    if args.all or args.lt:
        print_lt_rules(LT_RULE_CANDIDATES)
    if args.all or args.sys3:
        print_sys3_rules_summary()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
