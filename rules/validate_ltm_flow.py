from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from working_memory.working_memory import WMNode, WorkingMemory


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def main() -> None:
    ltm_path = REPO_ROOT / "rules" / "ltm_store.yaml"
    registry_path = REPO_ROOT / "rules" / "nl_rule_registry.yaml"
    ltm_before = _read(ltm_path)
    registry_before = _read(registry_path)

    try:
        # Episode 1
        wm = WorkingMemory(default_ttl=3)
        wm.update(
            {
                WMNode.create("Quantity", layer="semantic", source="test", owner="Bob", item="apples", qty=10),
                WMNode.create("Query", layer="semantic", source="test", goal="remaining_quantity", args=("Bob", "apples")),
            }
        )
        wm.inject_nl_rule(
            "If a later state exists for the same owner and item, prefer the later state.",
            source="ltm-validator",
            target_status="proposed",
        )
        wm.ltm_write_episode(reward=1.0, outcome="solved")

        # Episode 2
        wm2 = WorkingMemory(default_ttl=3)
        wm2.update(
            {
                WMNode.create("Remaining", layer="reasoning", source="test", owner="Bob", item="apples", qty=7),
                WMNode.create("Query", layer="semantic", source="test", goal="remaining_quantity", args=("Bob", "apples")),
            }
        )
        wm2.ltm_write_episode(reward=0.5, outcome="solved")
        report = wm2.ltm_sleep_maintenance(min_support=1, forget_threshold=0.0, keep_last=1)
        assert report["consolidated"]["semantic_count"] >= 1

        # Episode 3: cue-only WM retrieves earlier Bob/apples memories
        wm3 = WorkingMemory(default_ttl=3)
        wm3.update({WMNode.create("Query", layer="semantic", source="test", goal="remaining_quantity", args=("Bob", "apples"))})
        recalled = wm3.ltm_associative_recall(top_k=2, min_score=0.01, inject=False)
        props = recalled.get("propositions", [])
        assert props, "Expected recalled propositions for Bob/apples cues"
        assert any((p.get("attrs") or {}).get("item") == "apples" for p in props)
    finally:
        if ltm_before == "":
            if ltm_path.exists():
                ltm_path.unlink()
        else:
            ltm_path.write_text(ltm_before, encoding="utf-8")
        if registry_before == "":
            if registry_path.exists():
                registry_path.unlink()
        else:
            registry_path.write_text(registry_before, encoding="utf-8")

    print("LTM flow validation passed.")


if __name__ == "__main__":
    main()
