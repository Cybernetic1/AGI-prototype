from __future__ import annotations

import sys
from pathlib import Path
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from working_memory.working_memory import WMNode, WorkingMemory
from working_memory.rule_registry import RuleRegistry
from working_memory.ltm import LongTermMemory


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="agi_ltm_flow_") as tmp:
        tmp_dir = Path(tmp)
        reg_path = tmp_dir / "registry.yaml"
        ltm_path = tmp_dir / "ltm.yaml"

        # Episode 1
        wm = WorkingMemory(default_ttl=3)
        wm.rule_registry = RuleRegistry(reg_path)
        wm.ltm = LongTermMemory(ltm_path)
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
        wm2.rule_registry = RuleRegistry(reg_path)
        wm2.ltm = LongTermMemory(ltm_path)
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
        wm3.rule_registry = RuleRegistry(reg_path)
        wm3.ltm = LongTermMemory(ltm_path)
        wm3.update({WMNode.create("Query", layer="semantic", source="test", goal="remaining_quantity", args=("Bob", "apples"))})
        recalled = wm3.ltm_associative_recall(top_k=2, min_score=0.01, inject=False)
        props = recalled.get("propositions", [])
        assert props, "Expected recalled propositions for Bob/apples cues"
        assert any((p.get("attrs") or {}).get("item") == "apples" for p in props)

    print("LTM flow validation passed.")


if __name__ == "__main__":
    main()
