from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
RULES_DIR = REPO_ROOT / "rules"
if str(RULES_DIR) not in sys.path:
    sys.path.insert(0, str(RULES_DIR))

from run_metarule_cycle import register_candidates
from working_memory.rule_registry import RuleRegistry


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def main() -> None:
    registry = RuleRegistry.default()
    before = _read(registry.path)
    try:
        recs = register_candidates(top_n=1, activate=False)
        assert recs, "Expected at least one metarule registration"
        rules = registry.list_rules()
        empirical = [r for r in rules if str(r.get("source")) == "metarule"]
        assert empirical, "Expected empirical metarule entries in registry"
        for e in empirical:
            assert e.get("provenance") == "empirical"
            assert e.get("mutability") in {"priority_only", "fully_mutable", "locked"}
    finally:
        if before == "":
            if registry.path.exists():
                registry.path.unlink()
        else:
            registry.path.write_text(before, encoding="utf-8")

    print("Metarule loop validation passed.")


if __name__ == "__main__":
    main()
