#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "rules" / "configs" / "corpus_sources_v1.json"
DEFAULT_OUTPUT = REPO_ROOT / "rules" / "reports" / "corpus_manifest_v1.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON at {path}")
    return payload


@dataclass(frozen=True)
class HFFile:
    path: str
    size: int


def _matches_any(path: str, globs: list[str]) -> bool:
    if not globs:
        return True
    return any(fnmatch.fnmatch(path, pattern) for pattern in globs)


def _list_repo_files_with_sizes(*, api: HfApi, repo_id: str, repo_type: str) -> list[HFFile]:
    nodes = api.list_repo_tree(repo_id=repo_id, repo_type=repo_type, recursive=True, expand=True)
    files: list[HFFile] = []
    for node in nodes:
        path = getattr(node, "path", None)
        node_type = getattr(node, "type", None)
        size = getattr(node, "size", None)
        if not path or str(node_type) != "file":
            continue
        if size is None:
            continue
        files.append(HFFile(path=str(path), size=int(size)))
    return files


def _select_files(
    *,
    files: list[HFFile],
    include_globs: list[str],
    exclude_globs: list[str],
    target_bytes: int,
) -> tuple[list[HFFile], int]:
    selected: list[HFFile] = []
    total = 0
    for item in sorted(files, key=lambda row: (row.path, row.size)):
        if not _matches_any(item.path, include_globs):
            continue
        if exclude_globs and _matches_any(item.path, exclude_globs):
            continue
        if item.size <= 0:
            continue
        if total + item.size > target_bytes and selected:
            break
        selected.append(item)
        total += item.size
    return selected, total


def _resolve_output(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a size-capped HF corpus manifest split across two disks.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--disk1-root", default="/media/yky/old-dreams1")
    parser.add_argument("--disk2-root", default="/media/yky/old-dreams2")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--hf-token", default="", help="Optional Hugging Face token.")
    args = parser.parse_args()

    config_path = _resolve_output(args.config)
    output_path = _resolve_output(args.output)
    cfg = _load_json(config_path)
    sources = cfg.get("sources") if isinstance(cfg.get("sources"), list) else []
    if not sources:
        raise ValueError("No sources configured.")

    root_map = {
        "disk1": str(args.disk1_root),
        "disk2": str(args.disk2_root),
    }

    api = HfApi(token=(args.hf_token or None))
    manifest_entries: list[dict[str, Any]] = []
    totals_by_root = {"disk1": 0, "disk2": 0}
    total_selected = 0

    for src in sources:
        if not isinstance(src, dict):
            continue
        source_id = str(src.get("source_id") or "")
        repo_id = str(src.get("repo_id") or "")
        repo_type = str(src.get("repo_type") or "dataset")
        target_root_key = str(src.get("target_root") or "disk1")
        include_globs = [str(x) for x in (src.get("include_globs") or [])]
        exclude_globs = [str(x) for x in (src.get("exclude_globs") or [])]
        target_gb = float(src.get("target_gb") or 0.0)
        target_bytes = max(1, int(target_gb * (1024 ** 3)))
        if not source_id or not repo_id:
            continue
        if target_root_key not in root_map:
            raise ValueError(f"Unknown target_root '{target_root_key}' for source {source_id}")

        files = _list_repo_files_with_sizes(api=api, repo_id=repo_id, repo_type=repo_type)
        selected, selected_bytes = _select_files(
            files=files,
            include_globs=include_globs,
            exclude_globs=exclude_globs,
            target_bytes=target_bytes,
        )
        target_root = root_map[target_root_key]
        totals_by_root[target_root_key] += selected_bytes
        total_selected += selected_bytes
        manifest_entries.append(
            {
                "source_id": source_id,
                "repo_id": repo_id,
                "repo_type": repo_type,
                "target_root_key": target_root_key,
                "target_root_path": target_root,
                "target_gb": target_gb,
                "selected_gb": round(selected_bytes / (1024 ** 3), 3),
                "selected_count": len(selected),
                "files": [{"path": item.path, "size": item.size} for item in selected],
            }
        )

    manifest = {
        "manifest_id": str(cfg.get("manifest_id") or "corpus-manifest"),
        "generated_at": _utc_now(),
        "config_path": str(config_path),
        "target_total_gb": float(cfg.get("target_total_gb") or 0),
        "roots": root_map,
        "entries": manifest_entries,
        "summary": {
            "total_sources": len(manifest_entries),
            "selected_total_gb": round(total_selected / (1024 ** 3), 3),
            "selected_disk1_gb": round(totals_by_root["disk1"] / (1024 ** 3), 3),
            "selected_disk2_gb": round(totals_by_root["disk2"] / (1024 ** 3), 3),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved corpus manifest to {output_path}")
    print(json.dumps(manifest["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
