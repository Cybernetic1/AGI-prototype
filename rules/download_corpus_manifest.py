#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "rules" / "reports" / "corpus_manifest_v1.json"
DEFAULT_REPORT = REPO_ROOT / "rules" / "reports" / "corpus_download_report_v1.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON at {path}")
    return payload


def _resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p)


def _sha256(path: Path, chunk_size: int = 2 ** 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Download corpus files listed in a prepared HF manifest.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--state-file", default="rules/reports/corpus_download_state_v1.json")
    parser.add_argument("--hf-token", default="", help="Optional Hugging Face token.")
    parser.add_argument("--only-source", default="", help="Optional source_id filter.")
    parser.add_argument("--max-files", type=int, default=0, help="Optional cap for smoke runs.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-sha256", action="store_true")
    args = parser.parse_args()

    manifest_path = _resolve_path(args.manifest)
    report_path = _resolve_path(args.report)
    state_path = _resolve_path(args.state_file)
    manifest = _load_json(manifest_path)
    entries = manifest.get("entries") if isinstance(manifest.get("entries"), list) else []
    token = args.hf_token or None

    if state_path.exists():
        state = _load_json(state_path)
    else:
        state = {"completed": {}, "updated_at": _utc_now()}

    completed = state.setdefault("completed", {})
    report_rows: list[dict[str, Any]] = []
    downloaded = skipped = failed = 0
    processed = 0
    max_files = int(args.max_files or 0)

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        source_id = str(entry.get("source_id") or "")
        if args.only_source and source_id != args.only_source:
            continue
        repo_id = str(entry.get("repo_id") or "")
        repo_type = str(entry.get("repo_type") or "dataset")
        target_root = Path(str(entry.get("target_root_path") or "")).expanduser()
        files = entry.get("files") if isinstance(entry.get("files"), list) else []
        for row in files:
            if not isinstance(row, dict):
                continue
            remote_path = str(row.get("path") or "")
            size = int(row.get("size") or 0)
            if not remote_path:
                continue
            if max_files > 0 and processed >= max_files:
                break
            processed += 1
            key = f"{source_id}:{remote_path}"
            if completed.get(key):
                skipped += 1
                report_rows.append({"source_id": source_id, "path": remote_path, "status": "skipped_completed"})
                continue
            local_dir = target_root / source_id
            local_file = local_dir / remote_path
            local_file.parent.mkdir(parents=True, exist_ok=True)

            if args.dry_run:
                report_rows.append(
                    {
                        "source_id": source_id,
                        "path": remote_path,
                        "status": "dry_run",
                        "target": str(local_file),
                        "size": size,
                    }
                )
                continue

            try:
                hf_hub_download(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    filename=remote_path,
                    token=token,
                    local_dir=str(local_dir),
                    local_dir_use_symlinks=False,
                    force_download=False,
                )
                if not local_file.exists():
                    raise FileNotFoundError(f"expected local file missing: {local_file}")
                on_disk_size = local_file.stat().st_size
                if size > 0 and on_disk_size != size:
                    raise ValueError(f"size mismatch for {local_file}: expected={size} got={on_disk_size}")
                sha = _sha256(local_file) if args.verify_sha256 else ""
                downloaded += 1
                completed[key] = {
                    "local_path": str(local_file),
                    "size": on_disk_size,
                    "sha256": sha,
                    "completed_at": _utc_now(),
                }
                report_rows.append(
                    {
                        "source_id": source_id,
                        "path": remote_path,
                        "status": "downloaded",
                        "size": on_disk_size,
                        "sha256": sha,
                    }
                )
            except Exception as exc:
                failed += 1
                report_rows.append({"source_id": source_id, "path": remote_path, "status": "failed", "error": str(exc)})
        if max_files > 0 and processed >= max_files:
            break

    state["updated_at"] = _utc_now()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    report = {
        "manifest_id": manifest.get("manifest_id"),
        "generated_at": _utc_now(),
        "manifest_path": str(manifest_path),
        "dry_run": bool(args.dry_run),
        "only_source": str(args.only_source),
        "processed_files": processed,
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
        "rows": report_rows,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved download report to {report_path}")
    print(f"processed={processed} downloaded={downloaded} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
