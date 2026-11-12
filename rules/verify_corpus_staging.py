#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _iter_manifest_files(payload: dict[str, Any]) -> list[dict[str, Any]]:
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"Manifest missing 'entries' list: {payload.get('manifest_id')}")
    rows: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        files = entry.get("files")
        if not isinstance(files, list):
            continue
        for row in files:
            if isinstance(row, dict):
                rows.append({**row, "source_id": entry.get("source_id", "unknown"), "target_root_path": entry.get("target_root_path", "")})
    return rows


def _check_file(file_path: Path, expected_size: int | None, source_id: str) -> tuple[bool, str]:
    if not file_path.exists():
        return False, f"missing: {file_path}"
    actual_size = file_path.stat().st_size
    if expected_size is not None and expected_size > 0 and actual_size != expected_size:
        return False, (
            f"size mismatch for {source_id}: expected={expected_size}, actual={actual_size}, file={file_path}"
        )
    return True, f"ok: {file_path} ({actual_size} bytes)"


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify downloaded corpus files match manifest expectations.")
    parser.add_argument("--manifest", default="rules/reports/corpus_manifest_v1.json")
    parser.add_argument("--report", default="rules/reports/corpus_download_report_v1.json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    manifest_path = _resolve(args.manifest)
    report_path = _resolve(args.report)

    try:
        manifest = _load_json(manifest_path)
    except FileNotFoundError:
        print(f"Manifest not found: {manifest_path}")
        print("Run the corpus manifest generator first, then re-run this verifier.")
        return 2

    try:
        report = _load_json(report_path) if report_path.exists() else None
    except Exception:
        report = None

    files = _iter_manifest_files(manifest)
    failures: list[str] = []
    checked = 0
    ok = 0

    for row in files:
        source_id = str(row.get("source_id") or "unknown")
        remote_path = str(row.get("path") or "")
        expected_size = row.get("size")
        target_root = str(row.get("target_root_path") or "")
        if not remote_path:
            continue
        if not target_root:
            failures.append(f"No target_root_path for {source_id}: {remote_path}")
            continue
        local_path = Path(target_root) / source_id / remote_path
        checked += 1
        success, message = _check_file(local_path, int(expected_size) if expected_size is not None else None, source_id)
        if success:
            ok += 1
            if not args.quiet:
                print(message)
        else:
            failures.append(message)

    if report is not None:
        processed = int(report.get("processed_files") or 0)
        downloaded = int(report.get("downloaded") or 0)
        skipped = int(report.get("skipped") or 0)
        failed = int(report.get("failed") or 0)
        print(f"Download report summary: processed={processed} downloaded={downloaded} skipped={skipped} failed={failed}")

    print(f"Manifest verification: checked={checked} ok={ok} failed={len(failures)}")
    if failures:
        print("Verification failures:")
        for item in failures[:20]:
            print(f"  - {item}")
        if len(failures) > 20:
            print(f"  ... {len(failures)-20} more failures")
        return 1

    print("Corpus staging verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
