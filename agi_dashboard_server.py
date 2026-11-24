#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import yaml


REPO_ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = REPO_ROOT / "runtime" / "dashboard"
PROCESS_STATE_PATH = RUNTIME_DIR / "process.json"
PROCESS_LOG_PATH = RUNTIME_DIR / "process.log"
DASHBOARD_PATH = REPO_ROOT / "agi_dashboard.html"
DASHBOARD_JSON_PATH = REPO_ROOT / "rules" / "reports" / "agi_progress_dashboard_v1.json"
SHARED_PYTHON = REPO_ROOT.parent / ".venv" / "bin" / "python"
PYTHON = SHARED_PYTHON if SHARED_PYTHON.exists() else Path(sys.executable)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_repo_path(raw: Any, *, must_exist: bool = False, allow_empty: bool = False) -> Path:
    text = str(raw or "").strip()
    if not text:
        if allow_empty:
            return REPO_ROOT
        raise ValueError("path is required")
    path = Path(text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    resolved = path.resolve()
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"path must stay inside the repository: {text}") from exc
    if must_exist and not resolved.exists():
        raise ValueError(f"path does not exist: {text}")
    return resolved


def _as_bool(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: Any, name: str, default: int, minimum: int = 0, maximum: int | None = None) -> int:
    try:
        result = int(value if value not in (None, "") else default)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result < minimum or (maximum is not None and result > maximum):
        raise ValueError(f"{name} is outside the allowed range")
    return result


def build_launch_command(payload: dict[str, Any]) -> list[str]:
    mode = str(payload.get("mode") or "").strip()
    if mode == "lt_train":
        dataset = _safe_repo_path(payload.get("dataset"), must_exist=True)
        checkpoint = _safe_repo_path(payload.get("checkpoint") or "checkpoints/lt_tinystories.pt")
        epochs = _as_int(payload.get("epochs"), "epochs", 20, 1)
        holdout = float(payload.get("holdout", 0.2) or 0.0)
        if not 0.0 <= holdout < 1.0:
            raise ValueError("holdout must be >= 0 and < 1")
        max_minutes = float(payload.get("max_minutes", 0) or 0.0)
        if max_minutes < 0:
            raise ValueError("max_minutes must be non-negative")
        command = [
            str(PYTHON),
            str(REPO_ROOT / "logic_transformer" / "train.py"),
            "--data-path",
            str(dataset),
            "--epochs",
            str(epochs),
            "--holdout-ratio",
            str(holdout),
            "--checkpoint",
            str(checkpoint),
            "--threads",
            str(_as_int(payload.get("threads"), "threads", 4, 1, 64)),
            "--seed",
            str(_as_int(payload.get("seed"), "seed", 0, 0)),
        ]
        if max_minutes > 0:
            command += ["--max-minutes", str(max_minutes)]
        if _as_bool(payload.get("resume")):
            command.append("--resume")
        return command

    if mode == "cognitive_loop":
        allowed_scenarios = {
            "dataset": "dataset",
            "family": "--family-benchmark",
            "finality": "--finality-regression",
            "template": "--template-generalization",
        }
        scenario = str(payload.get("scenario") or "dataset").strip()
        if scenario not in allowed_scenarios:
            raise ValueError("unsupported cognitive-loop scenario")
        command = [
            str(PYTHON),
            str(REPO_ROOT / "cognitive_loop.py"),
            "--limit",
            str(_as_int(payload.get("limit"), "limit", 5, 1)),
            "--cycles",
            str(_as_int(payload.get("cycles"), "cycles", 4, 1, 100)),
        ]
        if scenario == "dataset":
            dataset = _safe_repo_path(payload.get("dataset"), must_exist=True)
            command += ["--data-path", str(dataset)]
        else:
            command.append(allowed_scenarios[scenario])
        if _as_bool(payload.get("sar_enabled")):
            command.append("--sar-enabled")
        if _as_bool(payload.get("sar_emit_metadata")):
            command.append("--sar-emit-metadata")
        if _as_bool(payload.get("quiet")):
            command.append("--quiet")
        return command

    if mode == "bootstrap":
        suites = {
            "template_generalization",
            "finality_regression",
            "tinystories_state_tracking",
            "mixed_narrative_symbolic",
            "combined",
            "combined_widening",
        }
        profiles = {"lt-heavy", "hybrid", "ilp-heavy"}
        suite = str(payload.get("suite") or "tinystories_state_tracking")
        profile = str(payload.get("profile") or "lt-heavy")
        if suite not in suites:
            raise ValueError("unsupported bootstrap suite")
        if profile not in profiles:
            raise ValueError("unsupported bootstrap profile")
        output = _safe_repo_path(
            payload.get("output") or "rules/reports/agi_dashboard_run.json"
        )
        command = [
            str(PYTHON),
            str(REPO_ROOT / "rules" / "run_bootstrap_baseline.py"),
            "--suite",
            suite,
            "--profiles",
            profile,
            "--max-cycles",
            str(_as_int(payload.get("cycles"), "cycles", 2, 1, 100)),
            "--limit",
            str(_as_int(payload.get("limit"), "limit", 0, 0)),
            "--output",
            str(output),
        ]
        if _as_bool(payload.get("sar_enabled")):
            command.append("--sar-enabled")
        if _as_bool(payload.get("ilp_enabled")):
            command += ["--ilp-enable-learning-stage", "--ilp-large-pool-mode"]
        if _as_bool(payload.get("activate_pool")):
            command.append("--ilp-activate-generated-pool-rules")
        if _as_bool(payload.get("quiet")):
            command.append("--quiet")
        return command

    raise ValueError("mode must be one of: lt_train, cognitive_loop, bootstrap")


class ProcessManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen[Any] | None = None
        self._log_handle = None
        self._state: dict[str, Any] = self._load_state()
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

    def _load_state(self) -> dict[str, Any]:
        if not PROCESS_STATE_PATH.exists():
            return {"status": "idle", "pid": None}
        try:
            state = json.loads(PROCESS_STATE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"status": "idle", "pid": None}
        if state.get("status") in {"running", "stopping"}:
            state["status"] = "stale"
            state["stale_reason"] = "dashboard server restarted; process handle is unavailable"
        return state

    def _save_state(self) -> None:
        PROCESS_STATE_PATH.write_text(json.dumps(self._state, indent=2), encoding="utf-8")

    def _refresh(self) -> None:
        process = self._process
        if process is None:
            return
        code = process.poll()
        if code is None:
            return
        with self._lock:
            if self._state.get("status") in {"running", "stopping"}:
                self._state["finished_at"] = _utc_now()
                self._state["exit_code"] = code
                if self._state.get("stop_requested_at"):
                    self._state["status"] = "stopped"
                else:
                    self._state["status"] = "exited" if code == 0 else "failed"
                self._save_state()
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
        self._process = None

    def _stale_managed_pid(self) -> int | None:
        if self._state.get("status") not in {"stale", "stopping"}:
            return None
        pid = int(self._state.get("pid") or 0)
        command = self._state.get("command") or []
        if pid <= 0 or not command:
            return None
        cmdline_path = Path(f"/proc/{pid}/cmdline")
        try:
            live = cmdline_path.read_bytes().split(b"\0")
            live = [part.decode("utf-8", errors="replace") for part in live if part]
        except OSError:
            return None
        expected = [str(part) for part in command[:2]]
        return pid if live[: len(expected)] == expected else None

    def status(self) -> dict[str, Any]:
        self._refresh()
        state = dict(self._state)
        if state.get("started_at"):
            try:
                start = datetime.fromisoformat(state["started_at"])
                end_text = state.get("finished_at")
                end = datetime.fromisoformat(end_text) if end_text else datetime.now(timezone.utc)
                state["elapsed_sec"] = round((end - start).total_seconds(), 1)
            except (TypeError, ValueError):
                pass
        log_payload = self.log_tail(300)
        live_progress = {}
        for line in reversed(log_payload["lines"]):
            match = re.search(r"Epoch\s+(\d+)", line)
            if match:
                live_progress["epoch"] = int(match.group(1))
                for key, value in re.findall(
                    r"([a-zA-Z0-9_]+)=(-?\d+(?:\.\d+)?)", line
                ):
                    live_progress[key] = float(value)
                live_progress["source_line"] = line
                break
        if live_progress:
            state["live_progress"] = live_progress
        return state

    def launch(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._refresh()
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise ValueError("a managed AGI process is already running")
            if self._stale_managed_pid() is not None:
                raise ValueError(
                    "a previously managed process is still running after dashboard restart; "
                    "use the stop control first"
                )
            command = build_launch_command(payload)
            RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
            mode = str(payload.get("mode") or "")
            self._log_handle = PROCESS_LOG_PATH.open("w", encoding="utf-8", buffering=1)
            self._log_handle.write(f"[dashboard] {_utc_now()} launching mode={mode}\n")
            self._log_handle.write("[dashboard] command: " + " ".join(command) + "\n")
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            self._process = process
            self._state = {
                "status": "running",
                "mode": mode,
                "pid": process.pid,
                "started_at": _utc_now(),
                "command": command,
                "log_path": str(PROCESS_LOG_PATH.relative_to(REPO_ROOT)),
            }
            self._save_state()
            return dict(self._state)

    def stop(self) -> dict[str, Any]:
        self._refresh()
        process = self._process
        stale_pid = self._stale_managed_pid() if process is None else None
        if (process is None or process.poll() is not None) and stale_pid is None:
            raise ValueError("no managed process is currently running")
        with self._lock:
            self._state["status"] = "stopping"
            self._state["stop_requested_at"] = _utc_now()
            self._save_state()
        if process is None and stale_pid is not None:
            os.killpg(stale_pid, signal.SIGTERM)
            for _ in range(100):
                if self._stale_managed_pid() is None:
                    break
                time.sleep(0.1)
            if self._stale_managed_pid() is not None:
                os.killpg(stale_pid, signal.SIGKILL)
            self._state["status"] = "stopped"
            self._state["finished_at"] = _utc_now()
            self._state["exit_code"] = None
            self._save_state()
            return self.status()
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        self._refresh()
        return self.status()

    @staticmethod
    def log_tail(limit: int = 300) -> dict[str, Any]:
        if not PROCESS_LOG_PATH.exists():
            return {"lines": [], "updated_at": None, "truncated": False}
        lines = PROCESS_LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
        bounded = max(20, min(2000, int(limit)))
        return {
            "lines": lines[-bounded:],
            "updated_at": datetime.fromtimestamp(
                PROCESS_LOG_PATH.stat().st_mtime, timezone.utc
            ).isoformat(),
            "truncated": len(lines) > bounded,
        }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_rule_report(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        return [row for row in payload.get("rules", []) if isinstance(row, dict)]
    return []


def _rules_payload() -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location(
        "print_rules_dashboard", REPO_ROOT / "rules" / "print_rules.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load rules/print_rules.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    experta = module._extract_experta_rules(REPO_ROOT / "forward_engine" / "system2_poc.py")
    canonical = _load_rule_report(REPO_ROOT / "rules" / "sys2_rule_kb.yaml")
    defeasible = _load_rule_report(REPO_ROOT / "rules" / "defeasible_rule_kb.yaml")
    registry = _load_rule_report(REPO_ROOT / "rules" / "nl_rule_registry.yaml")
    metarules = _load_rule_report(
        REPO_ROOT / "rules" / "reports" / "metarules_snapshot_v1.json"
    )
    return {
        "counts": {
            "experta": len(experta),
            "canonical_sys2": len(canonical),
            "defeasible": len(defeasible),
            "registry": len(registry),
            "metarules": len(metarules),
        },
        "experta": experta,
        "canonical_sys2": canonical,
        "defeasible": defeasible,
        "registry": registry,
        "metarules": metarules,
    }


def _dataset_payload() -> dict[str, Any]:
    roots = [REPO_ROOT / "data", REPO_ROOT / "logic_transformer" / "data", REPO_ROOT / "rules" / "reports"]
    rows = []
    seen: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.jsonl")):
            rel = str(path.relative_to(REPO_ROOT))
            if rel in seen:
                continue
            seen.add(rel)
            rows.append(
                {
                    "path": rel,
                    "bytes": path.stat().st_size,
                    "modified_at": datetime.fromtimestamp(
                        path.stat().st_mtime, timezone.utc
                    ).isoformat(),
                }
            )
    return {
        "datasets": rows,
        "accepted_extensions": [".jsonl"],
        "checkpoint_dir": "checkpoints",
    }


def _memory_payload() -> dict[str, Any]:
    paths = {
        "ltm": REPO_ROOT / "rules" / "ltm_store.yaml",
        "ontology": REPO_ROOT / "rules" / "reports" / "unified_ontology_v1.json",
        "alignment": REPO_ROOT
        / "rules"
        / "reports"
        / "semantic_ar_worked_examples_ontology_aligned_v1.json",
    }
    out: dict[str, Any] = {}
    for key, path in paths.items():
        out[key] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
        }
    ontology = _load_json(paths["ontology"])
    if ontology:
        out["ontology"]["summary"] = {
            "keys": sorted(ontology.keys()),
            "concept_count": len(ontology.get("concepts") or ontology.get("lemma_index") or {}),
        }
    return out


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, manager: ProcessManager, **kwargs):
        self.manager = manager
        super().__init__(*args, directory=str(REPO_ROOT), **kwargs)

    def _json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 256 * 1024:
            raise ValueError("request body must be a JSON object under 256KB")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/dashboard":
            self._json(_load_json(DASHBOARD_JSON_PATH))
        elif parsed.path in {"/api/status", "/api/processes"}:
            self._json(self.manager.status())
        elif parsed.path == "/api/logs":
            query = parse_qs(parsed.query)
            limit = _as_int(query.get("limit", [300])[0], "limit", 300, 20, 2000)
            self._json(self.manager.log_tail(limit))
        elif parsed.path == "/api/rules":
            self._json(_rules_payload())
        elif parsed.path == "/api/datasets":
            self._json(_dataset_payload())
        elif parsed.path == "/api/memory":
            self._json(_memory_payload())
        elif parsed.path in {"/", "/index.html"}:
            self.path = "/agi_dashboard.html"
            super().do_GET()
        else:
            super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/launch":
                self._json(self.manager.launch(self._read_json()), HTTPStatus.ACCEPTED)
            elif parsed.path == "/api/stop":
                self._json(self.manager.stop())
            elif parsed.path == "/api/refresh":
                subprocess.run(
                    [
                        str(PYTHON),
                        str(REPO_ROOT / "rules" / "generate_agi_progress_dashboard.py"),
                        "--output-json",
                        str(DASHBOARD_JSON_PATH),
                        "--output-html",
                        str(RUNTIME_DIR / "refreshed_dashboard.html"),
                    ],
                    cwd=REPO_ROOT,
                    check=True,
                    timeout=120,
                )
                self._json(_load_json(DASHBOARD_JSON_PATH))
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except (ValueError, FileNotFoundError, subprocess.SubprocessError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(f"[dashboard] {self.log_date_time_string()} " + fmt % args + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the live AGI dashboard and training controls.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print(
            "WARNING: non-loopback binding exposes process controls to the network. "
            "Use SSH port forwarding and keep the default host for private operation.",
            file=sys.stderr,
        )
    if not DASHBOARD_PATH.exists():
        raise FileNotFoundError(f"missing root dashboard: {DASHBOARD_PATH}")
    manager = ProcessManager()

    def handler(*handler_args, **handler_kwargs):
        DashboardHandler(*handler_args, manager=manager, **handler_kwargs)

    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"AGI dashboard serving on http://{args.host}:{args.port}/")
    print("Remote/private use: ssh -L 8765:127.0.0.1:8765 user@server")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
