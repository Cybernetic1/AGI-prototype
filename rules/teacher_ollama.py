from __future__ import annotations

import json
from typing import Any
from urllib import request


def extract_json_payload(text: str) -> Any:
    raw = str(text or "").strip()
    if not raw:
        return {}
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    return json.loads(raw)


def request_ollama_teacher_proposals(
    *,
    ollama_url: str,
    model: str,
    question: str,
    family: str,
    timeout_sec: int,
) -> tuple[Any, str]:
    if not ollama_url or not model:
        return {}, ""
    system_prompt = (
        "You are a strict symbolic tutor for GSM-style reasoning. "
        "Return JSON only with keys: rules, facts, counterfactuals. "
        "rules: list of {rule_id, raw_rule, score}. "
        "facts: list of {fact:{kind,id,owner,item,qty,phase}, score}. "
        "counterfactuals: list of {hypothesis, if:{...}, then:{...}, score}. "
        "Keep proposals short and high precision."
    )
    user_prompt = (
        f"family: {family}\n"
        f"question: {question}\n"
        "Return 1-3 high-confidence proposals total. JSON only."
    )
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    return request_ollama_chat_json(
        ollama_url=ollama_url,
        model=model,
        payload=payload,
        timeout_sec=timeout_sec,
    )


def request_ollama_chat_json(
    *,
    ollama_url: str,
    model: str,
    payload: dict[str, Any],
    timeout_sec: int,
) -> tuple[Any, str]:
    if not ollama_url or not model:
        return {}, ""
    req = request.Request(
        str(ollama_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=max(3, int(timeout_sec))) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return {}, f"ollama_request_failed: {exc}"
    try:
        decoded = json.loads(body)
        content = (((decoded or {}).get("message") or {}).get("content")) if isinstance(decoded, dict) else ""
        proposals = extract_json_payload(content if isinstance(content, str) else "")
        if isinstance(proposals, (dict, list)):
            return proposals, ""
        return {}, "ollama_invalid_payload_type"
    except Exception as exc:
        return {}, f"ollama_parse_failed: {exc}"
