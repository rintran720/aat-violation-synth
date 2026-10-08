"""Codex usage, with explicit unknown fields and legacy input-only readings."""
import json
import re
from pathlib import Path

FIELDS = ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens")


def count(value):
    return value if type(value) is int and value >= 0 else None


def legacy_usage(value):
    return dict.fromkeys(FIELDS) | {"input_tokens": count(value), "source": "legacy_input" if count(value) is not None else "unknown"}


def read_usage(log: Path) -> dict:
    try:
        text = log.read_text(errors="replace")
    except OSError:
        return dict.fromkeys(FIELDS) | {"source": "unknown"}
    turns = []
    for line in text.splitlines():
        if not line.startswith("{") or '"turn.completed"' not in line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict):
            turns.append({key: count(event["usage"].get(key)) for key in FIELDS})
    if turns:
        return {key: sum(t[key] for t in turns) if all(t[key] is not None for t in turns) else None
                for key in FIELDS} | {"source": "codex_json"}
    matches = re.findall(r"tokens used\s*\n\s*([\d,]+)", text)
    return legacy_usage(int(matches[-1].replace(",", ""))) if matches else dict.fromkeys(FIELDS) | {"source": "unknown"}


def summarize(usages):
    usages = list(usages)
    result = {}
    for key in FIELDS:
        known = [u[key] for u in usages if count(u.get(key)) is not None]
        result[key] = sum(known) if known else None
    result["known_total_tokens"] = sum((u.get("input_tokens") or 0) + (u.get("output_tokens") or 0) for u in usages)
    result["complete"] = all(count(u.get("input_tokens")) is not None and count(u.get("output_tokens")) is not None for u in usages)
    result["attempts_counted"] = len(usages)
    result["unknown_attempts"] = sum(u.get("input_tokens") is None and u.get("output_tokens") is None for u in usages)
    result["legacy_attempts"] = sum(u.get("source") == "legacy_input" for u in usages)
    return result


def migrate_task(task):
    """Idempotent migration; the old scalar becomes input, never inferred output."""
    if "token_attempts" in task:
        return False
    task["token_attempts"] = []
    if "tokens" in task or task.get("status") in ("done", "failed", "interrupted"):
        task["token_attempts"].append({"usage": legacy_usage(task.get("tokens")), "status": task.get("status")})
    update_task(task)
    return True


def update_task(task):
    task["token_usage"] = summarize(a["usage"] for a in task["token_attempts"])
    task["tokens"] = task["token_usage"]["known_total_tokens"]
