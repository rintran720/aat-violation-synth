"""The Codex plan's usage limit as the service sees it, and how many outputs it still leaves room for.

Codex (signed in with a ChatGPT plan, no API key) reports its limit only as a share of a window: the app-server's
read-only `account/rateLimits/read` gives the window (e.g. 10080 min, a week), the percent of it used and when it
resets, and whether ordinary use is still allowed. It gives no token budget, so the room left in outputs is
calibrated here: every reading is kept in work/service/usage.json together with the outputs and tokens the service
had made by then, and the percent used per output is the rise in percent over the rise in outputs between readings of
the same window (a window reset in between drops that pair, and so does a pair with no new output: Codex use outside
the service already shows in the percent used, so it lowers the room left without inflating the per-output rate).
Outside use between two readings that do span outputs is counted against them, so the figure errs on the safe side.
The percent is a whole number, so the estimate firms up as more outputs are made.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

CACHE_SECONDS = 60
MIN_CALIBRATION_OUTPUTS = 6       # outputs over which the percent must have been seen to rise before it is trusted


def read_rate_limits(codex_bin: str = "codex", timeout: float = 30) -> dict:
    """One `account/rateLimits/read` through a short-lived `codex app-server` on stdio. Raises RuntimeError."""
    try:
        process = subprocess.Popen([codex_bin, "app-server"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL, text=True)
    except OSError as error:
        raise RuntimeError(f"Codex CLI unavailable: {error}") from error
    deadline = time.time() + timeout
    try:
        def send(message: dict) -> None:
            process.stdin.write(json.dumps(message) + "\n")
            process.stdin.flush()

        def answer(request_id: int) -> dict:
            while time.time() < deadline:
                line = process.stdout.readline()
                if not line:
                    break
                message = json.loads(line)
                if message.get("id") == request_id:
                    if "error" in message:
                        raise RuntimeError(f"Codex app-server: {message['error']}")
                    return message["result"]
            raise RuntimeError("Codex app-server did not answer")

        send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
              "params": {"clientInfo": {"name": "violation-generator", "version": "1"}}})
        answer(1)
        send({"jsonrpc": "2.0", "method": "initialized"})
        send({"jsonrpc": "2.0", "id": 2, "method": "account/rateLimits/read"})
        result = answer(2)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Codex app-server: {error}") from error
    finally:
        process.kill()
    limits = result.get("rateLimits") or {}
    primary = limits.get("primary") or {}
    secondary = limits.get("secondary") or {}
    return {
        "plan": limits.get("planType"),
        "ordinary_usage_allowed": bool(result.get("ordinaryUsageAllowed", True)),
        "limit_reached": limits.get("rateLimitReachedType") is not None or bool(limits.get("spendControlReached")),
        "used_percent": primary.get("usedPercent"),
        "window_minutes": primary.get("windowDurationMins"),
        "resets_at": primary.get("resetsAt"),
        "secondary_used_percent": secondary.get("usedPercent"),
        "secondary_resets_at": secondary.get("resetsAt"),
        "read_at": time.time(),
    }


class CodexUsage:
    """Cached readings, their history on disk and the per-output calibration."""

    def __init__(self, path: Path, totals, reader=read_rate_limits):
        self.path = path
        self.totals = totals              # () -> (outputs made so far, tokens used so far) by the service
        self.reader = reader
        self.lock = threading.Lock()
        self.last: dict | None = None
        self.error: str | None = None

    def _points(self) -> list[dict]:
        try:
            return json.loads(self.path.read_text()).get("points", [])
        except (OSError, ValueError):
            return []

    def _save(self, points: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f"{self.path.name}.{threading.get_ident()}.tmp")
        temporary.write_text(json.dumps({"points": points[-500:]}, indent=1))
        os.replace(temporary, self.path)

    def refresh(self, force: bool = False) -> dict | None:
        """The latest reading (from cache within CACHE_SECONDS unless force), kept as a calibration point."""
        with self.lock:
            if not force and self.last and time.time() - self.last["read_at"] < CACHE_SECONDS:
                return self.last
            try:
                reading = self.reader()
            except RuntimeError as error:
                self.error = str(error)
                return self.last
            self.error = None
            self.last = reading
            outputs, tokens = self.totals()
            points = self._points()
            point = {"t": reading["read_at"], "used_percent": reading["used_percent"],
                     "resets_at": reading["resets_at"], "outputs": outputs, "tokens": tokens}
            previous = points[-1] if points else None
            if not previous or any(previous.get(k) != point[k] for k in ("used_percent", "resets_at", "outputs")):
                points.append(point)
                self._save(points)
            return reading

    def calibration(self) -> dict:
        """Percent of the window per output (and per 1k tokens), pooled over reading pairs of one window."""
        points = self._points()
        d_percent = d_outputs = d_tokens = 0.0
        for a, b in zip(points, points[1:]):
            if (a.get("resets_at") != b.get("resets_at") or a.get("used_percent") is None
                    or b.get("used_percent") is None or b["outputs"] <= a["outputs"]
                    or b["used_percent"] < a["used_percent"]):
                continue
            d_percent += b["used_percent"] - a["used_percent"]
            d_outputs += b["outputs"] - a["outputs"]
            d_tokens += b["tokens"] - a["tokens"]
        calibrated = d_outputs >= MIN_CALIBRATION_OUTPUTS and d_percent > 0
        return {"calibrated": calibrated, "outputs_seen": int(d_outputs), "percent_seen": d_percent,
                "percent_per_output": d_percent / d_outputs if calibrated else None,
                "percent_per_1k_tokens": d_percent / d_tokens * 1000 if calibrated and d_tokens else None}

    def status(self, pending_outputs: int = 0, force: bool = False) -> dict:
        """The plan's state, the calibration and the room left, in outputs, after the outputs still to make."""
        reading = self.refresh(force)
        calibration = self.calibration()
        room = None
        if reading and reading.get("used_percent") is not None and calibration["calibrated"]:
            left = max(0.0, 100 - reading["used_percent"] - pending_outputs * calibration["percent_per_output"])
            room = int(left / calibration["percent_per_output"])
        blocked = bool(reading) and (reading["limit_reached"] or not reading["ordinary_usage_allowed"])
        return {"available": reading is not None, "error": self.error, "reading": reading,
                "calibration": calibration, "pending_outputs": pending_outputs, "outputs_left": room,
                "blocked": blocked}

    def check(self, outputs: int, pending_outputs: int) -> dict:
        """Whether a new job of `outputs` fits: {fits, reason, status}. Unknown room (no reading or no
        calibration yet) fits, with the reason saying so."""
        status = self.status(pending_outputs)
        if status["blocked"]:
            return {"fits": False, "status": status,
                    "reason": "Codex reports the plan's usage limit is reached; no edit can run until it resets."}
        room = status["outputs_left"]
        if room is not None and outputs > room:
            return {"fits": False, "status": status,
                    "reason": f"This job makes {outputs} outputs, but the Codex plan has room for about {room} more "
                              "(after the outputs already queued) before its limit resets."}
        reason = None if room is not None else ("The room left is not known yet: it is calibrated from finished "
                                                "outputs." if status["available"] else "The Codex usage limit could "
                                                "not be read.")
        return {"fits": True, "status": status, "reason": reason}
