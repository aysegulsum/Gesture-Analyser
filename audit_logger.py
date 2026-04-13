"""
Unified Audit Logger
====================
Appends one JSON-Lines record per event to ``logs/audit_<session_id>.jsonl``.

Each record is a single JSON object followed by a newline::

    {"ts": 1718000000.123, "session": "a1b2c3d4", "mode": "Normal",
     "event": "success", "target": 5, "score": 3}

Fields
------
ts       Unix wall-clock timestamp (float, millisecond precision).
session  Hex UUID prefix that is stable for one application run.
mode     Active game-mode name at the time of the event.
event    Short label identifying the event type (see table below).

Common event types
------------------
app_start          Application launched.
app_quit           Application closed cleanly.
mode_change        User cycled to a new mode (M key).
session_restart    User restarted the current mode (R key).
round_start        New target / question / challenge picked.
success            A gesture or answer was validated.
failure            A challenge timed out or was failed.
game_over          Math mode 60-second timer expired.
step_passed        Sequential mode: one step completed.
step_timed_out     Sequential mode: one step timed out.
session_complete   Sequential or Touch Test: all steps done.
challenge_success  Liveness mode: one challenge passed.
challenge_failed   Liveness mode: one challenge failed/timed out.
verified_100       Liveness mode: all challenges completed.
touch_success      Touch Test: one touch command confirmed.
shape_eval_result  Shape Eval: one round result recorded.

Usage::

    from audit_logger import log

    log("success", mode="Normal", target=5, score=3)
"""

import json
import time
import uuid
from pathlib import Path
from threading import Lock


# ── Module-level singletons ──────────────────────────────────────────

SESSION_ID: str = uuid.uuid4().hex[:12]

_LOG_DIR = Path("logs")
_LOG_DIR.mkdir(exist_ok=True)
_LOG_PATH = _LOG_DIR / f"audit_{SESSION_ID}.jsonl"
_LOCK = Lock()


# ── Public API ───────────────────────────────────────────────────────

def log(event: str, *, mode: str = "", **data) -> None:
    """Append one JSON-Lines entry to the current session audit file.

    Parameters
    ----------
    event : str
        Short event-type label, e.g. ``"success"``, ``"round_start"``.
    mode : str
        Active game-mode name (``"Normal"``, ``"Math"``, …).
    **data
        Arbitrary key-value payload included verbatim in the entry.
        Values that are not JSON-serialisable are coerced to strings.
    """
    entry: dict = {
        "ts": round(time.time(), 3),
        "session": SESSION_ID,
        "mode": mode,
        "event": event,
    }
    entry.update(data)
    line = json.dumps(entry, default=str)
    with _LOCK:
        with _LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
