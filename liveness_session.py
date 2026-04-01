"""
Fast-Response Liveness Challenge
================================
Single-command, 4-second-window liveness detection with Z-axis
(depth / proximity) movement commands.

State machine:

    ACTIVE  ->  DEBOUNCE  ->  SUCCESS  ->  (next challenge)
       |                         ^
       +-- timeout ------------> FAILED

Command Types
-------------
1. **Gesture commands**: Same finger-count validation as other modes.
   e.g. "QUICK: SHOW 4 FINGERS!", "QUICK: LEFT FIST!"

2. **Spatial / Z-axis commands**: Track the normalised Wrist-to-
   MiddleMCP distance (``depth_proxy``) as a size/depth metric.
   When the hand moves closer this distance grows; away it shrinks.

   - "MOVE HAND CLOSER"  -> depth_proxy must increase by >= 20%
   - "MOVE HAND AWAY"    -> depth_proxy must decrease by >= 20%

Non-blocking: all timing uses ``time.time()`` comparisons inside
``update()``; the caller's frame loop is never blocked.
"""

import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from hand_tracker import HandResult
from gesture_validator import GestureValidator, depth_proxy


# ── Command definitions ─────────────────────────────────────────────

class CmdType(Enum):
    GESTURE = auto()
    MOVE_CLOSER = auto()
    MOVE_AWAY = auto()


@dataclass(frozen=True)
class Command:
    name: str                          # display text
    cmd_type: CmdType
    gesture_reqs: dict[str, int] | None = None  # for GESTURE type


# Gesture commands
_GESTURE_CMDS = [
    Command("SHOW LEFT FIST!",          CmdType.GESTURE, {"Left": 0}),
    Command("SHOW RIGHT FIST!",         CmdType.GESTURE, {"Right": 0}),
    Command("SHOW 2 FINGERS!",          CmdType.GESTURE, {"Right": 2}),
    Command("SHOW 3 FINGERS!",          CmdType.GESTURE, {"Right": 3}),
    Command("SHOW 4 FINGERS!",          CmdType.GESTURE, {"Left": 4}),
    Command("OPEN BOTH HANDS!",         CmdType.GESTURE, {"Left": 5, "Right": 5}),
    Command("LEFT THUMB UP!",           CmdType.GESTURE, {"Left": 1}),
    Command("RIGHT OPEN HAND!",         CmdType.GESTURE, {"Right": 5}),
    Command("LEFT 3 + RIGHT 2!",        CmdType.GESTURE, {"Left": 3, "Right": 2}),
    Command("BOTH FISTS!",              CmdType.GESTURE, {"Left": 0, "Right": 0}),
]

# Spatial commands
_SPATIAL_CMDS = [
    Command("MOVE HAND CLOSER!",  CmdType.MOVE_CLOSER),
    Command("MOVE HAND AWAY!",    CmdType.MOVE_AWAY),
]

ALL_COMMANDS = _GESTURE_CMDS + _SPATIAL_CMDS


class LivenessState(Enum):
    ACTIVE = auto()     # command displayed, waiting for user
    DEBOUNCE = auto()   # correct action detected, confirming (0.5s)
    SUCCESS = auto()    # challenge passed
    FAILED = auto()     # timeout or wrong action


# ── Depth helpers ───────────────────────────────────────────────────
# depth_proxy is imported from gesture_validator (Wrist-to-MiddleMCP).


# ── LivenessChallenge class ─────────────────────────────────────────

@dataclass
class LivenessChallenge:
    """Fast-response, single-command liveness detector.

    Parameters
    ----------
    time_limit : float
        Seconds allowed per challenge (default 4.0).
    debounce_seconds : float
        Hold time to confirm a correct action (default 0.5).
    area_change_threshold : float
        Required relative change in hand bbox area for spatial
        commands (default 0.20 = 20%).
    pause_after_result : float
        Seconds to display SUCCESS/FAILED before the next challenge.
    smoothing_window : int
        Passed through to GestureValidator.
    """

    time_limit: float = 4.0
    debounce_seconds: float = 0.5
    area_change_threshold: float = 0.20
    pause_after_result: float = 1.5
    smoothing_window: int = 7

    # -- internal state ---------------------------------------------------
    state: LivenessState = field(init=False, default=LivenessState.ACTIVE)
    current_cmd: Command = field(init=False, default=None)
    score: int = field(init=False, default=0)
    streak: int = field(init=False, default=0)

    _challenge_start: Optional[float] = field(init=False, default=None)
    _debounce_start: Optional[float] = field(init=False, default=None)
    _result_at: Optional[float] = field(init=False, default=None)

    # For spatial commands: baseline hand area captured at challenge start.
    _baseline_area: Optional[float] = field(init=False, default=None)
    _current_area: Optional[float] = field(init=False, default=None)

    _validator: GestureValidator = field(init=False)

    def __post_init__(self):
        self._validator = GestureValidator(smoothing_window=self.smoothing_window)
        self._pick_command()

    # -- Command selection ------------------------------------------------

    def _pick_command(self) -> None:
        self.current_cmd = random.choice(ALL_COMMANDS)
        self._challenge_start = None    # starts on first update()
        self._debounce_start = None
        self._result_at = None
        self._baseline_area = None
        self._current_area = None
        self.state = LivenessState.ACTIVE
        self._validator.clear_buffers()

    # -- Gesture matching (reuses validator) ------------------------------

    def _gesture_matched(self, hands: list[HandResult]) -> bool:
        reqs = self.current_cmd.gesture_reqs
        if reqs is None:
            return False
        hand_map: dict[str, int] = {}
        for h in hands:
            hand_map[h.handedness] = self._validator.count_fingers(
                h.landmarks, h.handedness
            )
        for label, count in reqs.items():
            if hand_map.get(label) != count:
                return False
        return True

    # -- Spatial matching (depth proxy: Wrist-to-MiddleMCP distance) -----

    def _get_any_hand_depth(self, hands: list[HandResult]) -> Optional[float]:
        """Return the depth proxy of the first detected hand."""
        if not hands:
            return None
        return depth_proxy(hands[0].landmarks)

    def _spatial_matched(self, hands: list[HandResult]) -> bool:
        area = self._get_any_hand_depth(hands)
        if area is None or self._baseline_area is None:
            return False

        self._current_area = area
        ratio = area / self._baseline_area

        if self.current_cmd.cmd_type == CmdType.MOVE_CLOSER:
            return ratio >= (1.0 + self.area_change_threshold)
        elif self.current_cmd.cmd_type == CmdType.MOVE_AWAY:
            return ratio <= (1.0 - self.area_change_threshold)
        return False

    # -- Main update ------------------------------------------------------

    def update(self, hands: list[HandResult]) -> LivenessState:
        """Call once per frame. Non-blocking."""
        now = time.time()

        # After SUCCESS/FAILED, pause then auto-advance.
        if self.state in (LivenessState.SUCCESS, LivenessState.FAILED):
            if self._result_at and now - self._result_at >= self.pause_after_result:
                self._pick_command()
            return self.state

        # Start clock on first frame.
        if self._challenge_start is None:
            self._challenge_start = now
            # Capture baseline area for spatial commands.
            if self.current_cmd.cmd_type in (CmdType.MOVE_CLOSER, CmdType.MOVE_AWAY):
                self._baseline_area = self._get_any_hand_depth(hands)

        # Check 4-second timeout.
        if now - self._challenge_start >= self.time_limit:
            self.state = LivenessState.FAILED
            self._result_at = now
            self.streak = 0
            return self.state

        # -- DEBOUNCE: waiting for 0.5s confirmation ----------------------
        if self.state == LivenessState.DEBOUNCE:
            matched = self._check_match(hands)
            if matched:
                if now - self._debounce_start >= self.debounce_seconds:
                    self.state = LivenessState.SUCCESS
                    self._result_at = now
                    self.score += 1
                    self.streak += 1
            else:
                # Lost the match during debounce -- back to ACTIVE.
                self.state = LivenessState.ACTIVE
                self._debounce_start = None
            return self.state

        # -- ACTIVE: looking for the correct action -----------------------
        # Update area tracking for spatial commands.
        if self.current_cmd.cmd_type in (CmdType.MOVE_CLOSER, CmdType.MOVE_AWAY):
            area = self._get_any_hand_depth(hands)
            if area is not None:
                self._current_area = area
                if self._baseline_area is None:
                    self._baseline_area = area

        matched = self._check_match(hands)
        if matched:
            self.state = LivenessState.DEBOUNCE
            self._debounce_start = now

        return self.state

    def _check_match(self, hands: list[HandResult]) -> bool:
        if self.current_cmd.cmd_type == CmdType.GESTURE:
            return self._gesture_matched(hands)
        return self._spatial_matched(hands)

    # -- Display helpers --------------------------------------------------

    @property
    def time_remaining(self) -> float:
        if self._challenge_start is None:
            return self.time_limit
        return max(0.0, self.time_limit - (time.time() - self._challenge_start))

    @property
    def debounce_progress(self) -> float:
        """0->1 progress through the debounce confirmation."""
        if self._debounce_start is None:
            return 0.0
        return min((time.time() - self._debounce_start) / self.debounce_seconds, 1.0)

    @property
    def area_change_pct(self) -> Optional[float]:
        """Current area change as a percentage (e.g. +25.3 or -18.1)."""
        if self._baseline_area is None or self._current_area is None:
            return None
        return ((self._current_area / self._baseline_area) - 1.0) * 100.0

    @property
    def command_label(self) -> str:
        """The action-style command string."""
        ct = self.current_cmd.cmd_type
        if ct == CmdType.GESTURE:
            return f"QUICK: {self.current_cmd.name}"
        return f"ACTION: {self.current_cmd.name}"

    @property
    def display_text(self) -> str:
        if self.state == LivenessState.SUCCESS:
            return "VERIFIED!"
        if self.state == LivenessState.FAILED:
            return "VERIFICATION FAILED"
        return self.command_label

    @property
    def status_label(self) -> str:
        if self.state == LivenessState.SUCCESS:
            return "Status: Challenge Passed"
        if self.state == LivenessState.FAILED:
            return "Status: Time's Up!"
        if self.state == LivenessState.DEBOUNCE:
            return "Status: Confirming..."
        return "Status: Respond NOW!"

    @property
    def is_flash_red(self) -> bool:
        """True when the UI should flash red (failure moment)."""
        if self.state != LivenessState.FAILED or self._result_at is None:
            return False
        return (time.time() - self._result_at) < 0.6  # flash for 0.6s

    def per_hand_counts(self, hands: list[HandResult]) -> dict[str, int]:
        return {
            h.handedness: self._validator.count_fingers(h.landmarks, h.handedness)
            for h in hands
        }

    def reset(self) -> None:
        """Full reset."""
        self.score = 0
        self.streak = 0
        self._pick_command()
