"""
Fast-Response Liveness Challenge -- v3: Robust Wave + Air Drawing
==================================================================
Command Types
-------------
1. GESTURE:     Finger-count match
2. MOVE_CLOSER: Depth proxy increase >= 20%
3. MOVE_AWAY:   Depth proxy decrease >= 20%
4. WAVE:        Oscillation filter with frequency check
5. DRAW_CIRCLE: Air-draw validated on timeout/finger-close
6. DRAW_SQUARE: Air-draw validated on timeout/finger-close

Key changes from v2:
- Wave uses timestamped oscillation filter with 1-4 Hz frequency band.
- Drawing only records points when index finger is open.
- Drawing validates ONCE at the end (timeout or finger close), not
  every frame.  This avoids premature false matches.
- 3-point moving average smoothing on the trajectory.
- process_every_nth: skip expensive motion math on alternate frames
  if performance is an issue.
"""

import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from hand_tracker import HandResult
from gesture_validator import GestureValidator, depth_proxy, is_finger_open, Finger
from motion_analyzer import WaveDetector, ShapeRecognizer


# ── Command definitions ─────────────────────────────────────────────

class CmdType(Enum):
    GESTURE = auto()
    MOVE_CLOSER = auto()
    MOVE_AWAY = auto()
    WAVE = auto()
    DRAW_CIRCLE = auto()
    DRAW_SQUARE = auto()


@dataclass(frozen=True)
class Command:
    name: str
    cmd_type: CmdType
    gesture_reqs: dict[str, int] | None = None


_GESTURE_CMDS = [
    Command("SHOW LEFT FIST!",      CmdType.GESTURE, {"Left": 0}),
    Command("SHOW RIGHT FIST!",     CmdType.GESTURE, {"Right": 0}),
    Command("SHOW 2 FINGERS!",      CmdType.GESTURE, {"Right": 2}),
    Command("SHOW 3 FINGERS!",      CmdType.GESTURE, {"Right": 3}),
    Command("SHOW 4 FINGERS!",      CmdType.GESTURE, {"Left": 4}),
    Command("OPEN BOTH HANDS!",     CmdType.GESTURE, {"Left": 5, "Right": 5}),
    Command("LEFT THUMB UP!",       CmdType.GESTURE, {"Left": 1}),
    Command("RIGHT OPEN HAND!",     CmdType.GESTURE, {"Right": 5}),
    Command("LEFT 3 + RIGHT 2!",    CmdType.GESTURE, {"Left": 3, "Right": 2}),
    Command("BOTH FISTS!",          CmdType.GESTURE, {"Left": 0, "Right": 0}),
]

_SPATIAL_CMDS = [
    Command("MOVE HAND CLOSER!",    CmdType.MOVE_CLOSER),
    Command("MOVE HAND AWAY!",      CmdType.MOVE_AWAY),
]

_MOTION_CMDS = [
    Command("WAVE YOUR HAND!",      CmdType.WAVE),
    Command("DRAW A CIRCLE!",       CmdType.DRAW_CIRCLE),
    Command("DRAW A SQUARE!",       CmdType.DRAW_SQUARE),
]

ALL_COMMANDS = _GESTURE_CMDS + _SPATIAL_CMDS + _MOTION_CMDS


class LivenessState(Enum):
    ACTIVE = auto()
    DEBOUNCE = auto()
    SUCCESS = auto()
    FAILED = auto()


_INDEX_TIP = 8


@dataclass
class LivenessChallenge:
    """Fast-response liveness detector with wave + air drawing.

    Parameters
    ----------
    time_limit : float
        Base seconds per challenge (drawing gets +3s).
    debounce_seconds : float
        Hold time for gesture/spatial matches.
    area_change_threshold : float
        Required depth change for spatial commands.
    pause_after_result : float
        Display time for result before next challenge.
    smoothing_window : int
        Passed to GestureValidator.
    process_every_nth : int
        Only run motion math every Nth frame (1 = every frame).
    """

    time_limit: float = 4.0
    debounce_seconds: float = 0.5
    area_change_threshold: float = 0.20
    pause_after_result: float = 1.5
    smoothing_window: int = 7
    process_every_nth: int = 1

    # -- internal state ---------------------------------------------------
    state: LivenessState = field(init=False, default=LivenessState.ACTIVE)
    current_cmd: Command = field(init=False, default=None)
    score: int = field(init=False, default=0)
    streak: int = field(init=False, default=0)

    _challenge_start: Optional[float] = field(init=False, default=None)
    _debounce_start: Optional[float] = field(init=False, default=None)
    _result_at: Optional[float] = field(init=False, default=None)

    _baseline_area: Optional[float] = field(init=False, default=None)
    _current_area: Optional[float] = field(init=False, default=None)

    _validator: GestureValidator = field(init=False)
    _wave: WaveDetector = field(init=False)
    _shape: ShapeRecognizer = field(init=False)

    _frame_counter: int = field(init=False, default=0)
    _was_drawing: bool = field(init=False, default=False)

    def __post_init__(self):
        self._validator = GestureValidator(smoothing_window=self.smoothing_window)
        self._wave = WaveDetector(
            buffer_size=40, min_swing=0.10,
            min_total_displacement=0.20, min_reversals=2,
        )
        self._shape = ShapeRecognizer(min_points=15)
        self._pick_command()

    def _pick_command(self) -> None:
        self.current_cmd = random.choice(ALL_COMMANDS)
        self._challenge_start = None
        self._debounce_start = None
        self._result_at = None
        self._baseline_area = None
        self._current_area = None
        self._was_drawing = False
        self._frame_counter = 0
        self.state = LivenessState.ACTIVE
        self._validator.clear_buffers()
        self._wave.reset()
        self._shape.reset()

    @property
    def _effective_time_limit(self) -> float:
        if self.current_cmd.cmd_type in (CmdType.DRAW_CIRCLE, CmdType.DRAW_SQUARE):
            return self.time_limit + 3.0  # 7s for drawing
        if self.current_cmd.cmd_type == CmdType.WAVE:
            return self.time_limit + 1.0  # 5s for wave
        return self.time_limit

    # -- Matching helpers -------------------------------------------------

    def _gesture_matched(self, hands: list[HandResult]) -> bool:
        reqs = self.current_cmd.gesture_reqs
        if reqs is None:
            return False
        hand_map: dict[str, int] = {}
        for h in hands:
            hand_map[h.handedness] = self._validator.count_fingers(h.landmarks, h.handedness)
        return all(hand_map.get(label) == count for label, count in reqs.items())

    def _get_depth(self, hands: list[HandResult]) -> Optional[float]:
        return depth_proxy(hands[0].landmarks) if hands else None

    def _spatial_matched(self, hands: list[HandResult]) -> bool:
        area = self._get_depth(hands)
        if area is None or self._baseline_area is None:
            return False
        self._current_area = area
        ratio = area / self._baseline_area
        if self.current_cmd.cmd_type == CmdType.MOVE_CLOSER:
            return ratio >= (1.0 + self.area_change_threshold)
        elif self.current_cmd.cmd_type == CmdType.MOVE_AWAY:
            return ratio <= (1.0 - self.area_change_threshold)
        return False

    def _index_is_open(self, hands: list[HandResult]) -> bool:
        """Check if the index finger of the first hand is extended."""
        if not hands:
            return False
        return is_finger_open(hands[0].landmarks, hands[0].handedness, Finger.INDEX)

    # -- Main update ------------------------------------------------------

    def update(self, hands: list[HandResult]) -> LivenessState:
        now = time.time()
        self._frame_counter += 1

        # After result, pause then auto-advance.
        if self.state in (LivenessState.SUCCESS, LivenessState.FAILED):
            if self._result_at and now - self._result_at >= self.pause_after_result:
                self._pick_command()
            return self.state

        # Timer starts on first hand detection.
        if self._challenge_start is None:
            if not hands:
                return self.state
            self._challenge_start = now
            if self.current_cmd.cmd_type in (CmdType.MOVE_CLOSER, CmdType.MOVE_AWAY):
                self._baseline_area = self._get_depth(hands)

        # Timeout -- for drawing commands, validate the shape first.
        if now - self._challenge_start >= self._effective_time_limit:
            if self.current_cmd.cmd_type in (CmdType.DRAW_CIRCLE, CmdType.DRAW_SQUARE):
                if self._validate_drawing_final():
                    self._succeed(now)
                    return self.state
            self.state = LivenessState.FAILED
            self._result_at = now
            self.streak = 0
            return self.state

        # Skip expensive motion math on non-processing frames.
        should_process = (self._frame_counter % self.process_every_nth == 0)

        # -- DEBOUNCE (gesture/spatial only) -------------------------------
        if self.state == LivenessState.DEBOUNCE:
            if should_process and self._check_static_match(hands):
                if now - self._debounce_start >= self.debounce_seconds:
                    self._succeed(now)
            else:
                if should_process:
                    self.state = LivenessState.ACTIVE
                    self._debounce_start = None
            return self.state

        # -- ACTIVE --------------------------------------------------------
        ct = self.current_cmd.cmd_type

        # Spatial: keep tracking.
        if ct in (CmdType.MOVE_CLOSER, CmdType.MOVE_AWAY):
            area = self._get_depth(hands)
            if area is not None:
                self._current_area = area
                if self._baseline_area is None:
                    self._baseline_area = area

        # Wave: push wrist X every frame, check on processing frames.
        if ct == CmdType.WAVE and hands:
            self._wave.push(hands[0].landmarks[0].x)
            if should_process and self._wave.is_waving():
                self._succeed(now)
                return self.state

        # Drawing: record index tip when finger is open.
        if ct in (CmdType.DRAW_CIRCLE, CmdType.DRAW_SQUARE):
            finger_open = self._index_is_open(hands)
            if finger_open and hands:
                tip = hands[0].landmarks[_INDEX_TIP]
                self._shape.push(tip.x, tip.y)
                self._was_drawing = True
            elif self._was_drawing and not finger_open and len(self._shape.path) >= self._shape.min_points:
                # Finger just closed after drawing -- validate now.
                if self._validate_drawing_final():
                    self._succeed(now)
                    return self.state
                self._was_drawing = False
            return self.state

        # Static commands (gesture/spatial).
        if should_process:
            if self._check_static_match(hands):
                self.state = LivenessState.DEBOUNCE
                self._debounce_start = now

        return self.state

    def _check_static_match(self, hands: list[HandResult]) -> bool:
        ct = self.current_cmd.cmd_type
        if ct == CmdType.GESTURE:
            return self._gesture_matched(hands)
        elif ct in (CmdType.MOVE_CLOSER, CmdType.MOVE_AWAY):
            return self._spatial_matched(hands)
        return False

    def _validate_drawing_final(self) -> bool:
        """One-shot shape validation (called on timeout or finger-close)."""
        if self.current_cmd.cmd_type == CmdType.DRAW_CIRCLE:
            return self._shape.finalize_circle().matched
        elif self.current_cmd.cmd_type == CmdType.DRAW_SQUARE:
            return self._shape.finalize_square().matched
        return False

    def _succeed(self, now: float):
        self.state = LivenessState.SUCCESS
        self._result_at = now
        self.score += 1
        self.streak += 1

    # -- Display helpers --------------------------------------------------

    @property
    def time_remaining(self) -> float:
        if self._challenge_start is None:
            return self._effective_time_limit
        return max(0.0, self._effective_time_limit - (time.time() - self._challenge_start))

    @property
    def debounce_progress(self) -> float:
        if self._debounce_start is None:
            return 0.0
        return min((time.time() - self._debounce_start) / self.debounce_seconds, 1.0)

    @property
    def area_change_pct(self) -> Optional[float]:
        if self._baseline_area is None or self._current_area is None:
            return None
        return ((self._current_area / self._baseline_area) - 1.0) * 100.0

    @property
    def wave_reversals(self) -> int:
        """Current reversal count for UI feedback."""
        return self._wave.reversal_count

    @property
    def command_label(self) -> str:
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
        if self._challenge_start is None:
            return "Status: Show your hand to start"
        return "Status: Respond NOW!"

    @property
    def is_flash_red(self) -> bool:
        if self.state != LivenessState.FAILED or self._result_at is None:
            return False
        return (time.time() - self._result_at) < 0.6

    @property
    def drawing_path(self) -> list[tuple[float, float]]:
        return self._shape.pixel_path

    @property
    def drawing_point_count(self) -> int:
        return len(self._shape.path)

    @property
    def is_drawing_cmd(self) -> bool:
        return self.current_cmd.cmd_type in (CmdType.DRAW_CIRCLE, CmdType.DRAW_SQUARE)

    @property
    def is_wave_cmd(self) -> bool:
        return self.current_cmd.cmd_type == CmdType.WAVE

    def per_hand_counts(self, hands: list[HandResult]) -> dict[str, int]:
        return {
            h.handedness: self._validator.count_fingers(h.landmarks, h.handedness)
            for h in hands
        }

    def reset(self) -> None:
        self.score = 0
        self.streak = 0
        self._pick_command()
