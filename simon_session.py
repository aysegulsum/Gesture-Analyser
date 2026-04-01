"""
Simon Says Session -- Sequence Memory & Liveness Challenge
==========================================================
State machine:

    SHOWING  ->  REPLICATING  ->  VERIFIED / FAILED
                     ^  |
                     +--+  (advance step)

Phases
------
SHOWING      -- Display each gesture in the sequence one-by-one with a
               countdown per step so the user can memorise it.
REPLICATING  -- User must perform every gesture in order.  Each step
               requires a 1.5-second stability hold.
VERIFIED     -- Full sequence completed within the time limit.
FAILED       -- Wrong gesture performed, or global timer expired.

Gesture Vocabulary
------------------
Each gesture is defined as a dict of per-hand requirements:
    {"Left": finger_count, "Right": finger_count}
A hand omitted from the dict means "don't care / not required".
Special value -1 means "hand must NOT be detected" (hidden).

This maps cleanly onto the existing GestureValidator which provides
per-hand smoothed finger counts.
"""

import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from hand_tracker import HandResult
from gesture_validator import GestureValidator


# ── Gesture catalogue ───────────────────────────────────────────────

@dataclass(frozen=True)
class Gesture:
    """A single gesture the user must perform."""
    name: str                      # human-readable label
    requirements: dict[str, int]   # {"Left": count, "Right": count}


# Predefined gestures using both hands.
GESTURE_POOL = [
    Gesture("Right Fist",          {"Right": 0}),
    Gesture("Left Fist",           {"Left": 0}),
    Gesture("Right 2 Fingers",     {"Right": 2}),
    Gesture("Left 2 Fingers",      {"Left": 2}),
    Gesture("Right 3 Fingers",     {"Right": 3}),
    Gesture("Left 3 Fingers",      {"Left": 3}),
    Gesture("Right Open Hand",     {"Right": 5}),
    Gesture("Left Open Hand",      {"Left": 5}),
    Gesture("Right Thumb Up",      {"Right": 1}),
    Gesture("Left Thumb Up",       {"Left": 1}),
    Gesture("Both Hands Open",     {"Left": 5, "Right": 5}),
    Gesture("Both Fists",          {"Left": 0, "Right": 0}),
    Gesture("Left 4 + Right 1",    {"Left": 4, "Right": 1}),
    Gesture("Right 3 + Left 2",    {"Left": 2, "Right": 3}),
]


class SimonState(Enum):
    SHOWING = auto()       # displaying the sequence to memorise
    REPLICATING = auto()   # user is performing gestures in order
    VERIFIED = auto()      # liveness verified -- full sequence matched
    FAILED = auto()        # wrong gesture or timeout


@dataclass
class SimonSaysGame:
    """Simon Says liveness challenge.

    Parameters
    ----------
    sequence_length : int
        How many gestures per round (3-5).
    show_duration : float
        Seconds to display each gesture during SHOWING phase.
    hold_seconds : float
        Stability hold required per step during REPLICATING.
    global_timeout : float
        Total seconds allowed for the REPLICATING phase.
    smoothing_window : int
        Passed through to GestureValidator.
    """

    sequence_length: int = 4
    show_duration: float = 2.0
    hold_seconds: float = 1.5
    global_timeout: float = 15.0
    smoothing_window: int = 7

    # -- internal state ---------------------------------------------------
    state: SimonState = field(init=False, default=SimonState.SHOWING)
    sequence: list[Gesture] = field(init=False, default_factory=list)
    current_step: int = field(init=False, default=0)
    rounds_completed: int = field(init=False, default=0)

    _show_start: Optional[float] = field(init=False, default=None)
    _show_index: int = field(init=False, default=0)

    _replicate_start: Optional[float] = field(init=False, default=None)
    _hold_start: Optional[float] = field(init=False, default=None)

    _validator: GestureValidator = field(init=False)

    def __post_init__(self):
        self._validator = GestureValidator(smoothing_window=self.smoothing_window)
        self._generate_sequence()

    # -- Sequence generation ----------------------------------------------

    def _generate_sequence(self) -> None:
        n = min(self.sequence_length, len(GESTURE_POOL))
        self.sequence = random.sample(GESTURE_POOL, k=n)
        self.current_step = 0
        self._show_index = 0
        self._show_start = None
        self._replicate_start = None
        self._hold_start = None
        self.state = SimonState.SHOWING
        self._validator.clear_buffers()

    # -- Gesture matching -------------------------------------------------

    def _match_gesture(self, hands: list[HandResult], gesture: Gesture) -> bool:
        """Check if current hands match *gesture* requirements."""
        hand_map: dict[str, int] = {}
        for h in hands:
            hand_map[h.handedness] = self._validator.count_fingers(
                h.landmarks, h.handedness
            )

        for label, required_count in gesture.requirements.items():
            actual = hand_map.get(label)
            if actual is None:
                # Required hand not detected.
                if required_count == -1:
                    continue  # -1 means hand should be absent -- OK
                return False
            if required_count == -1:
                return False  # hand should be absent but is detected
            if actual != required_count:
                return False
        return True

    def _is_wrong_gesture(self, hands: list[HandResult], gesture: Gesture) -> bool:
        """Return True if user is clearly showing the *wrong* gesture.

        We only flag a wrong gesture when at least one required hand is
        visible but showing a different count.  If no hands are detected
        we just wait (the user might be transitioning).
        """
        if not hands:
            return False

        hand_map: dict[str, int] = {}
        for h in hands:
            hand_map[h.handedness] = self._validator.count_fingers(
                h.landmarks, h.handedness
            )

        for label, required_count in gesture.requirements.items():
            if required_count == -1:
                continue
            actual = hand_map.get(label)
            if actual is not None and actual != required_count:
                return True
        return False

    # -- Main update ------------------------------------------------------

    def update(self, hands: list[HandResult]) -> SimonState:
        """Call once per frame. Returns the current state."""
        now = time.monotonic()

        # -- SHOWING phase: display gestures one by one -------------------
        if self.state == SimonState.SHOWING:
            if self._show_start is None:
                self._show_start = now

            elapsed = now - self._show_start
            idx = int(elapsed / self.show_duration)

            if idx >= len(self.sequence):
                # Done showing -- move to replication.
                self.state = SimonState.REPLICATING
                self.current_step = 0
                self._replicate_start = now
                self._hold_start = None
                self._validator.clear_buffers()
            else:
                self._show_index = idx

            return self.state

        # -- Terminal states ----------------------------------------------
        if self.state in (SimonState.VERIFIED, SimonState.FAILED):
            return self.state

        # -- REPLICATING phase --------------------------------------------
        # Global timeout check.
        if now - self._replicate_start > self.global_timeout:
            self.state = SimonState.FAILED
            return self.state

        target = self.sequence[self.current_step]

        if self._match_gesture(hands, target):
            # Correct gesture -- run stability timer.
            if self._hold_start is None:
                self._hold_start = now
            elif now - self._hold_start >= self.hold_seconds:
                # Step confirmed -- advance.
                self.current_step += 1
                self._hold_start = None
                self._validator.clear_buffers()

                if self.current_step >= len(self.sequence):
                    self.state = SimonState.VERIFIED
                    self.rounds_completed += 1
        else:
            # Not matching.  Check if it's an active wrong gesture.
            if self._is_wrong_gesture(hands, target):
                self.state = SimonState.FAILED
            # If hands just aren't detected, reset hold but don't fail.
            self._hold_start = None

        return self.state

    # -- Display helpers --------------------------------------------------

    @property
    def showing_gesture(self) -> Optional[Gesture]:
        """The gesture currently being displayed (SHOWING phase only)."""
        if self.state != SimonState.SHOWING:
            return None
        if self._show_index < len(self.sequence):
            return self.sequence[self._show_index]
        return None

    @property
    def show_progress(self) -> float:
        """0->1 progress through the SHOWING countdown for the current gesture."""
        if self._show_start is None:
            return 0.0
        elapsed = time.monotonic() - self._show_start
        within_step = elapsed - (self._show_index * self.show_duration)
        return min(max(within_step / self.show_duration, 0.0), 1.0)

    @property
    def hold_progress(self) -> float:
        """0->1 progress toward confirming the current replication step."""
        if self._hold_start is None:
            return 0.0
        return min((time.monotonic() - self._hold_start) / self.hold_seconds, 1.0)

    @property
    def time_remaining(self) -> float:
        """Seconds left in the REPLICATING phase."""
        if self._replicate_start is None:
            return self.global_timeout
        return max(0.0, self.global_timeout - (time.monotonic() - self._replicate_start))

    @property
    def current_target_gesture(self) -> Optional[Gesture]:
        """The gesture the user must perform right now (REPLICATING only)."""
        if self.state != SimonState.REPLICATING:
            return None
        if self.current_step < len(self.sequence):
            return self.sequence[self.current_step]
        return None

    @property
    def display_text(self) -> str:
        if self.state == SimonState.SHOWING:
            g = self.showing_gesture
            name = g.name if g else "..."
            return f"Memorise: {name}"
        if self.state == SimonState.REPLICATING:
            g = self.current_target_gesture
            name = g.name if g else "..."
            return f"Step {self.current_step + 1}/{len(self.sequence)}: {name}"
        if self.state == SimonState.VERIFIED:
            return "LIVENESS VERIFIED!"
        return "VERIFICATION FAILED"

    @property
    def status_label(self) -> str:
        if self.state == SimonState.SHOWING:
            return f"Status: Showing {self._show_index + 1}/{len(self.sequence)}"
        if self.state == SimonState.REPLICATING:
            if self._hold_start is not None:
                return "Status: Hold steady..."
            return "Status: Perform the gesture"
        if self.state == SimonState.VERIFIED:
            return "Status: Sequence Complete"
        return "Status: Challenge Failed"

    @property
    def sequence_summary(self) -> list[str]:
        """List of gesture names in the current sequence."""
        return [g.name for g in self.sequence]

    def reset(self) -> None:
        """Start a brand-new round."""
        self.rounds_completed = 0
        self._generate_sequence()

    def next_round(self) -> None:
        """Generate a new sequence (keeps rounds_completed counter)."""
        self._generate_sequence()
