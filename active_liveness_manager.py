"""
Active Liveness Manager  --  FIVUCSAS
======================================
Unified multi-task active liveness orchestrator for the FIVUCSAS biometric
authentication system.

Selects 3 random challenges (no repeats) from the challenge pool:

  MATH     — solve a simple arithmetic question with fingers (0-10 result)
  GESTURE  — show a specific number / configuration of fingers
  TOUCH    — pinch specific fingertips (Z-validated, 10-frame hold)
  TRACE    — follow a visual shape template (DTW-verified, point-to-point)

State machine
-------------
  TRANSITION  →  brief inter-challenge screen ("Next Task..." 2 s)
  CHALLENGE   →  one challenge is live and accepting input
  COMPLETE    →  all 3 challenges done; active_liveness_score computed

Scoring
-------
  Each of the 3 challenges contributes equally (weight = 1/3).

  Pass (GESTURE / TOUCH / MATH)  →  quality = 1.0
  Pass (TRACE)                   →  quality = DTW similarity (0.0–1.0)
  Fail (any)                     →  quality = 0.0

  active_liveness_score = mean(qualities)       ∈ [0.0, 1.0]
  is_verified            = score >= 0.67        (≥ 2 clean passes)

Backend integration
-------------------
  When state == ALMState.COMPLETE:

    session.active_liveness_score   float   0.0–1.0
    session.is_verified             bool
    session.results                 list[ChallengeResult]
    session.session_duration_s      float
    session.backend_payload         dict    (JSON-serialisable)

  Example payload::

    {
      "active_liveness_score": 0.8733,
      "is_verified": true,
      "challenges": [
        {"type": "gesture", "passed": true,  "quality": 1.0,  "duration_s": 2.8,  "detail": "SHOW 3 FINGERS!"},
        {"type": "touch",   "passed": true,  "quality": 1.0,  "duration_s": 3.9,  "detail": "PINCH: THUMB + INDEX"},
        {"type": "trace",   "passed": true,  "quality": 0.62, "duration_s": 10.1, "detail": "TRACE THE CIRCLE!"}
      ],
      "session_duration_s": 24.3
    }

Stability contract
------------------
  This module is a pure orchestrator — it does NOT rewrite any underlying
  challenge logic.  All coordinate systems, thresholds, and frame-level
  detectors (DTW, Z-axis gate, tremor detector) remain intact inside their
  respective session classes.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from hand_tracker import HandResult
from liveness_session import (
    LivenessChallenge, LivenessState, CmdType,
    _GESTURE_CMDS, _TOUCH_CMDS, _SHAPE_CMDS,
)
from math_session import MathSession, MathState
from shape_tracer import TracerState


# ── Configuration ────────────────────────────────────────────────────

NUM_CHALLENGES        = 3     # challenges per session
VERIFIED_THRESHOLD    = 0.67  # minimum score to be considered verified
TRANSITION_DURATION   = 2.0   # seconds for "Next Task" screen
MATH_TIME_LIMIT       = 30.0  # seconds per math challenge
RESULT_DISPLAY_TIME   = 1.2   # seconds to show pass/fail before advancing


# ── Public types ─────────────────────────────────────────────────────

class ChallengeType(Enum):
    MATH    = "math"
    GESTURE = "gesture"
    TOUCH   = "touch"
    TRACE   = "trace"

    @property
    def label(self) -> str:
        return self.value.upper()

    @property
    def description(self) -> str:
        return {
            "math":    "Solve the equation — show the answer in fingers",
            "gesture": "Perform the hand gesture shown on screen",
            "touch":   "Pinch your fingertips together and hold",
            "trace":   "Trace the shape from START to END point",
        }[self.value]


@dataclass
class ChallengeResult:
    """Outcome of a single completed challenge within the session."""
    challenge_type: ChallengeType
    passed:         bool
    quality:        float   # 0.0–1.0  (1.0 for gesture/touch/math; similarity for trace)
    duration_s:     float
    detail:         str     # human-readable label shown in the breakdown HUD


class ALMState(Enum):
    TRANSITION = auto()   # "Next Task" inter-challenge countdown screen
    CHALLENGE  = auto()   # challenge is live and accepting input
    COMPLETE   = auto()   # all challenges done; final score computed


# ── Internal challenge adapters ──────────────────────────────────────
# Each adapter wraps one existing session class and exposes a uniform:
#   update(hands) -> Optional[bool]   None = in progress, True = pass, False = fail
# The adapter owns result-display timing so the orchestrator stays simple.

class _MathAdapter:
    """Wraps MathSession.  Succeeds on the first correct answer; fails on timeout."""

    def __init__(self) -> None:
        self._ses = MathSession(
            stability_seconds=2.0,
            pause_after_success=1.5,
            game_duration=MATH_TIME_LIMIT,
            smoothing_window=7,
        )
        self._start       = time.time()
        self._success_at: Optional[float] = None

    # -- Adapter public API -----------------------------------------------

    def update(self, hands: list[HandResult]) -> Optional[bool]:
        state = self._ses.update(hands)

        # Once success is captured, hold the display for RESULT_DISPLAY_TIME.
        if self._success_at is not None:
            if time.time() - self._success_at >= RESULT_DISPLAY_TIME:
                return True
            return None

        if state == MathState.SUCCESS:
            self._success_at = time.time()
            return None   # show success screen first

        if state == MathState.GAME_OVER:
            return False

        return None

    # -- Display helpers --------------------------------------------------

    @property
    def label(self) -> str:
        return self._ses.equation_text

    @property
    def time_remaining(self) -> float:
        return self._ses.time_remaining

    @property
    def hold_progress(self) -> float:
        return self._ses.hold_progress

    @property
    def math_state(self) -> MathState:
        return self._ses.state

    @property
    def detected_total(self) -> Optional[int]:
        return self._ses.detected_total()

    def per_hand_counts(self, hands: list[HandResult]) -> dict[str, int]:
        return self._ses.per_hand_counts(hands)

    @property
    def score(self) -> int:
        return self._ses.score

    @property
    def answer(self) -> int:
        return self._ses.answer


class _LivenessAdapter:
    """
    Wraps LivenessChallenge restricted to a single randomly-chosen command
    from the supplied pool.  Handles result-display timing internally.
    """

    def __init__(self, pool: list, challenge_type: ChallengeType) -> None:
        chosen_cmd = random.choice(pool)

        # Build a 1-challenge liveness session, then override the queue so it
        # runs exactly our chosen command.  Calling _pick_next() a second time
        # is safe — it simply overwrites the transient state set by __post_init__.
        self._lv = LivenessChallenge(
            time_limit=4.0,
            debounce_seconds=0.5,
            area_change_threshold=0.20,
            pause_after_result=RESULT_DISPLAY_TIME,
            smoothing_window=7,
            num_challenges=1,
        )
        self._lv._challenge_queue  = [chosen_cmd]
        self._lv.challenges_completed = 0
        self._lv._pick_next()

        self._type      = challenge_type
        self._failed_at: Optional[float] = None

    # -- Adapter public API -----------------------------------------------

    def update(self, hands: list[HandResult]) -> Optional[bool]:
        state = self._lv.update(hands)

        # SUCCESS path: liveness internals wait pause_after_result then call
        # _pick_next() which transitions to VERIFIED_100 (queue exhausted).
        if state == LivenessState.VERIFIED_100:
            return True

        # FAILED path: capture first FAILED frame, then wait for the display
        # pause before returning False.
        if self._failed_at is None and state == LivenessState.FAILED:
            self._failed_at = time.time()

        if self._failed_at is not None:
            if time.time() - self._failed_at >= self._lv.pause_after_result:
                return False

        return None

    # -- Display helpers --------------------------------------------------

    @property
    def inner(self) -> LivenessChallenge:
        """Direct access to the wrapped LivenessChallenge for HUD rendering."""
        return self._lv

    @property
    def label(self) -> str:
        return self._lv.current_cmd.name if self._lv.current_cmd else ""

    @property
    def challenge_type(self) -> ChallengeType:
        return self._type

    @property
    def quality(self) -> float:
        """1.0 for gesture/touch; DTW similarity fraction for trace."""
        if (self._type == ChallengeType.TRACE
                and self._lv.shape_tracer is not None):
            return min(1.0, self._lv.shape_tracer.similarity_pct / 100.0)
        return 1.0


# ── Main orchestrator ────────────────────────────────────────────────

class ActiveLivenessSession:
    """
    Orchestrates a 3-challenge active liveness session for FIVUCSAS.

    Picks 3 challenge types at random (no repeats within one session),
    runs them sequentially with brief transition screens, and computes
    a final active_liveness_score once all challenges are complete.

    Usage::

        session = ActiveLivenessSession()
        while True:
            state = session.update(hands)
            if state == ALMState.COMPLETE:
                payload = session.backend_payload
                break
    """

    def __init__(self) -> None:
        self.state:                 ALMState       = ALMState.TRANSITION
        self.results:               list[ChallengeResult] = []
        self.active_liveness_score: float          = 0.0
        self.is_verified:           bool           = False
        self.session_duration_s:    float          = 0.0

        # Pick NUM_CHALLENGES types without repetition.
        pool = list(ChallengeType)
        random.shuffle(pool)
        self._types:    list[ChallengeType]             = pool[:NUM_CHALLENGES]
        self._idx:      int                             = 0
        self._current:  Optional[_MathAdapter | _LivenessAdapter] = None

        self._transition_start: float = time.time()
        self._challenge_start:  float = 0.0
        self._session_start:    float = time.time()

        # Pre-build the first adapter so it's ready when the transition ends.
        self._next = self._build_adapter(self._types[0])

    # -- Construction helpers ----------------------------------------

    def _build_adapter(
        self, ct: ChallengeType
    ) -> _MathAdapter | _LivenessAdapter:
        if ct == ChallengeType.MATH:
            return _MathAdapter()
        if ct == ChallengeType.GESTURE:
            return _LivenessAdapter(_GESTURE_CMDS, ChallengeType.GESTURE)
        if ct == ChallengeType.TOUCH:
            return _LivenessAdapter(_TOUCH_CMDS, ChallengeType.TOUCH)
        if ct == ChallengeType.TRACE:
            return _LivenessAdapter(_SHAPE_CMDS, ChallengeType.TRACE)
        raise ValueError(f"Unknown ChallengeType: {ct}")

    # -- Main update -------------------------------------------------

    def update(self, hands: list[HandResult]) -> ALMState:
        now = time.time()

        if self.state == ALMState.COMPLETE:
            return self.state

        # -- TRANSITION -----------------------------------------------
        if self.state == ALMState.TRANSITION:
            if now - self._transition_start >= TRANSITION_DURATION:
                self._current        = self._next
                self._challenge_start = now
                self.state           = ALMState.CHALLENGE
            return self.state

        # -- CHALLENGE ------------------------------------------------
        result = self._current.update(hands)

        if result is None:
            return self.state   # challenge still running

        # Challenge concluded — record result.
        ct       = self._types[self._idx]
        duration = round(now - self._challenge_start, 2)

        if result:  # passed
            quality = getattr(self._current, "quality", 1.0)
            self.results.append(ChallengeResult(
                challenge_type=ct,
                passed=True,
                quality=round(quality, 4),
                duration_s=duration,
                detail=self._current.label,
            ))
        else:       # failed
            self.results.append(ChallengeResult(
                challenge_type=ct,
                passed=False,
                quality=0.0,
                duration_s=duration,
                detail=self._current.label,
            ))

        self._idx += 1

        if self._idx >= NUM_CHALLENGES:
            self._finalise(now)
        else:
            # Pre-build next adapter during the transition screen.
            self._next             = self._build_adapter(self._types[self._idx])
            self._transition_start = now
            self.state             = ALMState.TRANSITION

        return self.state

    # -- Finalisation ------------------------------------------------

    def _finalise(self, now: float) -> None:
        self.session_duration_s    = round(now - self._session_start, 2)
        contributions              = [(r.quality if r.passed else 0.0) for r in self.results]
        self.active_liveness_score = round(sum(contributions) / NUM_CHALLENGES, 4)
        self.is_verified           = self.active_liveness_score >= VERIFIED_THRESHOLD
        self.state                 = ALMState.COMPLETE

    # -- Display helpers ---------------------------------------------

    @property
    def progress_text(self) -> str:
        done = len(self.results)
        return f"Task {min(done + 1, NUM_CHALLENGES)} / {NUM_CHALLENGES}"

    @property
    def current_type(self) -> ChallengeType:
        """Challenge type that is currently active (or last if complete)."""
        return self._types[min(self._idx, NUM_CHALLENGES - 1)]

    @property
    def upcoming_type(self) -> Optional[ChallengeType]:
        """Type of the challenge waiting behind the current transition screen."""
        return self._types[self._idx] if self._idx < NUM_CHALLENGES else None

    @property
    def transition_remaining(self) -> float:
        """Seconds left on the TRANSITION countdown (0 if not in transition)."""
        if self.state != ALMState.TRANSITION:
            return 0.0
        return max(0.0, TRANSITION_DURATION - (time.time() - self._transition_start))

    @property
    def results_text(self) -> list[str]:
        """One line per completed challenge for the breakdown panel."""
        lines = []
        for i, r in enumerate(self.results):
            icon    = "✓" if r.passed else "✗"
            q_str   = f"  ({r.quality*100:.0f}%)" if r.challenge_type == ChallengeType.TRACE else ""
            lines.append(f"{icon} {r.challenge_type.label}{q_str}  {r.duration_s:.1f}s")
        return lines

    @property
    def backend_payload(self) -> dict:
        """JSON-serialisable dict ready for backend transmission."""
        return {
            "active_liveness_score": self.active_liveness_score,
            "is_verified":           self.is_verified,
            "challenges": [
                {
                    "type":       r.challenge_type.value,
                    "passed":     r.passed,
                    "quality":    r.quality,
                    "duration_s": r.duration_s,
                    "detail":     r.detail,
                }
                for r in self.results
            ],
            "session_duration_s": self.session_duration_s,
        }

    def reset(self) -> None:
        """Restart the session with a fresh random challenge sequence."""
        self.__init__()
