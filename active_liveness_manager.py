"""
Active Liveness Manager  --  FIVUCSAS
======================================
Unified multi-task active liveness orchestrator for the FIVUCSAS biometric
authentication system.

Selects 5 challenges from the pool (all 4 types guaranteed, 1 random repeat):

  MATH     — solve a simple arithmetic question with fingers (0-10 result)
  GESTURE  — show a specific number / configuration of fingers
  TOUCH    — pinch specific fingertips (Z-validated, 10-frame hold)
  TRACE    — follow a visual shape template (DTW-verified, point-to-point)
             TRACE gets 1 automatic retry if the first attempt fails.
             Final quality = mean(attempt_qualities).

State machine
-------------
  TRANSITION  →  brief inter-challenge screen ("Next Task..." 2 s)
  CHALLENGE   →  one challenge is live and accepting input
  COMPLETE    →  all 5 challenges done; active_liveness_score computed

Scoring
-------
  Each of the 5 challenges contributes equally (weight = 1/5).

  Pass (GESTURE / TOUCH / MATH)  →  quality = 1.0
  Pass (TRACE, attempt 1)        →  quality = similarity_pct / 100
  Pass (TRACE, attempt 2 retry)  →  quality = mean(attempt1_q, attempt2_q)
  Fail (any, all retries used)   →  quality = 0.0

  active_liveness_score = mean(qualities)       ∈ [0.0, 1.0]
  is_verified            = score >= 0.67        (≥ 4 clean passes out of 5,
                                                 or 3 passes + good trace quality)

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
      "active_liveness_score": 0.764,
      "is_verified": true,
      "challenges": [
        {"type": "gesture", "passed": true,  "quality": 1.0,  "duration_s": 2.8,  "detail": "SHOW 3 FINGERS!", "attempts": 1},
        {"type": "math",    "passed": true,  "quality": 1.0,  "duration_s": 6.1,  "detail": "4 + 3 = ?",       "attempts": 1},
        {"type": "touch",   "passed": true,  "quality": 1.0,  "duration_s": 3.9,  "detail": "PINCH: THUMB + INDEX", "attempts": 1},
        {"type": "trace",   "passed": false, "quality": 0.0,  "duration_s": 14.2, "detail": "TRACE THE CIRCLE!", "attempts": 2},
        {"type": "gesture", "passed": true,  "quality": 1.0,  "duration_s": 2.1,  "detail": "SHOW 2 FINGERS!", "attempts": 1}
      ],
      "session_duration_s": 47.8
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
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from hand_tracker import HandResult
from liveness_session import (
    LivenessChallenge, LivenessState,
    _GESTURE_CMDS, _TOUCH_CMDS, _SHAPE_CMDS,
)
from math_session import MathSession, MathState
from shape_tracer import TracerState


# ── Configuration ────────────────────────────────────────────────────

NUM_CHALLENGES      = 5     # challenges per session (all 4 types + 1 random repeat)
VERIFIED_THRESHOLD  = 0.67  # min score to be verified (~4/5 clean passes)
TRANSITION_DURATION = 2.0   # seconds for "Next Task" screen
MATH_TIME_LIMIT     = 30.0  # seconds per math challenge
RESULT_DISPLAY_TIME = 1.2   # seconds to show pass/fail screen before advancing
TRACE_MAX_ATTEMPTS  = 2     # 1 retry allowed on TRACE failure
TRACE_RETRY_DISPLAY = 2.0   # seconds for the "Retrying..." screen between attempts


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
    quality:        float   # 0.0–1.0
    duration_s:     float
    detail:         str     # human-readable label for the HUD breakdown
    attempts:       int     # 1 normally; 2 if TRACE retry was used


class ALMState(Enum):
    TRANSITION = auto()   # "Next Task" inter-challenge countdown screen
    CHALLENGE  = auto()   # challenge is live
    COMPLETE   = auto()   # all challenges done; final score computed


# ── Internal challenge adapters ──────────────────────────────────────
# Each adapter exposes a uniform interface:
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
        self._success_at: Optional[float] = None
        self.quality: float = 1.0
        self.attempts: int  = 1

    def update(self, hands: list[HandResult]) -> Optional[bool]:
        state = self._ses.update(hands)

        if self._success_at is not None:
            if time.time() - self._success_at >= RESULT_DISPLAY_TIME:
                return True
            return None

        if state == MathState.SUCCESS:
            self._success_at = time.time()
            return None

        if state == MathState.GAME_OVER:
            self.quality = 0.0
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
    def answer(self) -> int:
        return self._ses.answer


class _LivenessAdapter:
    """
    Wraps LivenessChallenge restricted to a single randomly-chosen command
    from the supplied pool.  Handles result-display timing internally.
    """

    def __init__(self, pool: list, challenge_type: ChallengeType) -> None:
        chosen_cmd = random.choice(pool)

        self._lv = LivenessChallenge(
            time_limit=4.0,
            debounce_seconds=0.5,
            area_change_threshold=0.20,
            pause_after_result=RESULT_DISPLAY_TIME,
            smoothing_window=7,
            num_challenges=1,
        )
        # Override the queue with exactly our chosen command.
        self._lv._challenge_queue     = [chosen_cmd]
        self._lv.challenges_completed = 0
        self._lv._pick_next()

        self._challenge_type = challenge_type
        self._failed_at: Optional[float] = None
        self.attempts: int = 1

    def update(self, hands: list[HandResult]) -> Optional[bool]:
        state = self._lv.update(hands)

        # VERIFIED_100 means the single challenge was passed + pause elapsed.
        if state == LivenessState.VERIFIED_100:
            return True

        # Capture first FAILED frame; wait for display pause before returning.
        if self._failed_at is None and state == LivenessState.FAILED:
            self._failed_at = time.time()

        if self._failed_at is not None:
            if time.time() - self._failed_at >= self._lv.pause_after_result:
                return False

        return None

    # -- Display helpers --------------------------------------------------

    @property
    def inner(self) -> LivenessChallenge:
        return self._lv

    @property
    def label(self) -> str:
        return self._lv.current_cmd.name if self._lv.current_cmd else ""

    @property
    def challenge_type(self) -> ChallengeType:
        return self._challenge_type

    @property
    def quality(self) -> float:
        if (self._challenge_type == ChallengeType.TRACE
                and self._lv.shape_tracer is not None):
            return min(1.0, self._lv.shape_tracer.similarity_pct / 100.0)
        return 1.0


class _TraceWithRetryAdapter:
    """
    Wraps up to TRACE_MAX_ATTEMPTS _LivenessAdapter(TRACE) instances.

    On first-attempt failure a 'Retrying...' screen is shown for
    TRACE_RETRY_DISPLAY seconds, then a fresh TRACE challenge starts.
    The final quality is the mean of all attempt qualities so that a poor
    first attempt is still penalised even if the retry passes.

    Scoring examples
    ----------------
    attempt 1 pass  (sim=0.85)                 → quality = 0.85   attempts=1
    attempt 1 fail, attempt 2 pass (sim=0.70)  → quality = 0.35   attempts=2
    attempt 1 fail, attempt 2 fail             → quality = 0.0    attempts=2
    """

    def __init__(self) -> None:
        self._attempt_num: int         = 1
        self._adapter                  = _LivenessAdapter(_SHAPE_CMDS, ChallengeType.TRACE)
        self._qualities: list[float]   = []

        # Retry-screen state
        self._retry_start: Optional[float] = None

        # Final outcome
        self._done: Optional[bool] = None
        self.quality:   float      = 0.0
        self.attempts:  int        = 1

    # -- Adapter public API -----------------------------------------------

    def update(self, hands: list[HandResult]) -> Optional[bool]:
        if self._done is not None:
            return self._done

        # --- Retry screen (between attempts) ---
        if self._retry_start is not None:
            if time.time() - self._retry_start >= TRACE_RETRY_DISPLAY:
                # Start fresh attempt
                self._retry_start = None
                self._adapter     = _LivenessAdapter(_SHAPE_CMDS, ChallengeType.TRACE)
            return None     # hold display during retry countdown

        # --- Active attempt ---
        result = self._adapter.update(hands)

        if result is None:
            return None

        # Attempt concluded
        q = self._adapter.quality if result else 0.0
        self._qualities.append(q)
        self.attempts = self._attempt_num

        if result:
            # Passed — average all attempt qualities
            self.quality = round(sum(self._qualities) / len(self._qualities), 4)
            self._done   = True
            return True

        # Failed this attempt
        if self._attempt_num < TRACE_MAX_ATTEMPTS:
            # Queue the retry screen
            self._attempt_num += 1
            self._retry_start  = time.time()
            return None         # keep returning None until retry screen times out

        # All attempts exhausted
        self.quality = 0.0
        self._done   = False
        return False

    # -- Display helpers --------------------------------------------------

    @property
    def inner(self) -> LivenessChallenge:
        """Current attempt's LivenessChallenge (for HUD rendering)."""
        return self._adapter.inner

    @property
    def label(self) -> str:
        return self._adapter.label

    @property
    def challenge_type(self) -> ChallengeType:
        return ChallengeType.TRACE

    @property
    def in_retry_screen(self) -> bool:
        return self._retry_start is not None

    @property
    def retry_remaining(self) -> float:
        if self._retry_start is None:
            return 0.0
        return max(0.0, TRACE_RETRY_DISPLAY - (time.time() - self._retry_start))

    @property
    def attempt_num(self) -> int:
        return self._attempt_num

    @property
    def attempt_qualities(self) -> list[float]:
        return list(self._qualities)


# ── Pool selection ───────────────────────────────────────────────────

def _pick_challenge_types() -> list[ChallengeType]:
    """
    Return a shuffled list of NUM_CHALLENGES (5) challenge types.

    All 4 types are always present; one is repeated at a random position.
    The repeated type is chosen randomly so no modality is systematically
    over-represented and the order is unpredictable.
    """
    pool  = list(ChallengeType)         # [MATH, GESTURE, TOUCH, TRACE]
    random.shuffle(pool)
    extra = random.choice(pool)         # one random repeat
    types = pool + [extra]              # 5 entries, all 4 + 1 duplicate
    random.shuffle(types)               # mix the repeat into a random position
    return types


# ── Main orchestrator ────────────────────────────────────────────────

class ActiveLivenessSession:
    """
    Orchestrates a 5-challenge active liveness session for FIVUCSAS.

    All 4 challenge types are always present; one is repeated at a random
    position.  TRACE challenges get 1 automatic retry on failure.

    Usage::

        session = ActiveLivenessSession()
        while True:
            state = session.update(hands)
            if state == ALMState.COMPLETE:
                payload = session.backend_payload
                break
    """

    def __init__(self) -> None:
        self.state:                 ALMState              = ALMState.TRANSITION
        self.results:               list[ChallengeResult] = []
        self.active_liveness_score: float                 = 0.0
        self.is_verified:           bool                  = False
        self.session_duration_s:    float                 = 0.0

        self._types:  list[ChallengeType] = _pick_challenge_types()
        self._idx:    int                 = 0
        self._current: Optional[
            _MathAdapter | _LivenessAdapter | _TraceWithRetryAdapter
        ] = None

        self._transition_start: float = time.time()
        self._challenge_start:  float = 0.0
        self._session_start:    float = time.time()

        # Pre-build first adapter so it's ready when the transition ends.
        self._next = self._build_adapter(self._types[0])

    # -- Construction helpers ----------------------------------------

    def _build_adapter(self, ct: ChallengeType):
        if ct == ChallengeType.MATH:
            return _MathAdapter()
        if ct == ChallengeType.GESTURE:
            return _LivenessAdapter(_GESTURE_CMDS, ChallengeType.GESTURE)
        if ct == ChallengeType.TOUCH:
            return _LivenessAdapter(_TOUCH_CMDS, ChallengeType.TOUCH)
        if ct == ChallengeType.TRACE:
            return _TraceWithRetryAdapter()
        raise ValueError(f"Unknown ChallengeType: {ct}")

    # -- Main update -------------------------------------------------

    def update(self, hands: list[HandResult]) -> ALMState:
        now = time.time()

        if self.state == ALMState.COMPLETE:
            return self.state

        # -- TRANSITION -----------------------------------------------
        if self.state == ALMState.TRANSITION:
            if now - self._transition_start >= TRANSITION_DURATION:
                self._current         = self._next
                self._challenge_start = now
                self.state            = ALMState.CHALLENGE
            return self.state

        # -- CHALLENGE ------------------------------------------------
        result = self._current.update(hands)

        if result is None:
            return self.state   # still running (or in retry screen)

        # Challenge concluded — record result.
        ct       = self._types[self._idx]
        duration = round(now - self._challenge_start, 2)
        quality  = getattr(self._current, "quality", 1.0 if result else 0.0)
        attempts = getattr(self._current, "attempts", 1)

        self.results.append(ChallengeResult(
            challenge_type=ct,
            passed=bool(result),
            quality=round(quality if result else 0.0, 4),
            duration_s=duration,
            detail=self._current.label,
            attempts=attempts,
        ))

        self._idx += 1

        if self._idx >= NUM_CHALLENGES:
            self._finalise(now)
        else:
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
        return self._types[min(self._idx, NUM_CHALLENGES - 1)]

    @property
    def upcoming_type(self) -> Optional[ChallengeType]:
        return self._types[self._idx] if self._idx < NUM_CHALLENGES else None

    @property
    def transition_remaining(self) -> float:
        if self.state != ALMState.TRANSITION:
            return 0.0
        return max(0.0, TRANSITION_DURATION - (time.time() - self._transition_start))

    @property
    def results_text(self) -> list[str]:
        lines = []
        for r in self.results:
            icon  = "PASS" if r.passed else "FAIL"
            q_str = f" ({r.quality*100:.0f}%)" if r.challenge_type == ChallengeType.TRACE else ""
            a_str = f" x{r.attempts}" if r.attempts > 1 else ""
            lines.append(f"{icon}  {r.challenge_type.label}{q_str}{a_str}  {r.duration_s:.1f}s")
        return lines

    @property
    def backend_payload(self) -> dict:
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
                    "attempts":   r.attempts,
                }
                for r in self.results
            ],
            "session_duration_s": self.session_duration_s,
        }

    def reset(self) -> None:
        """Restart the session with a fresh random challenge sequence."""
        self.__init__()
