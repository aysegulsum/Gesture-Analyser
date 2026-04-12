"""
Finger Touch Detector -- Liveness Challenge Module
===================================================
Detects specific fingertip pinch gestures with three layers of protection:

  1. Normalized Euclidean distance
       Distance between the two landmarks is divided by the hand bounding-box
       diagonal, making the threshold invariant to hand size and camera distance.

  2. Z-axis depth validation
       Rejects cases where one finger passes *behind* the other in 3D space.
       This blocks the common 2D spoofing attack where fingers look adjacent
       in the image plane but are at very different depths.

  3. 10-frame consecutive hold requirement
       A fleeting or accidental brush does not satisfy the challenge.
       The touch must be maintained for ``verify_frames`` (default 10)
       consecutive video frames before the gesture is marked 'Verified'.

Supported commands
------------------
  THUMB_TO_INDEX    -- Landmark 4 (thumb tip) & 8  (index tip), same hand
  THUMB_TO_MIDDLE   -- Landmark 4 (thumb tip) & 12 (middle tip), same hand
  THUMB_TO_RING     -- Landmark 4 (thumb tip) & 16 (ring tip),   same hand
  THUMB_TO_PINKY    -- Landmark 4 (thumb tip) & 20 (pinky tip),  same hand
  DOUBLE_THUMB_TOUCH-- Landmark 4 Left hand  & 4  Right hand
"""

import math
from enum import Enum, auto

from gesture_validator import _euclidean, _LM


# ── Command catalogue ────────────────────────────────────────────────

class TouchCommand(Enum):
    THUMB_TO_INDEX     = auto()   # IDs 4 & 8  (same hand)
    THUMB_TO_MIDDLE    = auto()   # IDs 4 & 12 (same hand)
    THUMB_TO_RING      = auto()   # IDs 4 & 16 (same hand)
    THUMB_TO_PINKY     = auto()   # IDs 4 & 20 (same hand)
    DOUBLE_THUMB_TOUCH = auto()   # ID 4 Left hand & ID 4 Right hand


# Single-hand landmark pairs (tip_a_id, tip_b_id)
_SINGLE_HAND_PAIRS: dict[TouchCommand, tuple[int, int]] = {
    TouchCommand.THUMB_TO_INDEX:  (_LM["THUMB_TIP"], _LM["INDEX_TIP"]),   # 4, 8
    TouchCommand.THUMB_TO_MIDDLE: (_LM["THUMB_TIP"], _LM["MIDDLE_TIP"]),  # 4, 12
    TouchCommand.THUMB_TO_RING:   (_LM["THUMB_TIP"], _LM["RING_TIP"]),    # 4, 16
    TouchCommand.THUMB_TO_PINKY:  (_LM["THUMB_TIP"], _LM["PINKY_TIP"]),   # 4, 20
}

# Default detector parameters
_DEFAULT_VERIFY_FRAMES   = 10    # frames of continuous touch required
_DEFAULT_TOUCH_THRESHOLD = 0.28  # fraction of bbox diagonal
_DEFAULT_Z_MAX_DIFF      = 0.04  # maximum allowed Z-depth gap between tips


# ── Geometry helpers ─────────────────────────────────────────────────

def _bbox_scale(landmarks) -> float:
    """Diagonal of the hand bounding box in normalised image coordinates.

    Grows when the hand is large (close to camera) and shrinks when small
    (far away), so dividing touch distances by this value makes the
    ``touch_threshold`` invariant to scale.
    """
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    diag = math.sqrt(w * w + h * h)
    return diag if diag > 1e-9 else 1e-9


def _z_valid(lm_a, lm_b, z_max_diff: float) -> bool:
    """Return True when both landmarks share a similar depth plane.

    MediaPipe Z values are positive further from the camera.  A large
    difference means one fingertip is behind the other — they only *look*
    close in the 2D image but are not actually touching.
    """
    return abs(lm_a.z - lm_b.z) <= z_max_diff


# ── Main detector class ──────────────────────────────────────────────

class FingerTouchDetector:
    """Per-frame detector for fingertip touch liveness challenges.

    Instantiate once per challenge, feed every video frame via ``update()``,
    and read ``is_verified`` (or use the bool return value) to know when the
    touch has been confirmed.

    Parameters
    ----------
    command : TouchCommand
        Which fingertip pair to watch.
    verify_frames : int
        Consecutive frames the touch must be held to become Verified.
        Default 10 (~0.33 s at 30 fps).
    touch_threshold : float
        Maximum normalised distance (as a fraction of the hand bounding-box
        diagonal) for the tips to count as touching.
    z_max_diff : float
        Maximum allowed Z-coordinate gap between the two landmark tips.
        Larger values tolerate more depth offset; smaller values are stricter
        about rejecting 2-D spoofing.
    """

    def __init__(
        self,
        command: TouchCommand,
        verify_frames: int  = _DEFAULT_VERIFY_FRAMES,
        touch_threshold: float = _DEFAULT_TOUCH_THRESHOLD,
        z_max_diff: float   = _DEFAULT_Z_MAX_DIFF,
    ):
        self.command         = command
        self.verify_frames   = verify_frames
        self.touch_threshold = touch_threshold
        self.z_max_diff      = z_max_diff
        self._consecutive    = 0
        self._verified       = False

    # -- Internal per-frame geometry checks ------------------------------

    def _check_single_hand(self, landmarks) -> bool:
        """Evaluate touch for same-hand landmark pairs."""
        tip_a_id, tip_b_id = _SINGLE_HAND_PAIRS[self.command]
        lm_a = landmarks[tip_a_id]
        lm_b = landmarks[tip_b_id]

        # Layer 2: Z-axis anti-spoof gate
        if not _z_valid(lm_a, lm_b, self.z_max_diff):
            return False

        # Layer 1: Normalised Euclidean distance
        scale = _bbox_scale(landmarks)
        dist  = _euclidean(lm_a, lm_b) / scale
        return dist < self.touch_threshold

    def _check_double_thumb(self, hands) -> bool:
        """Evaluate DOUBLE_THUMB_TOUCH across left and right hands."""
        left_lm = right_lm = None
        for h in hands:
            if h.handedness == "Left":
                left_lm = h.landmarks
            elif h.handedness == "Right":
                right_lm = h.landmarks

        if left_lm is None or right_lm is None:
            return False  # both hands must be present

        lm_a = left_lm[_LM["THUMB_TIP"]]   # left  thumb tip (ID 4)
        lm_b = right_lm[_LM["THUMB_TIP"]]  # right thumb tip (ID 4)

        # Layer 2: Z-axis anti-spoof gate
        if not _z_valid(lm_a, lm_b, self.z_max_diff):
            return False

        # Normalise against the *average* of both hand bounding boxes so
        # that the threshold does not depend on inter-hand separation.
        scale = (_bbox_scale(left_lm) + _bbox_scale(right_lm)) / 2.0
        dist  = _euclidean(lm_a, lm_b) / scale
        return dist < self.touch_threshold

    def _is_touching_this_frame(self, hands) -> bool:
        """Raw, single-frame touch check (no temporal component)."""
        if not hands:
            return False
        if self.command == TouchCommand.DOUBLE_THUMB_TOUCH:
            return self._check_double_thumb(hands)
        return self._check_single_hand(hands[0].landmarks)

    # -- Public API ------------------------------------------------------

    def update(self, hands) -> bool:
        """Feed one video frame and check whether the touch is now Verified.

        Returns True on the frame that verification is first achieved, and
        on all subsequent frames (the verified state is sticky until
        ``reset()`` is called).

        Layer 3 -- consecutive-frame persistence:
          The touch must be held for ``verify_frames`` unbroken frames.
          Any frame without a valid touch resets the counter to zero.
        """
        if self._verified:
            return True

        if self._is_touching_this_frame(hands):
            self._consecutive += 1
            if self._consecutive >= self.verify_frames:
                self._verified = True
                return True
        else:
            self._consecutive = 0

        return False

    # -- Observable state ------------------------------------------------

    @property
    def consecutive_frames(self) -> int:
        """Current unbroken frame count toward verification (0 … verify_frames)."""
        return self._consecutive

    @property
    def progress(self) -> float:
        """Fraction of verify_frames reached (0.0 – 1.0), for progress bars."""
        return min(self._consecutive / self.verify_frames, 1.0)

    @property
    def is_verified(self) -> bool:
        """True once the touch has been held for the required number of frames."""
        return self._verified

    def reset(self) -> None:
        """Return the detector to its initial state."""
        self._consecutive = 0
        self._verified    = False
