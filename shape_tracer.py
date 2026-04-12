"""
Dynamic Shape Tracing -- Liveness Challenge Module
===================================================
The user must trace a randomly generated geometric shape (Circle, Square,
Triangle, or S-Curve) using their index finger in real-time.

Architecture
------------

              ┌────────────┐
   start      │            │  show shape guide on screen;
  ───────────►│  WAITING   │  await index finger at sufficient depth
              │            │
              └─────┬──────┘
                    │ index finger extended (depth gate passed)
                    ▼
              ┌────────────┐
              │            │  record Index Tip (ID 8) path each frame;
              │  DRAWING   │  overlay trace on screen;
              │            │  liveness gate: hand_scale > min_hand_scale
              └──┬──────┬──┘
     fist / done │      │ time expired
                 ▼      ▼
              ┌────────────┐
              │            │  single-frame DTW computation:
              │ VERIFYING  │    1. resample both paths to N points
              │            │    2. centroid-normalise (translation + scale invariant)
              └──┬──────┬──┘    3. DTW cost matrix (numpy distance mat + Python DP)
      cost ≤ th  │      │  cost > th / too few points
                 ▼      ▼
          ┌────────┐  ┌────────┐
          │VERIFIED│  │ FAILED │
          └────────┘  └────────┘

Shape templates (normalised screen coords, [0, 1]²)
----------------------------------------------------
  CIRCLE   -- 49-point parametric circle, centre (0.50, 0.50), r = 0.18
  SQUARE   -- 5-waypoint closed rectangle
  TRIANGLE -- 4-waypoint closed triangle
  S_CURVE  -- 18 control points (two opposing quarter-circle arcs)

Verification algorithm (DTW)
-----------------------------
  1. Resample both paths to N = 50 evenly-spaced points (arc-length).
  2. Centroid-normalise: subtract centroid, divide by max radius.
     → invariant to translation AND scale; only shape matters.
  3. Build Euclidean distance matrix D[i,j] with numpy (vectorised).
  4. Fill DTW cost matrix with a Python DP loop (2 500 iterations ≈ 2 ms).
  5. Normalise total cost by N → per-point average distance.
  6. Accept if normalised DTW cost ≤ dtw_threshold (default 0.25).

Similarity score: max(0, 1 - cost / threshold)  → 0 … 1  → multiplied by
100 for display.  Green ≥ 70 %, Yellow ≥ 40 %, Red below that.
"""

import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np

from hand_tracker import HandResult
from gesture_validator import hand_scale as _hs, is_finger_open, Finger, _LM


# ── Shape catalogue ──────────────────────────────────────────────────

class ShapeType(Enum):
    CIRCLE   = auto()
    SQUARE   = auto()
    TRIANGLE = auto()
    S_CURVE  = auto()


@dataclass(frozen=True)
class ShapeTemplate:
    """A target shape defined by ordered (x, y) waypoints in [0, 1]².

    Waypoints represent the expected tracing direction (clockwise for
    closed shapes, top-to-bottom for open curves).  They are drawn as a
    semi-transparent guide on the video frame.
    """
    shape_type: ShapeType
    label:      str
    waypoints:  tuple[tuple[float, float], ...]


# ── Template builders ────────────────────────────────────────────────

def _circle_template(cx: float = 0.50, cy: float = 0.50,
                     r: float = 0.18, n: int = 48) -> ShapeTemplate:
    """Full circle starting at the 3-o'clock position, traced clockwise."""
    pts = tuple(
        (round(cx + r * math.cos(2 * math.pi * i / n), 6),
         round(cy + r * math.sin(2 * math.pi * i / n), 6))
        for i in range(n + 1)   # +1 closes the loop back to start
    )
    return ShapeTemplate(ShapeType.CIRCLE, "CIRCLE", pts)


def _square_template(l: float = 0.32, t: float = 0.30,
                     r: float = 0.68, b: float = 0.70) -> ShapeTemplate:
    """Closed rectangle traced clockwise from the top-left corner."""
    return ShapeTemplate(ShapeType.SQUARE, "SQUARE", (
        (l, t), (r, t), (r, b), (l, b), (l, t),
    ))


def _triangle_template(cx: float = 0.50, ty: float = 0.25,
                        by: float = 0.72, hw: float = 0.22) -> ShapeTemplate:
    """Isosceles triangle: top → bottom-right → bottom-left → top."""
    return ShapeTemplate(ShapeType.TRIANGLE, "TRIANGLE", (
        (cx,       ty),
        (cx + hw,  by),
        (cx - hw,  by),
        (cx,       ty),
    ))


def _s_curve_template() -> ShapeTemplate:
    """S-shape from two opposing quarter-circle arcs, traced top-to-bottom."""
    pts: list[tuple[float, float]] = []
    # Upper arc: concave-right  (centre 0.62, 0.375, radius ≈ 0.125)
    for i in range(9):
        a = math.pi + (math.pi / 2) * (i / 8)      # π → 3π/2
        pts.append((round(0.62 + 0.14 * math.cos(a), 6),
                    round(0.375 + 0.125 * math.sin(a), 6)))
    # Lower arc: concave-left   (centre 0.38, 0.625, radius ≈ 0.125)
    for i in range(9):
        a = math.pi * 2 + (math.pi / 2) * (i / 8)  # 2π → 5π/2
        pts.append((round(0.38 + 0.14 * math.cos(a), 6),
                    round(0.625 + 0.125 * math.sin(a), 6)))
    return ShapeTemplate(ShapeType.S_CURVE, "S-CURVE", tuple(pts))


_TEMPLATES: list[ShapeTemplate] = [
    _circle_template(),
    _square_template(),
    _triangle_template(),
    _s_curve_template(),
]


def generate_random_shape() -> ShapeTemplate:
    """Return one of the four shape templates chosen at random."""
    return random.choice(_TEMPLATES)


# ── Path mathematics ─────────────────────────────────────────────────

def _resample(path: list[tuple[float, float]], n: int) -> list[tuple[float, float]]:
    """Resample a polyline to exactly *n* evenly-spaced points by arc-length.

    This ensures both the template and the traced path have the same number
    of points before DTW, eliminating length bias from the alignment cost.
    """
    if len(path) == 0:
        return [(0.0, 0.0)] * n
    if len(path) == 1:
        return [path[0]] * n

    # Cumulative arc-lengths
    dists: list[float] = [0.0]
    for i in range(1, len(path)):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]
        dists.append(dists[-1] + math.sqrt(dx * dx + dy * dy))
    total = dists[-1]
    if total < 1e-9:
        return [path[0]] * n

    resampled: list[tuple[float, float]] = []
    j = 0
    for k in range(n):
        target = k * total / (n - 1)
        while j < len(dists) - 2 and dists[j + 1] < target:
            j += 1
        seg = dists[j + 1] - dists[j]
        t = ((target - dists[j]) / seg) if seg > 1e-9 else 0.0
        x = path[j][0] + t * (path[j + 1][0] - path[j][0])
        y = path[j][1] + t * (path[j + 1][1] - path[j][1])
        resampled.append((x, y))
    return resampled


def _centroid_normalise(path: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Translate to centroid; scale by max radius → points in ≈ [-1, 1]².

    This makes the DTW comparison invariant to both translation and scale:
    a small circle traced in the corner and a large circle traced in the
    centre produce the same normalised representation.
    """
    n = len(path)
    if n == 0:
        return []
    cx = sum(p[0] for p in path) / n
    cy = sum(p[1] for p in path) / n
    centred = [(p[0] - cx, p[1] - cy) for p in path]
    max_r = max(math.sqrt(x * x + y * y) for x, y in centred) or 1.0
    return [(x / max_r, y / max_r) for x, y in centred]


def dtw_normalised_cost(
    path_a: list[tuple[float, float]],
    path_b: list[tuple[float, float]],
) -> float:
    """Return the DTW alignment cost normalised by path length.

    Steps
    -----
    1. Build a Euclidean distance matrix D[i, j] with numpy (vectorised).
    2. Fill the DTW cost table with a Python DP loop.
    3. Divide the total cost by max(n, m) → per-point average distance.

    Both input paths should already be resampled to the same length N.
    For N = 50, the DP loop runs 2 500 iterations ≈ 1–3 ms.
    """
    n, m = len(path_a), len(path_b)
    if n == 0 or m == 0:
        return float("inf")

    a = np.asarray(path_a, dtype=np.float32)   # (n, 2)
    b = np.asarray(path_b, dtype=np.float32)   # (m, 2)

    # Vectorised Euclidean distance matrix
    diff = a[:, None, :] - b[None, :, :]        # (n, m, 2)
    D: np.ndarray = np.sqrt((diff ** 2).sum(axis=-1))  # (n, m)

    # Standard DTW DP with (n+1)×(m+1) table, INF borders
    dtw = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = float(D[i - 1, j - 1])
            dtw[i, j] = cost + min(
                float(dtw[i - 1, j]),
                float(dtw[i, j - 1]),
                float(dtw[i - 1, j - 1]),
            )

    return float(dtw[n, m]) / max(n, m)


# ── Tracer state machine ─────────────────────────────────────────────

class TracerState(Enum):
    WAITING   = auto()  # shape displayed; waiting for index finger at depth
    DRAWING   = auto()  # recording Index Tip path in real time
    VERIFYING = auto()  # single-frame DTW computation
    VERIFIED  = auto()  # DTW cost ≤ threshold → accepted
    FAILED    = auto()  # timed out or DTW cost too high


# Defaults used by ShapeTracerSession and by liveness_session integration
DEFAULT_DRAW_TIME    = 8.0   # seconds per trace attempt
DEFAULT_DTW_THRESH   = 0.25  # normalised per-point DTW cost threshold
DEFAULT_RESAMPLE_N   = 50    # number of points for DTW comparison
DEFAULT_MIN_HS       = 0.10  # minimum hand_scale for depth liveness gate


@dataclass
class ShapeTracerSession:
    """State machine for one Dynamic Shape Tracing liveness challenge.

    Parameters
    ----------
    time_limit : float
        Seconds the user has to complete a single tracing attempt.
    min_hand_scale : float
        Minimum wrist-to-MiddleMCP distance (hand_scale) that counts as
        'hand in frame'.  Acts as a depth-based liveness gate: a printed
        photo of a hand will typically score below this threshold.
    dtw_threshold : float
        Maximum normalised per-point DTW cost accepted as a valid trace.
        Lower = stricter.  Default 0.25 corresponds to ~25% of the shape
        radius average error.
    resample_n : int
        Number of resampled points used in the DTW comparison.
    pause_after_result : float
        Seconds to hold VERIFIED / FAILED before the session may be reset.
    """

    time_limit:         float = DEFAULT_DRAW_TIME
    min_hand_scale:     float = DEFAULT_MIN_HS
    dtw_threshold:      float = DEFAULT_DTW_THRESH
    resample_n:         int   = DEFAULT_RESAMPLE_N
    pause_after_result: float = 2.0

    # -- observable (read by HUD) -----------------------------------------
    state:      TracerState = field(init=False, default=TracerState.WAITING)
    template:   ShapeTemplate = field(init=False)
    similarity: float = field(init=False, default=0.0)   # 0 … 1
    dtw_cost:   float = field(init=False, default=0.0)

    # -- internal ---------------------------------------------------------
    _traced:         list[tuple[float, float]] = field(init=False, default_factory=list)
    _raw_buf:        list[tuple[float, float]] = field(init=False, default_factory=list)
    _draw_start:     Optional[float] = field(init=False, default=None)
    _result_at:      Optional[float] = field(init=False, default=None)
    _was_index_open: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.template = generate_random_shape()

    # -- Point recording (3-point moving-average smoothing) ---------------

    def _push_point(self, x: float, y: float) -> None:
        """Append smoothed point; mirrors ShapeRecognizer.push() logic."""
        self._raw_buf.append((x, y))
        if len(self._raw_buf) > 3:
            self._raw_buf.pop(0)
        n  = len(self._raw_buf)
        sx = sum(p[0] for p in self._raw_buf) / n
        sy = sum(p[1] for p in self._raw_buf) / n
        # Skip near-duplicates to keep the path compact.
        if self._traced:
            lx, ly = self._traced[-1]
            if abs(sx - lx) < 0.003 and abs(sy - ly) < 0.003:
                return
        self._traced.append((sx, sy))

    # -- Verification (runs in exactly one frame) -------------------------

    def _run_verification(self) -> None:
        """Compute DTW similarity and transition to VERIFIED or FAILED."""
        now = time.time()
        if len(self._traced) < 10:
            self.state      = TracerState.FAILED
            self._result_at = now
            return

        t_rs   = _resample(list(self.template.waypoints), self.resample_n)
        u_rs   = _resample(self._traced,                  self.resample_n)
        t_norm = _centroid_normalise(t_rs)
        u_norm = _centroid_normalise(u_rs)

        cost             = dtw_normalised_cost(t_norm, u_norm)
        self.dtw_cost    = cost
        # Clamp: 1.0 at perfect match, 0.0 at cost == threshold, negative clamped to 0
        self.similarity  = max(0.0, min(1.0, 1.0 - cost / self.dtw_threshold))
        self.state       = TracerState.VERIFIED if cost <= self.dtw_threshold else TracerState.FAILED
        self._result_at  = now

    # -- Main update ------------------------------------------------------

    def update(self, hands: list[HandResult]) -> TracerState:
        """Feed one video frame.  Returns the current TracerState.

        Call this every frame; read ``state``, ``traced_path``, and
        ``template_waypoints`` to drive the HUD renderer.
        """
        now = time.time()

        # VERIFYING runs for exactly one frame then transitions.
        if self.state == TracerState.VERIFYING:
            self._run_verification()
            return self.state

        # Terminal states are sticky until reset() is called.
        if self.state in (TracerState.VERIFIED, TracerState.FAILED):
            return self.state

        # No hand detected.
        if not hands:
            if self.state == TracerState.DRAWING:
                if len(self._traced) >= 10:
                    self.state = TracerState.VERIFYING   # verify what was drawn
                else:
                    self._reset_drawing()                # too short, discard
            return self.state

        hand = hands[0]
        lm   = hand.landmarks

        # ── Depth liveness gate ───────────────────────────────────────
        # hand_scale = Dist(Wrist, MiddleMCP).  A printed photo or a
        # hand held very far from the camera produces a small value.
        if _hs(lm) < self.min_hand_scale:
            return self.state

        index_tip  = lm[_LM["INDEX_TIP"]]
        index_open = is_finger_open(lm, hand.handedness, Finger.INDEX)

        # ── WAITING ───────────────────────────────────────────────────
        if self.state == TracerState.WAITING:
            if index_open:
                self._reset_drawing()
                self._draw_start     = now
                self._was_index_open = True
                self.state           = TracerState.DRAWING
                self._push_point(index_tip.x, index_tip.y)

        # ── DRAWING ───────────────────────────────────────────────────
        elif self.state == TracerState.DRAWING:
            # Hard time-limit: verify whatever has been collected.
            if now - self._draw_start >= self.time_limit:
                self.state = TracerState.VERIFYING
                return self.state

            if index_open:
                self._push_point(index_tip.x, index_tip.y)
                self._was_index_open = True
            else:
                # Fist after a valid drawing stroke → end stroke, verify.
                if self._was_index_open and len(self._traced) >= 10:
                    self.state = TracerState.VERIFYING

        return self.state

    # -- Internals --------------------------------------------------------

    def _reset_drawing(self) -> None:
        self._traced.clear()
        self._raw_buf.clear()
        self._was_index_open = False
        self._draw_start     = None

    # -- Observable helpers for HUD ---------------------------------------

    @property
    def traced_path(self) -> list[tuple[float, float]]:
        """Current recorded trace in normalised screen coordinates."""
        return list(self._traced)

    @property
    def template_waypoints(self) -> list[tuple[float, float]]:
        return list(self.template.waypoints)

    @property
    def time_remaining(self) -> float:
        if self._draw_start is None:
            return self.time_limit
        return max(0.0, self.time_limit - (time.time() - self._draw_start))

    @property
    def draw_progress(self) -> float:
        """0.0 → 1.0 fraction of the time window used (for the ring timer)."""
        if self._draw_start is None:
            return 0.0
        return min((time.time() - self._draw_start) / self.time_limit, 1.0)

    @property
    def similarity_pct(self) -> float:
        """0 – 100 similarity percentage (only meaningful after VERIFYING)."""
        return self.similarity * 100.0

    @property
    def point_count(self) -> int:
        return len(self._traced)

    def reset(self) -> None:
        """Pick a new random shape and return to WAITING."""
        self.state      = TracerState.WAITING
        self.template   = generate_random_shape()
        self.similarity = 0.0
        self.dtw_cost   = 0.0
        self._result_at = None
        self._reset_drawing()
