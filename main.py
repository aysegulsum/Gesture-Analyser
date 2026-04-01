"""
Desktop Demo -- v5 Hardened with GameManager
=============================================
Run with:  python main.py

Four modes, cycle with **m** key:
  1. [Normal]    Show N fingers to match the prompt (1-10).
  2. [Math]      Solve arithmetic with fingers (60s timer).
  3. [Simon]     Memorise & replicate a gesture sequence.
  4. [Liveness]  Fast-response single-command challenge (4s window).

Press **r** to restart the current game.
Press **q** or **Esc** to quit.

Architecture
------------
HandTracker   -- detection only (hardened: 0.8 confidence, wrist re-labeling)
GameManager   -- owns all game sessions, handles mode switching and the
                 global "hand presence" guard that protects every mode
                 from running logic on empty frames.
main loop     -- capture + render only; no game logic here.
"""

import cv2
import numpy as np

from hand_tracker import HandTracker, HandResult, draw_landmarks
from gesture_session import GestureSession, SessionState
from math_session import MathSession, MathState
from simon_session import SimonSaysGame, SimonState
from liveness_session import LivenessChallenge, LivenessState, CmdType


# -- colour palette -------------------------------------------------------
WHITE = (255, 255, 255)
GREEN = (0, 220, 100)
YELLOW = (0, 220, 255)
RED = (60, 60, 220)
CYAN = (220, 220, 0)
MAGENTA = (200, 50, 255)
ORANGE = (0, 160, 255)
DARK_BG = (40, 40, 40)

MODE_NAMES = ["Normal", "Math", "Simon", "Liveness"]


# ── GameManager ─────────────────────────────────────────────────────

class GameManager:
    """Centralised mode switching + global hand-presence guard.

    Separates all game logic from the capture/render loop.
    """

    def __init__(self):
        self.modes = MODE_NAMES
        self.current_mode = 0

        self.gesture = GestureSession(
            stability_seconds=1.0, pause_after_success=2.0,
            targets=list(range(1, 11)), smoothing_window=7,
        )
        self.math = MathSession(
            stability_seconds=2.0, pause_after_success=1.5,
            game_duration=60.0, smoothing_window=7,
        )
        self.simon = SimonSaysGame(
            sequence_length=4, show_duration=2.0,
            hold_seconds=1.5, global_timeout=15.0, smoothing_window=7,
        )
        self.liveness = LivenessChallenge(
            time_limit=4.0, debounce_seconds=0.5,
            area_change_threshold=0.20, pause_after_result=1.5,
            smoothing_window=7,
        )

        self.hands_present = False

    @property
    def mode_name(self) -> str:
        return self.modes[self.current_mode]

    def cycle_mode(self):
        self.current_mode = (self.current_mode + 1) % len(self.modes)
        # Reset the newly-selected mode.
        if self.current_mode == 1:
            self.math.reset()
        elif self.current_mode == 2:
            self.simon.reset()
        elif self.current_mode == 3:
            self.liveness.reset()

    def restart(self):
        if self.current_mode == 0:
            pass  # Normal auto-cycles
        elif self.current_mode == 1:
            self.math.reset()
        elif self.current_mode == 2:
            self.simon.next_round()
        elif self.current_mode == 3:
            self.liveness.reset()

    def update(self, hands: list[HandResult]):
        """Global hand-presence check, then delegate to active mode.

        If no hands are detected, we set hands_present=False and pass
        an empty list to the active session.  Each session already
        handles empty hands (resets timers, shows "no hand" status).
        This centralised guard ensures consistent behaviour across all
        modes and prevents false timeouts or logic crashes.
        """
        self.hands_present = len(hands) > 0

        if self.current_mode == 0:
            self.gesture.update(hands)
        elif self.current_mode == 1:
            self.math.update(hands)
        elif self.current_mode == 2:
            self.simon.update(hands)
        else:
            self.liveness.update(hands)


# ── Drawing utilities ───────────────────────────────────────────────

def put_text_with_bg(frame, text, org, font_scale=0.8, color=WHITE, bg=DARK_BG, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    cv2.rectangle(frame, (x - 4, y - th - 6), (x + tw + 4, y + 6), bg, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)


def put_text_centered(frame, text, y, font_scale=1.2, color=WHITE, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    h, w = frame.shape[:2]
    x = (w - tw) // 2
    cv2.rectangle(frame, (x - 8, y - th - 8), (x + tw + 8, y + 8), DARK_BG, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)


def draw_progress_bar(frame, progress: float, y: int = 70, color_full=GREEN, color_fill=YELLOW):
    h, w = frame.shape[:2]
    bar_w = int(w * 0.4)
    x_start = (w - bar_w) // 2
    cv2.rectangle(frame, (x_start, y), (x_start + bar_w, y + 18), DARK_BG, -1)
    fill_w = int(bar_w * progress)
    color = color_full if progress >= 1.0 else color_fill
    cv2.rectangle(frame, (x_start, y), (x_start + fill_w, y + 18), color, -1)
    cv2.rectangle(frame, (x_start, y), (x_start + bar_w, y + 18), WHITE, 1)


def draw_countdown_ring(frame, remaining, total, cx, cy, radius=60):
    progress = max(0.0, remaining / total)
    angle = int(360 * progress)
    cv2.circle(frame, (cx, cy), radius, DARK_BG, -1)
    cv2.circle(frame, (cx, cy), radius, WHITE, 2)
    color = GREEN if remaining > 2 else YELLOW if remaining > 1 else RED
    if angle > 0:
        cv2.ellipse(frame, (cx, cy), (radius - 4, radius - 4), -90, 0, angle, color, 6)
    text = f"{remaining:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 1.0, 2)
    cv2.putText(frame, text, (cx - tw // 2, cy + th // 2), font, 1.0, color, 2)


# ── Per-mode HUD renderers ──────────────────────────────────────────

def draw_no_hand_overlay(frame):
    """Shown when no hands are detected -- replaces mode-specific status."""
    h, w = frame.shape[:2]
    put_text_centered(frame, "No Hand Detected", h // 2, font_scale=1.0, color=RED, thickness=2)


def draw_normal_hud(frame, gm: GameManager, hands):
    ses = gm.gesture
    state = ses.state
    prompt_color = GREEN if state == SessionState.VALIDATED else WHITE
    put_text_with_bg(frame, ses.display_text, (20, 40), font_scale=1.0, color=prompt_color, thickness=2)
    draw_progress_bar(frame, ses.hold_progress)

    status_color = GREEN if state == SessionState.VALIDATED else YELLOW if ses.hold_progress > 0 else RED
    put_text_with_bg(frame, ses.status_label, (20, 110), font_scale=0.7, color=status_color)

    per_hand = ses.per_hand_counts(hands)
    total = ses.detected_total(hands)
    y_off = 145
    for label in ("Left", "Right"):
        if label in per_hand:
            put_text_with_bg(frame, f"{label}: {per_hand[label]}", (20, y_off), font_scale=0.6)
            y_off += 32
    if total is not None:
        c = GREEN if total == ses.current_target else WHITE
        put_text_with_bg(frame, f"Total: {total}", (20, y_off), font_scale=0.7, color=c, thickness=2)


def draw_math_hud(frame, gm: GameManager, hands):
    ses = gm.math
    h, w = frame.shape[:2]
    ms = ses.state

    eq_color = RED if ms == MathState.GAME_OVER else GREEN if ms == MathState.SUCCESS else CYAN
    put_text_centered(frame, ses.equation_text, h // 2 - 30, font_scale=1.5, color=eq_color, thickness=3)

    if ms == MathState.GAME_OVER:
        put_text_centered(frame, ses.display_text, h // 2 + 30, font_scale=1.0, color=RED, thickness=2)
        put_text_centered(frame, "Press R to restart", h // 2 + 70, font_scale=0.7, color=YELLOW)
    elif ms == MathState.SUCCESS:
        put_text_centered(frame, "CORRECT!", h // 2 + 30, font_scale=1.0, color=GREEN, thickness=2)
    else:
        status_color = YELLOW if ms == MathState.HOLDING else WHITE
        put_text_with_bg(frame, ses.status_label, (20, 110), font_scale=0.7, color=status_color)
        draw_progress_bar(frame, ses.hold_progress)

    remaining = ses.time_remaining
    timer_color = RED if remaining < 10 else YELLOW if remaining < 30 else WHITE
    put_text_with_bg(frame, f"Time: {remaining:.0f}s", (w - 180, 40), font_scale=0.8, color=timer_color, thickness=2)
    put_text_with_bg(frame, f"Score: {ses.score}", (20, 40), font_scale=0.8, color=GREEN, thickness=2)

    per_hand = ses.per_hand_counts(hands)
    total = ses.detected_total()
    y_off = h - 80
    for label in ("Left", "Right"):
        if label in per_hand:
            put_text_with_bg(frame, f"{label}: {per_hand[label]}", (20, y_off), font_scale=0.6)
            y_off += 28
    if total is not None:
        c = GREEN if total == ses.answer else WHITE
        put_text_with_bg(frame, f"Total: {total}", (20, y_off), font_scale=0.7, color=c, thickness=2)


def draw_simon_hud(frame, gm: GameManager, hands):
    simon = gm.simon
    h, w = frame.shape[:2]
    ss = simon.state

    box_w, box_h, gap = 50, 30, 10
    total_w = len(simon.sequence) * (box_w + gap) - gap
    x_start = (w - total_w) // 2
    y_box = 15

    for i, _ in enumerate(simon.sequence):
        x = x_start + i * (box_w + gap)
        if ss == SimonState.SHOWING:
            bg_col = MAGENTA if i == simon._show_index else (80, 80, 80) if i < simon._show_index else (50, 50, 50)
        elif ss == SimonState.REPLICATING:
            bg_col = GREEN if i < simon.current_step else YELLOW if i == simon.current_step else (80, 80, 80)
        elif ss == SimonState.VERIFIED:
            bg_col = GREEN
        else:
            bg_col = RED if i >= simon.current_step else GREEN
        cv2.rectangle(frame, (x, y_box), (x + box_w, y_box + box_h), bg_col, -1)
        cv2.rectangle(frame, (x, y_box), (x + box_w, y_box + box_h), WHITE, 1)
        cv2.putText(frame, str(i + 1), (x + box_w // 2 - 5, y_box + box_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

    if ss == SimonState.SHOWING:
        g = simon.showing_gesture
        if g:
            put_text_centered(frame, f"MEMORISE: {g.name}", h // 2 - 20, font_scale=1.1, color=MAGENTA, thickness=2)
            draw_progress_bar(frame, simon.show_progress, y=h // 2 + 20, color_full=MAGENTA, color_fill=MAGENTA)
            put_text_centered(frame, f"Step {simon._show_index + 1} of {len(simon.sequence)}", h // 2 + 65, font_scale=0.7, color=WHITE)
    elif ss == SimonState.REPLICATING:
        g = simon.current_target_gesture
        if g:
            put_text_centered(frame, simon.display_text, h // 2 - 20, font_scale=1.0, color=YELLOW, thickness=2)
            draw_progress_bar(frame, simon.hold_progress, y=h // 2 + 20)
        status_color = YELLOW if simon.hold_progress > 0 else WHITE
        put_text_with_bg(frame, simon.status_label, (20, 80), font_scale=0.7, color=status_color)
        remaining = simon.time_remaining
        timer_color = RED if remaining < 5 else YELLOW if remaining < 10 else WHITE
        put_text_with_bg(frame, f"Time: {remaining:.1f}s", (w - 200, 80), font_scale=0.8, color=timer_color, thickness=2)
    elif ss == SimonState.VERIFIED:
        put_text_centered(frame, "LIVENESS VERIFIED!", h // 2 - 10, font_scale=1.3, color=GREEN, thickness=3)
        put_text_centered(frame, f"Rounds: {simon.rounds_completed}", h // 2 + 40, font_scale=0.8, color=WHITE)
        put_text_centered(frame, "Press R for next round", h // 2 + 80, font_scale=0.7, color=YELLOW)
    elif ss == SimonState.FAILED:
        put_text_centered(frame, "VERIFICATION FAILED", h // 2 - 10, font_scale=1.3, color=RED, thickness=3)
        put_text_centered(frame, "Press R to retry", h // 2 + 60, font_scale=0.7, color=YELLOW)

    y_off = h - 60
    for hand in hands:
        count = simon._validator.count_fingers(hand.landmarks, hand.handedness)
        put_text_with_bg(frame, f"{hand.handedness}: {count}", (20, y_off), font_scale=0.6)
        y_off += 28


def draw_liveness_hud(frame, gm: GameManager, hands):
    lv = gm.liveness
    h, w = frame.shape[:2]
    ls = lv.state

    if lv.is_flash_red:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    draw_countdown_ring(frame, lv.time_remaining, lv.time_limit, w - 90, 90, radius=55)
    put_text_with_bg(frame, f"Score: {lv.score}", (20, 40), font_scale=0.8, color=GREEN, thickness=2)
    if lv.streak > 1:
        put_text_with_bg(frame, f"Streak: {lv.streak}", (20, 75), font_scale=0.65, color=ORANGE)

    if ls == LivenessState.SUCCESS:
        put_text_centered(frame, "VERIFIED!", h // 2 - 20, font_scale=1.4, color=GREEN, thickness=3)
    elif ls == LivenessState.FAILED:
        put_text_centered(frame, "FAILED!", h // 2 - 20, font_scale=1.4, color=RED, thickness=3)
        put_text_centered(frame, "Next challenge incoming...", h // 2 + 30, font_scale=0.7, color=YELLOW)
    else:
        cmd_color = ORANGE if ls == LivenessState.DEBOUNCE else CYAN
        put_text_centered(frame, lv.command_label, h // 2 - 30, font_scale=1.0, color=cmd_color, thickness=2)
        if ls == LivenessState.DEBOUNCE:
            draw_progress_bar(frame, lv.debounce_progress, y=h // 2 + 10, color_full=GREEN, color_fill=ORANGE)
            put_text_centered(frame, "Confirming...", h // 2 + 50, font_scale=0.7, color=ORANGE)

    status_color = GREEN if ls == LivenessState.SUCCESS else RED if ls == LivenessState.FAILED else YELLOW
    put_text_with_bg(frame, lv.status_label, (20, h - 90), font_scale=0.7, color=status_color)

    # Depth meter for spatial commands
    if lv.current_cmd and lv.current_cmd.cmd_type in (CmdType.MOVE_CLOSER, CmdType.MOVE_AWAY):
        pct = lv.area_change_pct
        if pct is not None:
            meter_x, meter_top, meter_bot = 50, h // 2 - 80, h // 2 + 80
            meter_h = meter_bot - meter_top
            meter_mid = (meter_top + meter_bot) // 2
            cv2.rectangle(frame, (meter_x - 15, meter_top), (meter_x + 15, meter_bot), DARK_BG, -1)
            cv2.rectangle(frame, (meter_x - 15, meter_top), (meter_x + 15, meter_bot), WHITE, 1)
            cv2.line(frame, (meter_x - 20, meter_mid), (meter_x + 20, meter_mid), WHITE, 2)
            clamped = max(-50, min(50, pct))
            fill_px = int((clamped / 50.0) * (meter_h // 2))
            fill_col = GREEN if abs(pct) >= lv.area_change_threshold * 100 else YELLOW
            if fill_px > 0:
                cv2.rectangle(frame, (meter_x - 12, meter_mid - fill_px), (meter_x + 12, meter_mid), fill_col, -1)
            elif fill_px < 0:
                cv2.rectangle(frame, (meter_x - 12, meter_mid), (meter_x + 12, meter_mid - fill_px), fill_col, -1)
            put_text_with_bg(frame, f"{pct:+.0f}%", (meter_x - 25, meter_bot + 25), font_scale=0.55, color=fill_col)
            sign = "+" if lv.current_cmd.cmd_type == CmdType.MOVE_CLOSER else "-"
            put_text_with_bg(frame, f"Need: {sign}{int(lv.area_change_threshold * 100)}%", (meter_x - 35, meter_top - 10), font_scale=0.5, color=WHITE)

    per_hand = lv.per_hand_counts(hands)
    y_off = h - 55
    for label in ("Left", "Right"):
        if label in per_hand:
            put_text_with_bg(frame, f"{label}: {per_hand[label]}", (w - 180, y_off), font_scale=0.6)
            y_off += 28


_HUD_DRAWERS = [draw_normal_hud, draw_math_hud, draw_simon_hud, draw_liveness_hud]


# ── Main loop ───────────────────────────────────────────────────────

def run():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    tracker = HandTracker(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        max_lost_frames=2,
    )
    gm = GameManager()

    print("Gesture Validator v5 -- 'M' cycle modes, 'R' restart, 'Q' quit.")

    with tracker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)

            # -- Detection (separated from game logic) --------------------
            hands = tracker.process(frame)

            # -- Draw landmarks -------------------------------------------
            for hand in hands:
                draw_landmarks(frame, hand)

            # -- Game logic (delegated to GameManager) --------------------
            gm.update(hands)

            # -- HUD rendering --------------------------------------------
            if not gm.hands_present:
                draw_no_hand_overlay(frame)
            _HUD_DRAWERS[gm.current_mode](frame, gm, hands)

            # Mode indicator (bottom-right)
            fh, fw = frame.shape[:2]
            put_text_with_bg(frame, f"[M] {gm.mode_name}", (fw - 220, fh - 20), font_scale=0.55, color=CYAN)

            cv2.imshow("Gesture Validator", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("m"):
                gm.cycle_mode()
                print(f"Mode: {gm.mode_name}")
            elif key == ord("r"):
                gm.restart()
                print(f"{gm.mode_name} restarted!")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
